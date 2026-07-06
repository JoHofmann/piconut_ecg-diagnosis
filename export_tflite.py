import argparse
import os
import subprocess

import numpy as np
import torch

from dataset import ECGDataset
from resnet import resnet34
from utils import split_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data (must match training)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth checkpoint')
    parser.add_argument('--onnx-path', type=str, default='models/model.onnx', help='Where to write the intermediate ONNX file')
    parser.add_argument('--output-dir', type=str, default='models/tflite_int8', help='onnx2tf output folder')
    parser.add_argument('--num-calib-samples', type=int, default=100, help='Number of samples used for int8 calibration')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cpu'  # export/quantization only needs CPU

    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)

    data_dir = os.path.normpath(args.data_dir)
    label_csv = os.path.join(data_dir, 'labels.csv')

    # Reuse the same fold split as training so calibration data matches
    # what the model actually saw during training.
    train_folds, _, _ = split_data(seed=args.seed)
    calib_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)

    sample_data, _ = calib_dataset[0]
    seq_len = sample_data.shape[-1]
    print(f'Detected input shape: (1, {nleads}, {seq_len})')

    # --- 1. Load the trained PyTorch model ---
    net = resnet34(input_channels=nleads)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    # --- 2. Export to ONNX with a fixed input shape ---
    os.makedirs(os.path.dirname(args.onnx_path) or '.', exist_ok=True)
    dummy_input = torch.randn(1, nleads, seq_len)
    torch.onnx.export(
        net,
        dummy_input,
        args.onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        do_constant_folding=True,
        # Force the legacy TorchScript-based exporter. The newer dynamo-based
        # exporter (default in recent PyTorch) doesn't have a decomposition
        # for aten.adaptive_max_pool2d yet, which resnet.py relies on.
        dynamo=False,
        # Fixed shapes on purpose: TFLite full-integer quantization needs
        # static shapes, and we're avoiding extra graph fusing/complexity.
    )
    print(f'Exported ONNX model to {args.onnx_path}')

    # --- 2b. Simplify the ONNX graph ---
    # resnet.py implements adaptive pooling via an unsqueeze + adaptive_max_pool2d
    # trick, which confuses onnx2tf's automatic NCHW->NHWC axis conversion and
    # produces reshape nodes with stale hardcoded shapes. Running the graph
    # through onnx-simplifier folds/cleans this up before onnx2tf sees it.
    import onnx
    from onnxsim import simplify

    onnx_model = onnx.load(args.onnx_path)
    simplified_model, check = simplify(
        onnx_model,
        overwrite_input_shapes={'input': [1, nleads, seq_len]},
    )
    if not check:
        raise RuntimeError('onnx-simplifier could not validate the simplified model')

    simplified_path = args.onnx_path.replace('.onnx', '_simplified.onnx')
    onnx.save(simplified_model, simplified_path)
    args.onnx_path = simplified_path
    print(f'Simplified ONNX model saved to {simplified_path}')

    # --- 3. Build calibration samples (kept as PyTorch-layout numpy arrays for now) ---
    num_samples = min(args.num_calib_samples, len(calib_dataset))
    calib_samples = []
    for i in range(num_samples):
        data, _ = calib_dataset[i]
        calib_samples.append(data.numpy().astype(np.float32))  # each: (nleads, seq_len)
    print(f'Prepared {len(calib_samples)} calibration samples')

    # --- 4. Convert ONNX -> plain (float32) SavedModel with onnx2tf ---
    # Deliberately NOT using onnx2tf's own -oiqt/-cind quantization flags here.
    # Those expect calibration .npy files in a very specific layout, and
    # silently mis-reading that layout is what produced the earlier
    # "num_input_elements != num_output_elements" crash. The plain conversion
    # below is the part that has been working reliably in your logs, so we
    # use it just to get a correct SavedModel and do quantization ourselves.
    os.makedirs(args.output_dir, exist_ok=True)
    cmd = ['onnx2tf', '-i', args.onnx_path, '-o', args.output_dir]
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError('onnx2tf conversion failed, see output above for details')

    # --- 5. Quantize to int8 ourselves with TFLiteConverter ---
    import tensorflow as tf

    saved_model_dir = args.output_dir
    loaded = tf.saved_model.load(saved_model_dir)
    infer = loaded.signatures['serving_default']
    input_specs = infer.structured_input_signature[1]
    input_name = list(input_specs.keys())[0]
    model_input_shape = list(input_specs[input_name].shape)  # e.g. [1, seq_len, nleads] or [1, nleads, seq_len]
    print(f'SavedModel expects input "{input_name}" with shape {model_input_shape}')

    # onnx2tf commonly converts 1D-conv models to a channels-last graph
    # internally (batch, sequence, channels) even though the ONNX model was
    # channels-first (batch, channels, sequence). Detect which layout the
    # real SavedModel signature wants and transpose our calibration samples
    # to match, instead of assuming one or the other.
    if model_input_shape[1] == nleads and model_input_shape[2] == seq_len:
        transpose_needed = False
    elif model_input_shape[1] == seq_len and model_input_shape[2] == nleads:
        transpose_needed = True
    else:
        raise RuntimeError(
            f'Could not match SavedModel input shape {model_input_shape} '
            f'against known layout (nleads={nleads}, seq_len={seq_len}).'
        )

    def representative_dataset_gen():
        for sample in calib_samples:  # sample shape: (nleads, seq_len)
            arr = sample.T if transpose_needed else sample  # (seq_len, nleads) or (nleads, seq_len)
            arr = np.expand_dims(arr, axis=0)  # add batch dim
            yield [arr.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Full-integer quantization: every op in TFLITE_BUILTINS_INT8 forced to int8,
    # no special fusing/graph surgery beyond TF's own standard quantizer.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    tflite_path = os.path.join(args.output_dir, 'model_int8.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f'Done. int8 quantized model written to {tflite_path}')


if __name__ == '__main__':
    main()
