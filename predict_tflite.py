import argparse
import os

import numpy as np
import scipy.io as sio
import tensorflow as tf
# from ai_edge_litert.interpreter import Interpreter, OpResolverType


def parse_header(hea_path):
    """Parse a WFDB .hea header file for per-lead gain/baseline metadata."""
    with open(hea_path, 'r') as f:
        lines = f.read().splitlines()

    record_line = lines[0].split()
    n_sig = int(record_line[1])
    fs = float(record_line[2])

    gains, baselines = [], []
    for i in range(1, n_sig + 1):
        fields = lines[i].split()
        # WFDB signal-line format: filename format gain(baseline)/units ...
        gain_field = fields[2]
        if '(' in gain_field:
            gain_str, rest = gain_field.split('(', 1)
            baseline_str = rest.split(')')[0]
            baseline = float(baseline_str)
        else:
            gain_str = gain_field.split('/')[0]  # e.g. "1000/mV" -> "1000"
            baseline = 0.0
        gain = float(gain_str)
        gains.append(gain)
        baselines.append(baseline)

    return {
        'n_sig': n_sig,
        'fs': fs,
        'gains': np.array(gains, dtype=np.float32),
        'baselines': np.array(baselines, dtype=np.float32),
    }


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_ecg(mat_path, hea_path, leads):
    """Mirrors ECGDataset.__getitem__ in dataset.py exactly (minus training-only augmentation)."""
    header = parse_header(hea_path)
    mat = sio.loadmat(mat_path)
    raw = mat['val'].astype(np.float64)  # (n_sig_in_file, n_samples), raw ADC units

    gains = header['gains'].reshape(-1, 1)
    baselines = header['baselines'].reshape(-1, 1)
    physical = (raw - baselines) / gains  # physical units (mV), matches what wfdb.rdsamp returns
    ecg_data = physical.T  # (n_samples, n_sig), time-major -- same convention as wfdb.rdsamp's output

    if leads == 'all':
        use_leads = np.arange(len(LEAD_NAMES))
    else:
        use_leads = np.where(np.isin(LEAD_NAMES, leads))[0]
    nleads = len(use_leads)

    # Exactly mirrors dataset.py: nsteps computed BEFORE cropping, right-aligned
    # crop for long signals, zero-pad at the front for short ones.
    nsteps, _ = ecg_data.shape
    ecg_data = ecg_data[-15000:, use_leads]
    result = np.zeros((15000, nleads))
    result[-nsteps:, :] = ecg_data

    return result.transpose()  # (nleads, 15000) -- matches the network's expected input layout


def run_inference(tflite_path, signal):
    print("--> Initializing Interpreter...")
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )

    # Get details BEFORE allocation so we can inspect and fix dynamic shapes
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    expected_shape = input_details['shape']
    nleads, seq_len = signal.shape

    # Determine structural layout
    if expected_shape[1] == seq_len or expected_shape[1] in [0, -1]:
        # Channels-last or dynamic batch configuration
        model_input = signal.T  # (seq_len, nleads)
    elif expected_shape[1] == nleads:
        model_input = signal  # (nleads, seq_len)
    else:
        model_input = signal.T # Default fallback

    model_input = np.expand_dims(model_input, axis=0)
    target_shape = list(model_input.shape)

    print(f"--> Model expected shape layout: {expected_shape}")
    print(f"--> Overriding/Resizing input tensor to exact shape: {target_shape}")

    # FORCE the interpreter to recognize the exact shape to prevent 0-byte allocation crashes
    interpreter.resize_tensor_input(input_details['index'], target_shape)

    print("--> Allocating tensors...")
    interpreter.allocate_tensors()

    # Quantize data if the model is INT8
    if input_details['dtype'] == np.int8:
        print("--> Quantizing input data to INT8...")
        scale, zero_point = input_details['quantization']
        # Ensure values stay clipped within valid int8 boundaries [-128, 127]
        model_input = np.clip(np.round(model_input / scale + zero_point), -128, 127)
        model_input = model_input.astype(np.int8)
    else:
        model_input = model_input.astype(input_details['dtype'])

    print("--> Loading input tensor...")
    interpreter.set_tensor(input_details['index'], model_input)

    print("--> Invoking interpreter (Running inference)...")
    interpreter.invoke()

    print("--> Reading output tensor...")
    output = interpreter.get_tensor(output_details['index'])

    if output_details['dtype'] == np.int8:
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    print("--> Inference completed successfully!")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite-path', type=str, required=True, help='Path to the .tflite model')
    parser.add_argument('--mat-path', type=str, required=True, help='Path to the record .mat file')
    parser.add_argument('--hea-path', type=str, default=None,
                         help='Path to the record .hea file (defaults to same basename as --mat-path)')
    parser.add_argument('--leads', type=str, default='all',
                         help="'all' or a comma-separated subset of I,II,III,aVR,aVL,aVF,V1..V6 (must match training)")
    args = parser.parse_args()

    hea_path = args.hea_path or os.path.splitext(args.mat_path)[0] + '.hea'
    leads = 'all' if args.leads == 'all' else args.leads.split(',')

    signal = load_ecg(args.mat_path, hea_path, leads)
    logits = run_inference(args.tflite_path, signal)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid, matching BCEWithLogitsLoss training

    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    print('Logits:', logits)
    print('Class probabilities:')
    for cls, p in zip(classes, probs.flatten()):
        print(f'  {cls}: {p:.4f}')


if __name__ == '__main__':
    main()
