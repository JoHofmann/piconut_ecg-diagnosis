#!/usr/bin/env python3
import argparse
import glob
import os
import re
from typing import List, Tuple

import numpy as np
import wfdb

try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore


ALL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def parse_args():
    p = argparse.ArgumentParser(
        description="Export CPSC ECG records to C uint8_t arrays using TFLite int8 quantization."
    )
    p.add_argument("--data-dir", default="data/CPSC")
    p.add_argument("--model", required=True, help="Path to int8 .tflite model")
    p.add_argument("--out", default="ecg_classification_input_data.c")
    p.add_argument("--prefix", default="ecg_classification_input_data")
    p.add_argument("--count", type=int, default=25, help="If --indices omitted, export first N records")
    p.add_argument("--indices", default=None, help="Example: 0,1,2,24 or 0-24")
    p.add_argument("--record-ids", default=None, help="Comma-separated record ids, e.g. A0001,A0002")
    p.add_argument("--target-len", type=int, default=15000)
    p.add_argument("--line-width", type=int, default=12)
    return p.parse_args()


def parse_indices(expr: str) -> List[int]:
    out = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def discover_records(data_dir: str) -> List[str]:
    recs = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(data_dir, "*.hea")))
    if not recs:
        raise FileNotFoundError(f"No WFDB records (*.hea) found in {data_dir}")
    return recs


def select_records(all_records: List[str], args) -> List[Tuple[int, str]]:
    if args.record_ids:
        ids = [x.strip() for x in args.record_ids.split(",") if x.strip()]
        missing = [x for x in ids if x not in all_records]
        if missing:
            raise ValueError(f"Unknown record ids: {missing}")
        return list(enumerate(ids))

    if args.indices:
        idxs = parse_indices(args.indices)
    else:
        idxs = list(range(min(args.count, len(all_records))))

    out = []
    for out_i, rec_idx in enumerate(idxs):
        if rec_idx < 0 or rec_idx >= len(all_records):
            raise IndexError(f"Record index {rec_idx} out of range")
        out.append((out_i, all_records[rec_idx]))
    return out


def get_input_quant(model_path: str):
    itp = Interpreter(model_path=model_path)
    itp.allocate_tensors()
    d = itp.get_input_details()[0]

    if d["dtype"] != np.int8:
        raise RuntimeError(f"Expected int8 input model, got {d['dtype']}")

    q = d.get("quantization_parameters", {})
    scales = q.get("scales", np.array([], dtype=np.float32))
    zps = q.get("zero_points", np.array([], dtype=np.int32))

    if len(scales) == 0 or len(zps) == 0:
        scale, zp = d.get("quantization", (0.0, 0))
        scales = np.array([scale], dtype=np.float32)
        zps = np.array([zp], dtype=np.int32)

    if len(scales) != 1 or len(zps) != 1:
        raise RuntimeError("Expected per-tensor quantization for model input")

    shape = d["shape"].tolist()  # expected [1,15000,12]
    return float(scales[0]), int(zps[0]), shape, d


def preprocess(sig: np.ndarray, target_len: int) -> np.ndarray:
    # Use all 12 leads in canonical order, match dataset semantics but output [15000,12]
    x = sig[:, :12].astype(np.float32)
    x = x[-target_len:, :]
    out = np.zeros((target_len, 12), dtype=np.float32)
    out[-x.shape[0]:, :] = x
    return out  # [15000,12]


def quantize_int8(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(x / scale + zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def bytes_as_u8(q_int8: np.ndarray) -> np.ndarray:
    return q_int8.view(np.uint8)


def arr_name(prefix: str, i: int) -> str:
    return f"{prefix}_{i:06d}"


def fmt_hex(data_u8: np.ndarray, per_line: int) -> str:
    vals = [f"0x{v:02x}" for v in data_u8.tolist()]
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(vals[i:i + per_line]))
    return ",\n".join(lines)


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.:/-]+", "_", s)


def main():
    args = parse_args()
    scale, zp, in_shape, in_detail = get_input_quant(args.model)

    # Strict shape expectation per your note: int8[1,15000,12]
    if in_shape != [1, args.target_len, 12]:
        raise RuntimeError(
            f"Model input shape is {in_shape}, expected [1,{args.target_len},12]. "
            "Adjust --target-len or model."
        )

    records = discover_records(args.data_dir)
    selected = select_records(records, args)

    exported = []  # (out_i, rec_id, u8_flat)
    for out_i, rec_id in selected:
        sig, _ = wfdb.rdsamp(os.path.join(args.data_dir, rec_id))   # [N,12] float
        x = preprocess(sig, args.target_len)                        # [15000,12] float
        q = quantize_int8(x, scale, zp)                             # [15000,12] int8

        # Flatten in row-major => t0c0,t0c1,...,t1c0,... (matches [1,15000,12])
        flat_u8 = bytes_as_u8(q.reshape(-1))
        exported.append((out_i, rec_id, flat_u8))

    lines = []
    lines.append("/* Auto-generated by export_cpsc_to_c_int8.py */")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"/* model: {sanitize(args.model)} */")
    lines.append(f"/* input dtype: int8, shape: {in_shape}, scale: {scale:.10g}, zero_point: {zp} */")
    lines.append("")

    names = []
    lens = []

    for out_i, rec_id, u8 in exported:
        n = arr_name(args.prefix, out_i)
        names.append(n)
        lens.append(f"{n}_len")
        lines.append(f"/* record_id: {sanitize(rec_id)} */")
        lines.append(f"const uint8_t {n}[] = {{")
        lines.append(fmt_hex(u8, args.line_width))
        lines.append("};")
        lines.append(f"const size_t {n}_len = {len(u8)};")
        lines.append("")

    lines.append(f"const uint8_t* {args.prefix}[] = {{{', '.join(names)}}};")
    lines.append("")
    lines.append(f"const size_t {args.prefix}_len[] = {{{', '.join(lens)}}};")
    lines.append(f"const size_t {args.prefix}_count = {len(names)};")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out}")
    for out_i, rec_id, u8 in exported:
        print(f"  {arr_name(args.prefix, out_i)} <- {rec_id} ({len(u8)} bytes)")


if __name__ == "__main__":
    main()
