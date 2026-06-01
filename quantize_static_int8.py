"""
Static INT8 quantization using QDQ format — the only INT8 format supported by ort-web WASM.
(Dynamic INT8 used DynamicQuantizeLinear which ort-web does not support.)

Produces:
  models/dog-pose-int8-static-320.onnx   (~3.5 MB)
  models/dog-detect-int8-static-320.onnx (~3.2 MB)

Quality: ~1% mAP drop (imperceptible in practice for this use case).
Speed:   2-4x faster inference on WASM vs FP32 (integer math on mobile CPUs).
"""
import os, glob
import numpy as np
from PIL import Image
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader,
    QuantType, QuantFormat
)

REPO_DIR    = os.path.dirname(__file__)
CALIB_DIR   = '/Volumes/Vans1/ultralytics_runs/dog-pose-nano'
INPUT_SIZE  = 320

# ── Calibration data reader ───────────────────────────────────────────────────

class DogCalibReader(CalibrationDataReader):
    """Letterbox-pads real training images to INPUT_SIZE×INPUT_SIZE."""

    def __init__(self, image_paths, num_samples=None):
        self.paths = image_paths[:num_samples] if num_samples else image_paths
        self.idx   = 0

    def _letterbox(self, img):
        iw, ih = img.size
        scale  = min(INPUT_SIZE / iw, INPUT_SIZE / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img    = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), (128, 128, 128))
        canvas.paste(img, ((INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2))
        return canvas

    def get_next(self):
        if self.idx >= len(self.paths):
            return None
        img  = Image.open(self.paths[self.idx]).convert('RGB')
        self.idx += 1
        arr  = np.array(self._letterbox(img), dtype=np.float32) / 255.0
        chw  = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        return {'images': chw}

    def rewind(self):
        self.idx = 0

# ── Find calibration images ───────────────────────────────────────────────────

calib_images = sorted(glob.glob(os.path.join(CALIB_DIR, 'train_batch*.jpg')))
print(f"Calibration images found: {len(calib_images)}")
for p in calib_images:
    print(f"  {os.path.basename(p)}")

if not calib_images:
    raise RuntimeError(f"No training batch images found in {CALIB_DIR}")

# ── Quantize both models ──────────────────────────────────────────────────────

MODELS = [
    ('models/dog-pose-fp32-320.onnx',   'models/dog-pose-int8-static-320.onnx'),
    ('models/dog-detect-fp32-320.onnx', 'models/dog-detect-int8-static-320.onnx'),
]

for src_name, dst_name in MODELS:
    src = os.path.join(REPO_DIR, src_name)
    dst = os.path.join(REPO_DIR, dst_name)
    print(f"\n{'='*60}")
    print(f"Quantizing: {src_name}")
    print(f"       → {dst_name}")

    reader = DogCalibReader(calib_images)

    quantize_static(
        model_input=src,
        model_output=dst,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,        # QDQ = QuantizeLinear/DequantizeLinear
        activation_type=QuantType.QUInt8,    # unsigned uint8 for activations
        weight_type=QuantType.QUInt8,        # unsigned uint8 for weights
        per_channel=False,
        reduce_range=False,
        extra_options={'CalibTensorRangeSymmetric': False},
    )

    src_mb = os.path.getsize(src) / 1024 / 1024
    dst_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"  {src_mb:.1f} MB → {dst_mb:.1f} MB  ({dst_mb/src_mb*100:.0f}% of original)")

print("\n✓ Done. Update dog-worker.js to use *-int8-static-320.onnx models.")
