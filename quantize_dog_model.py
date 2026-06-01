"""
Quantize dog-pose.onnx from FP32 to INT8 (dynamic).
- No calibration data needed
- Weights are INT8; activations stay FP32 at runtime
- JS inference code is unchanged (still passes float32 tensors)
- Typical size reduction: ~75% (14 MB → ~3-4 MB)
"""
import os
import shutil
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

REPO_DIR  = os.path.dirname(__file__)
SRC       = os.path.join(REPO_DIR, 'models', 'dog-pose.onnx')
DST       = os.path.join(REPO_DIR, 'models', 'dog-pose-int8.onnx')

print(f"Source:  {SRC}")
print(f"Output:  {DST}")
src_mb = os.path.getsize(SRC) / 1024 / 1024
print(f"Source size: {src_mb:.1f} MB")

quantize_dynamic(
    model_input=SRC,
    model_output=DST,
    weight_type=QuantType.QUInt8,   # unsigned int8 — best for WebGPU/WASM
    per_channel=False,               # simpler, works well for YOLOv8-nano
    reduce_range=False,              # keep full range for WebGPU
)

dst_mb = os.path.getsize(DST) / 1024 / 1024
print(f"\nOutput size: {dst_mb:.1f} MB  ({100 * dst_mb / src_mb:.0f}% of original)")

# Sanity-check: confirm input/output shapes are unchanged
m = onnx.load(DST)
print("\nTensor shapes (quantized model):")
for inp in m.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  Input  '{inp.name}': {shape}")
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  Output '{out.name}': {shape}")

if dst_mb > 10:
    print(f"\nWARNING: still > 10 MB — do NOT git add yet, investigate further.")
else:
    print(f"\nOK — under 10 MB, safe to swap into the app.")
    print("\nNext steps:")
    print("  1. Update DOG_MODEL_URL in index.html → '/models/dog-pose-int8.onnx'")
    print("  2. npx serve . and test Dog Mode end-to-end")
    print("  3. Commit both the new model and the URL change")
