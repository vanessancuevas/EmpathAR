"""
Re-export dog-pose model at 320x320 input + quantize to INT8.
No retraining — loads best.pt weights directly.
Output: models/dog-pose-int8-320.onnx
"""
import os, shutil
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

SETTINGS.update({
    'datasets_dir': '/Volumes/Vans1/ultralytics_datasets',
    'runs_dir':     '/Volumes/Vans1/ultralytics_runs',
})

BEST_PT   = '/Volumes/Vans1/ultralytics_runs/dog-pose-nano/weights/best.pt'
REPO_DIR  = os.path.dirname(__file__)
FP32_ONNX = os.path.join(REPO_DIR, 'models', 'dog-pose-fp32-320.onnx')
INT8_ONNX = os.path.join(REPO_DIR, 'models', 'dog-pose-int8-320.onnx')

print("=== Step 1: export best.pt → ONNX at 320×320 ===")
model = YOLO(BEST_PT)
model.export(format='onnx', imgsz=320, opset=12, simplify=True)

exported = '/Volumes/Vans1/ultralytics_runs/dog-pose-nano/weights/best.onnx'
shutil.copy(exported, FP32_ONNX)
print(f"FP32 saved → {FP32_ONNX}  ({os.path.getsize(FP32_ONNX)/1024/1024:.1f} MB)")

print("\n=== Step 2: quantize to INT8 ===")
quantize_dynamic(
    model_input=FP32_ONNX,
    model_output=INT8_ONNX,
    weight_type=QuantType.QUInt8,
    per_channel=False,
    reduce_range=False,
)
print(f"INT8 saved → {INT8_ONNX}  ({os.path.getsize(INT8_ONNX)/1024/1024:.1f} MB)")

print("\n=== Step 3: verify tensor shapes ===")
m = onnx.load(INT8_ONNX)
for inp in m.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  Input  '{inp.name}': {shape}")
for out in m.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  Output '{out.name}': {shape}")

os.remove(FP32_ONNX)
print("\nDone. FP32 intermediate removed.")
