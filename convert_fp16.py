"""
Convert FP32 ONNX models to FP16.
FP16 halves weight size (~6.5MB each) and uses no unsupported ops in ort-web WASM.
INT8 dynamic quantization used DynamicQuantizeLinear which ort-web doesn't support.
"""
import os
import onnx
from onnxconverter_common import float16

REPO_DIR = os.path.dirname(__file__)

models = [
    ('models/dog-pose-fp32-320.onnx',  'models/dog-pose-fp16-320.onnx'),
    ('models/dog-detect-fp32-320.onnx', 'models/dog-detect-fp16-320.onnx'),
]

# We need the FP32 source models — re-export them first if they don't exist
import shutil
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({'runs_dir': '/Volumes/Vans1/ultralytics_runs'})

def ensure_fp32(fp32_path, source, imgsz, is_detect=False):
    if os.path.exists(fp32_path):
        return
    print(f"  Re-exporting {source} at {imgsz}px...")
    model = YOLO(source)
    model.export(format='onnx', imgsz=imgsz, opset=12, simplify=True)
    exported_name = 'yolov8n.onnx' if is_detect else 'best.onnx'
    exported = os.path.join(os.path.dirname(source), exported_name) if not is_detect else exported_name
    shutil.copy(exported, fp32_path)

ensure_fp32(
    os.path.join(REPO_DIR, 'models/dog-pose-fp32-320.onnx'),
    '/Volumes/Vans1/ultralytics_runs/dog-pose-nano/weights/best.pt',
    320,
)
ensure_fp32(
    os.path.join(REPO_DIR, 'models/dog-detect-fp32-320.onnx'),
    'yolov8n.pt',
    320,
    is_detect=True,
)

for src_name, dst_name in models:
    src = os.path.join(REPO_DIR, src_name)
    dst = os.path.join(REPO_DIR, dst_name)
    print(f"\nConverting {src_name} → {dst_name}")
    model = onnx.load(src)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, dst)
    src_mb = os.path.getsize(src) / 1024 / 1024
    dst_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"  {src_mb:.1f} MB → {dst_mb:.1f} MB")

print("\nDone. Verify shapes:")
for _, dst_name in models:
    m = onnx.load(os.path.join(REPO_DIR, dst_name))
    for inp in m.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {dst_name} input: {shape} (dtype {inp.type.tensor_type.elem_type})")
