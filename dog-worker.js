// Dog inference Web Worker — runs ORT entirely off the main thread.
// Main thread sends: { type: 'init' }
//                    { type: 'infer', bitmap: ImageBitmap, vw: number, vh: number }
// Worker posts back: { type: 'ready' }
//                    { type: 'result', dogs: Detection[], lb: Letterbox }
//                    { type: 'error', message: string }

const ORT_CDN     = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js';
const MODEL_URL   = '/models/dog-pose-int8.onnx';
const INPUT_SIZE  = 640;
const NUM_KP      = 24;
const CONF_THRESH = 0.5;
const NMS_IOU     = 0.45;

let session = null;
const offscreen = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
const octx      = offscreen.getContext('2d');

function iou(a, b) {
  const ax1 = a.bbox.cx - a.bbox.w / 2, ay1 = a.bbox.cy - a.bbox.h / 2;
  const ax2 = a.bbox.cx + a.bbox.w / 2, ay2 = a.bbox.cy + a.bbox.h / 2;
  const bx1 = b.bbox.cx - b.bbox.w / 2, by1 = b.bbox.cy - b.bbox.h / 2;
  const bx2 = b.bbox.cx + b.bbox.w / 2, by2 = b.bbox.cy + b.bbox.h / 2;
  const ix    = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const iy    = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = ix * iy;
  const union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter;
  return union > 0 ? inter / union : 0;
}

async function init() {
  importScripts(ORT_CDN);
  // Point WASM files at CDN so they resolve correctly from the worker context
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu', 'webgl', 'wasm'],
    });
    postMessage({ type: 'ready' });
  } catch (e) {
    postMessage({ type: 'error', message: e.message });
  }
}

async function infer(bitmap, vw, vh) {
  if (!session) { bitmap.close(); return; }

  // Letterbox: fit video into INPUT_SIZE×INPUT_SIZE preserving aspect ratio
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const newW  = Math.round(vw * scale);
  const newH  = Math.round(vh * scale);
  const padX  = (INPUT_SIZE - newW) / 2;
  const padY  = (INPUT_SIZE - newH) / 2;

  octx.fillStyle = '#808080';
  octx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  octx.drawImage(bitmap, padX, padY, newW, newH);
  bitmap.close();

  // Build CHW Float32 tensor normalised to [0,1]
  const pixels = octx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const N      = INPUT_SIZE * INPUT_SIZE;
  const tensor = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    tensor[i]         = pixels[i * 4]     / 255;
    tensor[i + N]     = pixels[i * 4 + 1] / 255;
    tensor[i + N * 2] = pixels[i * 4 + 2] / 255;
  }

  const feed   = { images: new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
  const result = await session.run(feed);
  const out    = result['output0'].data;
  const A      = 8400;

  // Parse raw detections
  const raw = [];
  for (let i = 0; i < A; i++) {
    const conf = out[4 * A + i];
    if (conf < CONF_THRESH) continue;
    const keypoints = [];
    for (let k = 0; k < NUM_KP; k++) {
      const base = (5 + k * 3) * A + i;
      keypoints.push({ x: out[base], y: out[base + A], conf: out[base + A * 2] });
    }
    raw.push({ bbox: { cx: out[0*A+i], cy: out[1*A+i], w: out[2*A+i], h: out[3*A+i], conf }, keypoints });
  }

  // NMS
  raw.sort((a, b) => b.bbox.conf - a.bbox.conf);
  const kept = [];
  for (const det of raw) {
    if (!kept.some(k => iou(det, k) > NMS_IOU)) kept.push(det);
  }

  postMessage({ type: 'result', dogs: kept, lb: { scale, padX, padY } });
}

self.onmessage = async ({ data }) => {
  if (data.type === 'init')  await init();
  if (data.type === 'infer') await infer(data.bitmap, data.vw, data.vh);
};
