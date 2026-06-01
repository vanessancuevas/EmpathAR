// Dog inference Web Worker — two-stage pipeline mirroring MediaPipe's architecture:
//   Stage 1 (cheap, full frame): COCO detector finds dog bbox
//   Stage 2 (precise, crop):     pose model runs only on the detected region
//   Tracking mode: Stage 1 skipped every REDETECT_N frames once a dog is found
//
// Main thread sends: { type: 'init' }
//                    { type: 'infer', bitmap: ImageBitmap, vw: number, vh: number }
// Worker posts back: { type: 'ready' }
//                    { type: 'result', dogs: Detection[], lb: Letterbox }
//                    { type: 'error', message: string }

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';

const MODEL_DETECT = '/models/dog-detect-fp32-320.onnx';
const MODEL_POSE   = '/models/dog-pose-fp32-320.onnx';
const INPUT_SIZE   = 320;
const DOG_CLASS    = 16;     // COCO class index for dog
const NUM_KP       = 24;
const DETECT_CONF  = 0.25;
const POSE_CONF    = 0.35;
const NMS_IOU      = 0.45;
const CROP_MARGIN  = 0.35;   // padding around detected bbox before cropping
const REDETECT_N   = 5;      // run Stage 1 every N pose inferences

let detectSession  = null;
let poseSession    = null;
let lastBbox       = null;   // { x1,y1,x2,y2 } in video-pixel space
let redetectCount  = 0;

const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
const octx   = canvas.getContext('2d');

// ── helpers ───────────────────────────────────────────────────────────────────

function letterboxFull(bmp, vw, vh) {
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const newW  = Math.round(vw * scale);
  const newH  = Math.round(vh * scale);
  const padX  = (INPUT_SIZE - newW) / 2;
  const padY  = (INPUT_SIZE - newH) / 2;
  octx.fillStyle = '#808080';
  octx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  octx.drawImage(bmp, 0, 0, vw, vh, padX, padY, newW, newH);
  return { scale, padX, padY };
}

function letterboxCrop(bmp, x1, y1, cropW, cropH) {
  const scale = Math.min(INPUT_SIZE / cropW, INPUT_SIZE / cropH);
  const newW  = Math.round(cropW * scale);
  const newH  = Math.round(cropH * scale);
  const padX  = (INPUT_SIZE - newW) / 2;
  const padY  = (INPUT_SIZE - newH) / 2;
  octx.fillStyle = '#808080';
  octx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  octx.drawImage(bmp, x1, y1, cropW, cropH, padX, padY, newW, newH);
  return { scale, padX, padY };
}

function buildTensor() {
  const pixels = octx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const N      = INPUT_SIZE * INPUT_SIZE;
  const tensor = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    tensor[i]         = pixels[i * 4]     / 255;
    tensor[i + N]     = pixels[i * 4 + 1] / 255;
    tensor[i + N * 2] = pixels[i * 4 + 2] / 255;
  }
  return new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
}

function iouRect(a, b) {
  const ix    = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy    = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

// ── init ─────────────────────────────────────────────────────────────────────

async function init() {
  try {
    detectSession = await ort.InferenceSession.create(MODEL_DETECT, { executionProviders: ['wasm'] });
    console.log('[dog-worker] detect session ready');
    poseSession   = await ort.InferenceSession.create(MODEL_POSE,   { executionProviders: ['wasm'] });
    console.log('[dog-worker] pose session ready');
    postMessage({ type: 'ready' });
  } catch (e) {
    console.error('[dog-worker] session create failed:', e);
    postMessage({ type: 'error', message: e.message });
  }
}

// ── Stage 1: detect dog bbox in full frame ────────────────────────────────────

async function detectDog(bitmap, vw, vh) {
  const lb  = letterboxFull(bitmap, vw, vh);
  const out = (await detectSession.run({ images: buildTensor() }))['output0'];
  const data = out.data;
  const A    = out.dims[2];

  const raw = [];
  for (let i = 0; i < A; i++) {
    const conf = data[(4 + DOG_CLASS) * A + i];
    if (conf < DETECT_CONF) continue;
    const cx = data[0*A+i], cy = data[1*A+i], w = data[2*A+i], h = data[3*A+i];
    const vx = (cx - lb.padX) / lb.scale;
    const vy = (cy - lb.padY) / lb.scale;
    const vw2 = w / lb.scale, vh2 = h / lb.scale;
    raw.push({ x1: vx-vw2/2, y1: vy-vh2/2, x2: vx+vw2/2, y2: vy+vh2/2, conf });
  }
  raw.sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const det of raw) {
    if (!kept.some(k => iouRect(det, k) > NMS_IOU)) kept.push(det);
  }
  if (kept.length > 0) console.log(`[detect] dog conf=${kept[0].conf.toFixed(2)} bbox=${JSON.stringify(kept[0])}`);
  else console.log(`[detect] no dog — top raw: ${raw.length > 0 ? raw[0].conf.toFixed(3) : 'none'}`);
  return kept.length > 0 ? kept[0] : null;
}

// ── Stage 2: pose on cropped region ──────────────────────────────────────────

async function poseOnCrop(bitmap, bbox, vw, vh) {
  const bw = bbox.x2 - bbox.x1, bh = bbox.y2 - bbox.y1;
  const mx = bw * CROP_MARGIN,   my = bh * CROP_MARGIN;
  const x1 = Math.max(0,  Math.round(bbox.x1 - mx));
  const y1 = Math.max(0,  Math.round(bbox.y1 - my));
  const x2 = Math.min(vw, Math.round(bbox.x2 + mx));
  const y2 = Math.min(vh, Math.round(bbox.y2 + my));
  const cropW = x2 - x1, cropH = y2 - y1;

  const poseLb = letterboxCrop(bitmap, x1, y1, cropW, cropH);
  const out    = (await poseSession.run({ images: buildTensor() }))['output0'];
  const data   = out.data;
  const A      = out.dims[2];

  // Full-frame letterbox — used to re-express keypoints in the coordinate system
  // the main thread expects (so dogModelToCanvas works unchanged)
  const fullScale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const fullPadX  = (INPUT_SIZE - Math.round(vw * fullScale)) / 2;
  const fullPadY  = (INPUT_SIZE - Math.round(vh * fullScale)) / 2;
  const fullLb    = { scale: fullScale, padX: fullPadX, padY: fullPadY };

  const raw = [];
  for (let i = 0; i < A; i++) {
    const conf = data[4 * A + i];
    if (conf < POSE_CONF) continue;
    const keypoints = [];
    for (let k = 0; k < NUM_KP; k++) {
      const base = (5 + k * 3) * A + i;
      // crop-input space → video-pixel space → full-frame model space
      const kx_vid = (data[base]     - poseLb.padX) / poseLb.scale + x1;
      const ky_vid = (data[base + A] - poseLb.padY) / poseLb.scale + y1;
      keypoints.push({
        x:    kx_vid * fullScale + fullPadX,
        y:    ky_vid * fullScale + fullPadY,
        conf: data[base + A * 2],
      });
    }
    const cx_m = data[0*A+i], cy_m = data[1*A+i], w_m = data[2*A+i], h_m = data[3*A+i];
    const bx_vid = (cx_m - poseLb.padX) / poseLb.scale + x1;
    const by_vid = (cy_m - poseLb.padY) / poseLb.scale + y1;
    raw.push({
      bbox: {
        cx: bx_vid * fullScale + fullPadX,
        cy: by_vid * fullScale + fullPadY,
        w:  w_m / poseLb.scale * fullScale,
        h:  h_m / poseLb.scale * fullScale,
        conf,
      },
      keypoints,
    });
  }

  raw.sort((a, b) => b.bbox.conf - a.bbox.conf);
  const kept = [];
  for (const det of raw) {
    const a = { x1: det.bbox.cx-det.bbox.w/2, y1: det.bbox.cy-det.bbox.h/2,
                x2: det.bbox.cx+det.bbox.w/2, y2: det.bbox.cy+det.bbox.h/2 };
    if (!kept.some(k => {
      const b = { x1: k.bbox.cx-k.bbox.w/2, y1: k.bbox.cy-k.bbox.h/2,
                  x2: k.bbox.cx+k.bbox.w/2, y2: k.bbox.cy+k.bbox.h/2 };
      return iouRect(a, b) > NMS_IOU;
    })) kept.push(det);
  }

  return { dogs: kept, lb: fullLb };
}

// ── main infer ────────────────────────────────────────────────────────────────

async function infer(bitmap, vw, vh) {
  if (!detectSession || !poseSession) { bitmap.close(); return; }

  // Stage 1: detect (throttled by REDETECT_N)
  if (redetectCount <= 0 || !lastBbox) {
    lastBbox = await detectDog(bitmap, vw, vh);
    redetectCount = REDETECT_N;
  } else {
    redetectCount--;
  }

  if (!lastBbox) {
    bitmap.close();
    postMessage({ type: 'result', dogs: [], lb: null });
    return;
  }

  // Stage 2: pose on crop
  const { dogs, lb } = await poseOnCrop(bitmap, lastBbox, vw, vh);
  bitmap.close();

  if (dogs.length === 0) lastBbox = null; // invalidate so we redetect next frame

  postMessage({ type: 'result', dogs, lb });
}

// ── message handler ───────────────────────────────────────────────────────────

self.onerror = (e) => postMessage({ type: 'error', message: `Worker uncaught: ${e.message}` });

self.onmessage = async ({ data }) => {
  if (data.type === 'init')  await init();
  if (data.type === 'infer') await infer(data.bitmap, data.vw, data.vh);
};
