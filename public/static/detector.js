// Runs the YOLO sticker model in the browser (port of app.py's detect_face).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1; // multithreading needs cross-origin isolation headers

const MODEL_URL = '/model/best.onnx';
const IMG_SIZE = 640, STRIDE = 32, CONF = 0.25, IOU = 0.7;
const NAMES = ['blue', 'center', 'green', 'orange', 'red', 'white', 'yellow'];
const COLORS = new Set(['white', 'yellow', 'green', 'blue', 'orange', 'red']);

let sessionPromise = null;
export function loadModel() {
  sessionPromise ??= ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] })
    .catch(err => { sessionPromise = null; throw err; });
  return sessionPromise;
}

// Same letterbox as Ultralytics: keep aspect ratio, pad with grey to a multiple of 32.
function letterbox(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const r = Math.min(IMG_SIZE / w, IMG_SIZE / h);
  const nw = Math.round(w * r), nh = Math.round(h * r);
  const padW = (IMG_SIZE - nw) % STRIDE, padH = (IMG_SIZE - nh) % STRIDE;
  const left = Math.round(padW / 2 - 0.1), top = Math.round(padH / 2 - 0.1);
  const W = nw + padW, H = nh + padH;
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, left, top, nw, nh);
  const { data } = ctx.getImageData(0, 0, W, H);
  const plane = W * H, tensor = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    tensor[i] = data[i * 4] / 255;
    tensor[plane + i] = data[i * 4 + 1] / 255;
    tensor[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', tensor, [1, 3, H, W]), r, left, top };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area = d => (d.x2 - d.x1) * (d.y2 - d.y1);
  return inter / (area(a) + area(b) - inter + 1e-9);
}

// Output is [1, 4 + classes, anchors]: cx, cy, w, h, then one score per class.
export function postprocess(data, anchors, classes = NAMES.length) {
  const boxes = [];
  for (let i = 0; i < anchors; i++) {
    let best = -1, score = CONF;
    for (let c = 0; c < classes; c++) {
      const s = data[(4 + c) * anchors + i];
      if (s > score) { score = s; best = c; }
    }
    if (best < 0) continue;
    const cx = data[i], cy = data[anchors + i], w = data[2 * anchors + i], h = data[3 * anchors + i];
    boxes.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score, cls: best });
  }
  boxes.sort((a, b) => b.score - a.score);
  const kept = []; // per-class NMS
  for (const b of boxes) {
    if (kept.every(k => k.cls !== b.cls || iou(k, b) <= IOU)) kept.push(b);
    if (kept.length >= 300) break;
  }
  return kept;
}

// Split values into 3 ordered bands with the lowest total squared error (exact 1-D k-means).
export function clusterAxis(values) {
  if (values.length < 3) throw Error('Fewer than three sticker centers along an axis.');
  const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const v = order.map(o => o[0]), n = v.length;
  const sse = (a, b) => { // elements a..b-1
    const s = v.slice(a, b), m = s.reduce((x, y) => x + y, 0) / s.length;
    return s.reduce((x, y) => x + (y - m) ** 2, 0);
  };
  let best = [Infinity, 1, 2];
  for (let i = 1; i < n - 1; i++) for (let j = i + 1; j < n; j++) {
    const cost = sse(0, i) + sse(i, j) + sse(j, n);
    if (cost < best[0]) best = [cost, i, j];
  }
  const labels = Array(n);
  order.forEach(([, idx], k) => { labels[idx] = k < best[1] ? 0 : k < best[2] ? 1 : 2; });
  return labels;
}

export function toGrid(detections) {
  if (detections.length !== 9) {
    throw Error(`Detected ${detections.length} color stickers; expected 9. Keep one face visible, square to the camera, and try again.`);
  }
  const cols = clusterAxis(detections.map(d => d.x));
  const rows = clusterAxis(detections.map(d => d.y));
  const grid = Array.from({ length: 3 }, () => Array(3).fill(null));
  detections.forEach((d, i) => {
    const old = grid[rows[i]][cols[i]];
    if (!old || d.score > old.score) grid[rows[i]][cols[i]] = d;
  });
  if (grid.flat().some(c => !c)) throw Error('The detections did not form a complete 3×3 grid. Recapture this face.');
  return grid.flat().map(c => c.color);
}

// Returns the 9 sticker colors of one face, row by row.
export async function detectFace(blob) {
  const session = await loadModel();
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const { tensor, r, left, top } = letterbox(bitmap);
  bitmap.close?.();
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const output = out[session.outputNames[0]];
  const detections = postprocess(output.data, output.dims[2])
    .filter(b => COLORS.has(NAMES[b.cls]))
    .map(b => ({ x: ((b.x1 + b.x2) / 2 - left) / r, y: ((b.y1 + b.y2) / 2 - top) / r, color: NAMES[b.cls], score: b.score }));
  return toGrid(detections);
}
