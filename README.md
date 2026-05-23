# Bodycount Detector

A five-second face read. Body count anchored to survey medians by age, perturbed by your face. Runs on your device — no photo taken, no frame uploaded.

## Stack

- **[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)** — 478 3D facial landmarks + 52 blendshape coefficients + facial transformation matrix, real-time in the browser.
- **[@vladmandic/human](https://github.com/vladmandic/human)** — TensorFlow.js-backed CNN for age estimation. Sampled every ~700 ms during the scan; median across samples drives the displayed age.
- **React + Vite + TypeScript**, all client-side.

## How it works

1. Webcam-only `getUserMedia` capture for 5 seconds, mirrored selfie-style.
2. Every frame: MediaPipe extracts 478 landmarks + blendshapes + head-pose matrix. Periodically: Human's CNN reads age.
3. After the scan, `scoring.ts` extracts 15 features (symmetry — head-pose corrected, jaw chin-angle, cheekbone prominence, eye aspect ratio, lip fullness, nose width, facial-thirds proportion, etc.), turns them into 7 named ratings (`sharp` jaw, `almond` eyes, `pillowy` lips, …), and combines them with the age-anchored Pew/NSFG partner-count median.

Math is documented inline in [`src/scoring.ts`](src/scoring.ts).

## Run locally

```bash
npm install
npm run dev
# → http://localhost:5173/
```

The webcam needs a **secure context**: `localhost` works out of the box, any non-localhost origin needs HTTPS (browsers block `getUserMedia` on plain HTTP).

## Build

```bash
npm run build
# → dist/
```

The `dist/` folder is a fully static bundle — drops onto any static host (Vercel / Netlify / Cloudflare Pages / GitHub Pages / Surge) with zero configuration.

First scan downloads ~15 MB of ML models from public CDNs (Google for MediaPipe WASM + model, GitHub Pages for the Human face description CNN). Browser-cached after that.

## Notes

- Body count math is **not real**. It blends age-band partner medians from Pew Research / CDC NSFG against a 7-feature attractiveness signal. Designed to feel believable, not to be believed.
- Age estimation has a documented ±5–10 yr error floor across every academic benchmark (AgeDB, Adience, Morph II). We blend CNN with a geometric estimator and only ever pull the displayed age *younger* on disagreement.
- This is an experiment. Take it personally anyway.
