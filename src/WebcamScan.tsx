import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import type { Human as HumanInstance } from '@vladmandic/human';
import { scoreScan, type FaceFrameSample, type ReadResult } from './scoring';
import styles from './WebcamScan.module.css';

interface Props {
  onComplete: (result: ReadResult) => void;
}

const SCAN_DURATION_MS = 5000;
// How often to run the heavier CNN (Human) during the scan. Geometric landmark
// tracking via MediaPipe still runs every frame; Human runs every ~700ms so we
// gather 6–8 age samples to average over the 5-second scan.
const HUMAN_SAMPLE_EVERY_MS = 700;

type Phase =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'awaiting-face' }
  | { kind: 'locking', countdown: number }
  | { kind: 'scanning', startedAt: number }
  | { kind: 'analyzing' }
  | { kind: 'quality-fail', flags: import('./scoring').ScanQualityFlag[] }
  | { kind: 'error', message: string };

const QUALITY_REASON: Record<import('./scoring').ScanQualityFlag, string> = {
  too_few_samples:    "we didn't get enough usable frames",
  face_pose_unstable: 'your face was turned away from the camera',
  too_much_motion:    'you moved too much during the scan',
};

const TELEMETRY = [
  'mapping facial geometry',
  'reading micro-expressions',
  'computing symmetry index',
  'estimating age vector',
  'matching archetype',
  'tuning aura',
  'finalizing verdict',
];

export default function WebcamScan({ onComplete }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const humanRef = useRef<HumanInstance | null>(null);
  const samplesRef = useRef<FaceFrameSample[]>([]);
  const humanAgesRef = useRef<number[]>([]);
  const lastHumanCallRef = useRef(0);
  const humanBusyRef = useRef(false);
  const rafRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastResultRef = useRef<FaceLandmarkerResult | null>(null);
  const phaseRef = useRef<Phase>({ kind: 'idle' });

  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [hasFaceLockedFrames, setHasFaceLockedFrames] = useState(0);
  const [telemetryIdx, setTelemetryIdx] = useState(0);

  // Keep a ref of phase so the rAF loop can read it without re-binding.
  useEffect(() => { phaseRef.current = phase; }, [phase]);

  // Telemetry cycles while we are scanning / analyzing.
  useEffect(() => {
    if (phase.kind !== 'scanning' && phase.kind !== 'analyzing') return;
    const id = window.setInterval(() => setTelemetryIdx((i) => i + 1), 700);
    return () => window.clearInterval(id);
  }, [phase.kind]);

  const teardown = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    humanRef.current = null;
  }, []);

  useEffect(() => () => teardown(), [teardown]);

  const start = useCallback(async () => {
    setPhase({ kind: 'loading' });
    samplesRef.current = [];
    humanAgesRef.current = [];
    lastHumanCallRef.current = 0;
    humanBusyRef.current = false;
    setHasFaceLockedFrames(0);

    try {
      // 1. Get webcam.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = videoRef.current;
      if (!video) throw new Error('video element missing');
      video.srcObject = stream;
      await video.play();

      // 2. Initialize FaceLandmarker (MediaPipe — for live 478-dot overlay).
      const fileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
      );
      const landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        // Drives the per-sample head-pose tag — lets scoring filter symmetry
        // computation to frontal frames so a turned head doesn't masquerade
        // as facial asymmetry.
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      landmarkerRef.current = landmarker;

      // 3. Initialize Human (CNN — real age estimation from pixels).
      // `face.description` is the FaceRes model that emits age + gender.
      // Dynamic import so the TF.js bundle isn't in the initial page chunk.
      const { Human } = await import('@vladmandic/human');
      const human = new Human({
        backend: 'webgl',
        modelBasePath: 'https://vladmandic.github.io/human-models/models/',
        cacheSensitivity: 0,
        face: {
          enabled: true,
          detector: { rotation: false, maxDetected: 1 },
          mesh: { enabled: false },
          attention: { enabled: false },
          iris: { enabled: false },
          description: { enabled: true },
          emotion: { enabled: false },
          antispoof: { enabled: false },
          liveness: { enabled: false },
          gear: { enabled: false },
        },
        body: { enabled: false },
        hand: { enabled: false },
        object: { enabled: false },
        gesture: { enabled: false },
        segmentation: { enabled: false },
        filter: { enabled: false },
      });
      await human.load();
      await human.warmup();
      humanRef.current = human;

      setPhase({ kind: 'awaiting-face' });
      tick();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setPhase({ kind: 'error', message: msg });
      teardown();
    }
  }, [teardown]);

  const tick = useCallback(() => {
    const video = videoRef.current;
    const overlay = overlayRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !overlay || !landmarker) return;

    if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const result = landmarker.detectForVideo(video, performance.now());
      lastResultRef.current = result;

      const hasFace = result.faceLandmarks && result.faceLandmarks.length > 0;
      const ph = phaseRef.current;

      if (hasFace) {
        // Sync canvas to video display size so overlay aligns 1:1.
        if (overlay.width !== video.videoWidth) overlay.width = video.videoWidth;
        if (overlay.height !== video.videoHeight) overlay.height = video.videoHeight;

        drawOverlay(overlay, result, ph);

        if (ph.kind === 'awaiting-face') {
          setHasFaceLockedFrames((n) => {
            const next = n + 1;
            // 12 stable frames before we trigger the countdown.
            if (next >= 12) {
              setPhase({ kind: 'locking', countdown: 3 });
            }
            return next;
          });
        }

        if (ph.kind === 'scanning') {
          // accumulate a sample
          const lm = result.faceLandmarks[0];
          const bs: Record<string, number> = {};
          const cats = result.faceBlendshapes?.[0]?.categories ?? [];
          for (const c of cats) bs[c.categoryName] = c.score;

          // Pull head pose (yaw/pitch/roll) out of MediaPipe's 4x4 facial
          // transformation matrix. Column-major rotation block: rows of R are
          // m[i], m[i+4], m[i+8] for i in {0,1,2}.
          //   pitch (X) = asin(-R[1][2])  =>  asin(-m[9])
          //   yaw   (Y) = atan2(R[0][2], R[2][2])  =>  atan2(m[8], m[10])
          //   roll  (Z) = atan2(R[1][0], R[1][1])  =>  atan2(m[1], m[5])
          const mat = result.facialTransformationMatrixes?.[0]?.data;
          let headPose: { yaw: number; pitch: number; roll: number } | undefined;
          if (mat && mat.length >= 16) {
            const pitch = Math.asin(Math.max(-1, Math.min(1, -mat[9])));
            const yaw   = Math.atan2(mat[8], mat[10]);
            const roll  = Math.atan2(mat[1], mat[5]);
            headPose = { yaw, pitch, roll };
          }

          samplesRef.current.push({
            landmarks: lm as NormalizedLandmark[],
            blendshapes: bs,
            headPose,
          });

          // Periodically run the CNN age model on the same frame. Fire-and-forget
          // so the rAF loop never blocks on TF.js inference.
          const now = performance.now();
          const human = humanRef.current;
          if (
            human &&
            !humanBusyRef.current &&
            now - lastHumanCallRef.current > HUMAN_SAMPLE_EVERY_MS
          ) {
            lastHumanCallRef.current = now;
            humanBusyRef.current = true;
            human.detect(video).then((res) => {
              const f = res.face?.[0];
              if (f && typeof f.age === 'number' && Number.isFinite(f.age)) {
                humanAgesRef.current.push(f.age);
              }
              humanBusyRef.current = false;
            }).catch(() => {
              humanBusyRef.current = false;
            });
          }

          const elapsed = performance.now() - ph.startedAt;
          if (elapsed >= SCAN_DURATION_MS) {
            setPhase({ kind: 'analyzing' });
            // brief beat of "analyzing" theatre before reveal
            window.setTimeout(() => {
              const ages = humanAgesRef.current;
              let cnnAge: number | undefined;
              let cnnAgeBand: number | undefined;
              if (ages.length > 0) {
                // Median is robust to a single bad frame (e.g. a blurred or
                // momentarily-occluded sample) — mean would let one outlier
                // skew the read. We still use mean stddev for the band so it
                // reflects across-sample agreement.
                const sorted = [...ages].sort((a, b) => a - b);
                const median = sorted[Math.floor(sorted.length / 2)];
                cnnAge = Math.round(median);

                // Floor the band at ±3 years. Human's CNN can be very
                // confident *and very wrong* (e.g. reading a 33-year-old as
                // 16 with sd<1). The 3-year floor honors documented
                // model-error magnitude on AgeDB / Adience benchmarks even
                // when the within-scan variance is small.
                if (ages.length >= 3) {
                  const mean = ages.reduce((s, x) => s + x, 0) / ages.length;
                  const variance = ages.reduce((s, x) => s + (x - mean) ** 2, 0) / (ages.length - 1);
                  const sd = Math.sqrt(variance);
                  cnnAgeBand = Math.max(3, Math.round(sd));
                } else {
                  cnnAgeBand = 4;
                }
              }
              const out = scoreScan(samplesRef.current, cnnAge, cnnAgeBand);
              if (out.quality.ok) {
                onComplete(out);
              } else {
                setPhase({ kind: 'quality-fail', flags: out.quality.flags });
              }
            }, 900);
          }
        }
      } else {
        // No face — clear overlay so dots don't ghost.
        const ctx = overlay.getContext('2d');
        ctx?.clearRect(0, 0, overlay.width, overlay.height);
        if (ph.kind === 'awaiting-face') setHasFaceLockedFrames(0);
      }
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [onComplete]);

  // Countdown → scan transition.
  useEffect(() => {
    if (phase.kind !== 'locking') return;
    const id = window.setTimeout(() => {
      if (phase.countdown > 1) {
        setPhase({ kind: 'locking', countdown: phase.countdown - 1 });
      } else {
        setPhase({ kind: 'scanning', startedAt: performance.now() });
      }
    }, 700);
    return () => window.clearTimeout(id);
  }, [phase]);

  // Render
  const progress = phase.kind === 'scanning'
    ? Math.min(1, (performance.now() - phase.startedAt) / SCAN_DURATION_MS)
    : 0;

  return (
    <div className={styles.wrap}>
      <div className={styles.heroBlock}>
        <div className={styles.kicker}>five-second face read · v1</div>
        <h1 className={styles.hero}>
          <span className={styles.heroUnderline}>bodycount</span> detector
        </h1>
        <p className={styles.lede}>
          <b>How it works:</b> we map <b>478 facial points</b> and{' '}
          <b>52 micro-expressions</b> over five seconds, then estimate your{' '}
          body count using survey medians by age perturbed by your face read.
          runs <b>on your device</b>. no photo taken. no frame uploaded.
        </p>
      </div>

      <div className={styles.stageFrame} data-phase={phase.kind}>
        <div className={styles.cornerTL} />
        <div className={styles.cornerTR} />
        <div className={styles.cornerBL} />
        <div className={styles.cornerBR} />
        <video
          ref={videoRef}
          className={styles.video}
          autoPlay
          playsInline
          muted
        />
        <canvas ref={overlayRef} className={styles.overlay} />

        <div className={styles.stageStampTL}>you</div>
        <div className={styles.stageStampBR}>
          {phase.kind === 'scanning' ? 'reading' :
           phase.kind === 'locking' ? 'locking' :
           phase.kind === 'analyzing' ? 'compiling' :
           phase.kind === 'awaiting-face' ? 'looking for you' :
           phase.kind === 'loading' ? 'loading model' :
           phase.kind === 'error' ? 'error' : 'standby'}
        </div>

        {/* Center overlays per phase */}
        {phase.kind === 'idle' && (
          <button type="button" className={styles.beginBtn} onClick={start}>
            tap to start →
          </button>
        )}

        {phase.kind === 'loading' && (
          <div className={styles.centerNote}>
            <div className={styles.spinner} />
            <div>loading model</div>
          </div>
        )}

        {phase.kind === 'awaiting-face' && (
          <div className={styles.alignHint}>
            <div className={styles.alignTop}>center your face</div>
            <div className={styles.alignBottom}>
              {hasFaceLockedFrames > 0 ? `locking (${hasFaceLockedFrames}/12)` : 'no face detected'}
            </div>
          </div>
        )}

        {phase.kind === 'locking' && (
          <div className={styles.countdown}>{phase.countdown}</div>
        )}

        {phase.kind === 'scanning' && (
          <>
            <div className={styles.scanLine} />
            <div className={styles.progressRing} aria-hidden>
              <svg viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="46" className={styles.ringTrack} />
                <circle
                  cx="50" cy="50" r="46"
                  className={styles.ringFill}
                  style={{ strokeDashoffset: 289 * (1 - progress) }}
                />
              </svg>
              <div className={styles.progressLabel}>
                {Math.round(progress * 100)}%
              </div>
            </div>
          </>
        )}

        {phase.kind === 'analyzing' && (
          <div className={styles.centerNote}>
            <div className={styles.spinner} />
            <div>compiling verdict</div>
          </div>
        )}

        {phase.kind === 'error' && (
          <div className={styles.errBlock}>
            <div className={styles.errTitle}>scan failed</div>
            <div className={styles.errBody}>{phase.message}</div>
            <button type="button" className={styles.retryBtn} onClick={start}>
              ↺ retry
            </button>
          </div>
        )}

        {phase.kind === 'quality-fail' && (
          <div className={styles.errBlock}>
            <div className={styles.errTitle}>scan quality low</div>
            <div className={styles.errBody}>
              {phase.flags.length === 1
                ? QUALITY_REASON[phase.flags[0]]
                : phase.flags.map((f) => `• ${QUALITY_REASON[f]}`).join('\n')}
              {'\n\n'}face the camera, hold still, and we'll try again.
            </div>
            <button type="button" className={styles.retryBtn} onClick={start}>
              ↺ re-scan
            </button>
          </div>
        )}
      </div>

      <div className={styles.telemetry}>
        {(phase.kind === 'scanning' || phase.kind === 'analyzing') && (
          <>
            <span className={styles.telDot} />
            <span className={styles.telText}>
              {TELEMETRY[telemetryIdx % TELEMETRY.length]}…
            </span>
            <span className={styles.telSamples}>
              SAMPLES: {samplesRef.current.length.toString().padStart(3, '0')}
            </span>
          </>
        )}
        {phase.kind === 'idle' && (
          <span className={styles.telText}>ready · 478 landmarks · 52 micro-expressions</span>
        )}
        {phase.kind === 'awaiting-face' && (
          <span className={styles.telText}>waiting for a face…</span>
        )}
        {phase.kind === 'locking' && (
          <span className={styles.telText}>locked · starting in {phase.countdown}…</span>
        )}
      </div>

      <ul className={styles.bullets}>
        <li><span className={styles.tick}>✓</span> all inference runs on your device — no upload</li>
        <li><span className={styles.tick}>✓</span> measures 478 facial points + 52 micro-expressions</li>
        <li><span className={styles.tick}>✓</span> body count anchored to Pew &amp; CDC NSFG medians by age</li>
        <li><span className={styles.tick}>✓</span> not a medical opinion (we checked)</li>
      </ul>
    </div>
  );
}

// ----- canvas drawing --------------------------------------------------------

// Use the static connection sets MediaPipe ships with FaceLandmarker. We keep
// the tessellation extremely dim and overlay the feature outlines + bright dots.
function drawOverlay(canvas: HTMLCanvasElement, result: FaceLandmarkerResult, phase: Phase) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!result.faceLandmarks || result.faceLandmarks.length === 0) return;
  const lm = result.faceLandmarks[0];

  const isScan = phase.kind === 'scanning';
  const isAwaiting = phase.kind === 'awaiting-face' || phase.kind === 'locking';

  // 1. Tessellation — extremely dim wireframe.
  ctx.lineWidth = 0.6;
  ctx.strokeStyle = isScan
    ? 'rgba(255, 88, 88, 0.35)'
    : 'rgba(160, 200, 220, 0.22)';
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_TESSELATION);

  // 2. Feature outlines (eyes, lips, brows, oval) — brighter.
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = isScan
    ? 'rgba(255, 130, 130, 0.95)'
    : 'rgba(180, 230, 255, 0.85)';
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL);
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE);
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE);
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW);
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW);
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_LIPS);

  ctx.lineWidth = 1.0;
  ctx.strokeStyle = 'rgba(255, 240, 100, 0.9)';
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS);
  drawConnections(ctx, lm, W, H, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS);

  // 3. Dot grid on a sparse subset so it's readable, not noise.
  const dotEvery = isScan ? 3 : 6;
  ctx.fillStyle = isScan
    ? 'rgba(255, 80, 100, 0.85)'
    : isAwaiting
      ? 'rgba(120, 230, 255, 0.7)'
      : 'rgba(255, 255, 255, 0.55)';
  for (let i = 0; i < lm.length; i += dotEvery) {
    const p = lm[i];
    ctx.beginPath();
    ctx.arc(p.x * W, p.y * H, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawConnections(
  ctx: CanvasRenderingContext2D,
  lm: NormalizedLandmark[],
  W: number,
  H: number,
  connections: { start: number; end: number }[],
) {
  ctx.beginPath();
  for (const { start, end } of connections) {
    const a = lm[start];
    const b = lm[end];
    if (!a || !b) continue;
    ctx.moveTo(a.x * W, a.y * H);
    ctx.lineTo(b.x * W, b.y * H);
  }
  ctx.stroke();
}
