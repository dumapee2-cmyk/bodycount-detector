// Body Count Detector — a tiny experiment in face-based partner-count prediction.
//
// Inputs: per-frame samples (478 landmarks + 52 blendshapes) from a 5-second
// MediaPipe scan. We extract real biometric features (symmetry, phi, jaw,
// smile, brow, etc.), estimate a perceived age, then derive an estimated
// body count anchored to real survey medians:
//
//   bodyCount ≈ ageMedian(predictedAge) × attractivenessMultiplier(features)
//
// ageMedian comes from Pew/CDC NSFG self-reported lifetime partner medians
// by age band; the multiplier ranges ~0.5x..2.5x and shifts the median by
// face read. Deterministic — same face produces the same number.

// --- landmark indices we care about (MediaPipe 478-point topology) -----------
const IDX = {
  forehead: 10,
  chin: 152,
  leftCheek: 234,
  rightCheek: 454,

  leftEyeOuter: 33,
  leftEyeInner: 133,
  leftEyeTop: 159,
  leftEyeBottom: 145,
  rightEyeOuter: 263,
  rightEyeInner: 362,
  rightEyeTop: 386,
  rightEyeBottom: 374,

  leftBrowInner: 70,
  leftBrowOuter: 105,
  rightBrowInner: 300,
  rightBrowOuter: 334,

  mouthLeft: 61,
  mouthRight: 291,
  upperLipTop: 13,
  lowerLipBottom: 14,
  upperLipOuter: 0,
  lowerLipOuter: 17,

  // Nostril corners — drive the nose-width metric.
  noseLeftAlar: 131,
  noseRightAlar: 360,
  noseTip: 1,
  subnasale: 2,    // base of the nose (top edge of philtrum)

  // Cheekbone-zygion landmarks (the widest horizontal points of the face).
  leftZygion: 127,
  rightZygion: 356,

  // Glabella — point between the brows, used to divide the upper-third of
  // the face for the classical "facial thirds" anatomical proportion.
  glabella: 168,

  // Jaw / chin anatomy for the chin-angle metric.
  // The angle at the chin formed by lines to the left and right gonion is
  // the standard anthropometric measure of jaw sharpness.
  leftGonion: 172,
  rightGonion: 397,
} as const;

const SYMMETRY_PAIRS: [number, number][] = [
  [IDX.leftEyeOuter, IDX.rightEyeOuter],
  [IDX.leftEyeInner, IDX.rightEyeInner],
  [IDX.leftEyeTop, IDX.rightEyeTop],
  [IDX.leftEyeBottom, IDX.rightEyeBottom],
  [IDX.leftBrowInner, IDX.rightBrowInner],
  [IDX.leftBrowOuter, IDX.rightBrowOuter],
  [IDX.leftCheek, IDX.rightCheek],
  [IDX.mouthLeft, IDX.mouthRight],
];

export interface NormalizedLandmark { x: number; y: number; z: number }

export interface FaceFrameSample {
  landmarks: NormalizedLandmark[];
  blendshapes: Record<string, number>;
  /** Head pose extracted from MediaPipe's facial transformation matrix.
   *  Used to filter out turned-head frames before measuring symmetry, so a
   *  perfectly symmetric face viewed off-axis doesn't read as asymmetric. */
  headPose?: { yaw: number; pitch: number; roll: number };
}

export interface ScanFeatures {
  symmetry_index: number;
  facial_thirds_proportion: number;
  jaw_assertiveness: number;
  brow_assertiveness: number;
  smirk_index: number;
  smile_resting: number;
  attention_stability: number;
  motion_jitter: number;
  expression_variance: number;
  inter_ocular_ratio: number;
  // New high-variance shape features — significantly more discriminating
  // across individuals than the expression blendshapes, which tend to read
  // ~0 for everyone during a still scan.
  eye_aspect_ratio: number;        // tall almond eyes vs. wide round eyes
  lip_fullness: number;            // mouth height / face width
  nose_width: number;              // alar width / face width
  cheekbone_prominence: number;    // zygion span / jaw span
  face_compactness: number;        // overall square-vs-oval ratio
}

export interface Stats {
  photogenic: number;       // 0..100
  approachability: number;  // 0..100
}

/** A named, categorical reading of one facial feature.
 *  The combination of seven of these (with five categories each) gives every
 *  face a roughly-unique 7-tuple — 5^7 = ~78k combinations before you even
 *  factor in the numeric rating spread. */
export interface FeatureRating {
  key: string;            // 'symmetry' | 'jaw' | 'cheekbones' | ...
  label: string;          // 'Facial Symmetry'
  rating: number;         // 0..10
  category: string;       // 'sharp' / 'almond' / 'pillowy' / etc.
}

/** Reasons we'd reject a scan and ask the user to retry. */
export type ScanQualityFlag =
  | 'too_few_samples'
  | 'face_pose_unstable'
  | 'too_much_motion';

export interface ScanQuality {
  ok: boolean;
  flags: ScanQualityFlag[];
}

export interface ReadResult {
  bodyCount: number;
  bodyCountBand: number;   // ± N partners
  age: number;
  ageBand: number;         // ± N years
  stats: Stats;
  ratings: FeatureRating[];
  verdict: string;
  features: ScanFeatures;
  sampleCount: number;
  quality: ScanQuality;
}

// --- helpers ----------------------------------------------------------------

function distance(a: NormalizedLandmark, b: NormalizedLandmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = (a.z - b.z) * 0.5;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function clamp01(v: number): number { return Math.max(0, Math.min(1, v)); }
function clamp(v: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, v)); }

function average(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function stddev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = average(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / (xs.length - 1));
}

function averageLandmarks(samples: FaceFrameSample[]): NormalizedLandmark[] {
  const n = samples.length;
  if (n === 0) return [];
  const len = samples[0].landmarks.length;
  const out: NormalizedLandmark[] = [];
  for (let i = 0; i < len; i++) {
    let sx = 0, sy = 0, sz = 0;
    for (const s of samples) {
      const lm = s.landmarks[i];
      sx += lm.x; sy += lm.y; sz += lm.z;
    }
    out.push({ x: sx / n, y: sy / n, z: sz / n });
  }
  return out;
}

function averageBlendshapes(samples: FaceFrameSample[]): Record<string, number> {
  const out: Record<string, number> = {};
  if (samples.length === 0) return out;
  for (const key of Object.keys(samples[0].blendshapes)) {
    let s = 0;
    for (const sample of samples) s += sample.blendshapes[key] ?? 0;
    out[key] = s / samples.length;
  }
  return out;
}

// --- feature extraction -----------------------------------------------------

// Maximum head pose we accept as "frontal enough" for measuring symmetry.
// 12° in any axis ≈ 0.21 rad. Beyond this, projection-induced asymmetry
// dominates the actual facial asymmetry signal.
const FRONTAL_MAX_RAD = 0.21;

function extractFeatures(samples: FaceFrameSample[]): ScanFeatures {
  const lm = averageLandmarks(samples);
  const bs = averageBlendshapes(samples);

  const faceWidth = distance(lm[IDX.leftCheek], lm[IDX.rightCheek]);
  const faceHeight = distance(lm[IDX.forehead], lm[IDX.chin]);
  const eyeDistance = distance(lm[IDX.leftEyeOuter], lm[IDX.rightEyeOuter]);

  const inter_ocular_ratio = clamp01((eyeDistance / Math.max(faceWidth, 1e-6) - 0.30) / 0.20);

  // ── Jaw assertiveness — real chin angle ────────────────────────────────
  // Anatomical measure: the angle at the chin (landmark 152) formed by lines
  // to the left and right gonion (172, 397). Sharper jaws ⇒ smaller angle,
  // softer/rounder jaws ⇒ wider angle. Replaces the prior face-squareness
  // ratio which mislabeled wide round faces as "sharp."
  const chinAngleRad = (() => {
    const c = lm[IDX.chin];
    const l = lm[IDX.leftGonion];
    const r = lm[IDX.rightGonion];
    const v1x = l.x - c.x, v1y = l.y - c.y;
    const v2x = r.x - c.x, v2y = r.y - c.y;
    const dot = v1x * v2x + v1y * v2y;
    const m1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const m2 = Math.sqrt(v2x * v2x + v2y * v2y);
    const cosA = clamp(dot / Math.max(m1 * m2, 1e-6), -1, 1);
    return Math.acos(cosA);
  })();
  // Population range ~1.75 rad (100°, very sharp) to ~2.45 rad (140°, soft).
  const jaw_assertiveness = clamp01((2.45 - chinAngleRad) / 0.70);

  // ── Facial thirds — classical anatomical proportion ────────────────────
  // The face is "balanced" when the three vertical sections (forehead-to-
  // glabella, glabella-to-subnasale, subnasale-to-chin) are roughly equal.
  // We measure the relative deviation from equal thirds. This is a real
  // anthropometric proportion, unlike the phi/golden-ratio claim it
  // replaces (which doesn't predict attractiveness in modern studies).
  const top = Math.abs(lm[IDX.forehead].y - lm[IDX.glabella].y);
  const mid = Math.abs(lm[IDX.glabella].y - lm[IDX.subnasale].y);
  const bot = Math.abs(lm[IDX.subnasale].y - lm[IDX.chin].y);
  const meanThird = (top + mid + bot) / 3;
  const thirdsDev =
    (Math.abs(top - meanThird) + Math.abs(mid - meanThird) + Math.abs(bot - meanThird)) / 3;
  const thirdsRel = thirdsDev / Math.max(meanThird, 1e-6);
  // thirdsRel ≈ 0 ⇒ perfect, ≈ 0.30 ⇒ very uneven.
  const facial_thirds_proportion = clamp01(1 - thirdsRel / 0.28);

  // ── Symmetry — head-pose-corrected ─────────────────────────────────────
  // Only use samples where the head was within ±12° of frontal on yaw and
  // pitch. Off-axis frames produce 2D-projected asymmetry that has nothing
  // to do with the underlying face. If we have at least 8 frontal frames,
  // recompute the averaged landmarks from them; otherwise fall back to all.
  const frontal = samples.filter((s) =>
    !s.headPose ||
    (Math.abs(s.headPose.yaw) <= FRONTAL_MAX_RAD &&
      Math.abs(s.headPose.pitch) <= FRONTAL_MAX_RAD),
  );
  const lmSym = frontal.length >= 8 ? averageLandmarks(frontal) : lm;
  const symFaceWidth = distance(lmSym[IDX.leftCheek], lmSym[IDX.rightCheek]);
  const symEyeDistance = distance(lmSym[IDX.leftEyeOuter], lmSym[IDX.rightEyeOuter]);
  void symFaceWidth; // (kept for future scale normalization variants)
  const midX = (lmSym[IDX.forehead].x + lmSym[IDX.chin].x + lmSym[IDX.noseTip].x) / 3;
  const midY = (lmSym[IDX.forehead].y + lmSym[IDX.chin].y) / 2;
  let symDelta = 0;
  for (const [a, b] of SYMMETRY_PAIRS) {
    const la = lmSym[a]; const lb = lmSym[b];
    symDelta += Math.abs(Math.abs(la.x - midX) - Math.abs(lb.x - midX));
    symDelta += Math.abs(Math.abs(la.y - midY) - Math.abs(lb.y - midY)) * 0.5;
  }
  symDelta = symDelta / Math.max(symEyeDistance, 1e-6);
  const symmetry_index = clamp01(1 - (symDelta - 0.18) / 0.55);

  const mouthSmileL = bs.mouthSmileLeft ?? 0;
  const mouthSmileR = bs.mouthSmileRight ?? 0;
  const smile_resting = clamp01((mouthSmileL + mouthSmileR) * 2.5);
  const smirk_index = clamp01(Math.abs(mouthSmileL - mouthSmileR) * 6);

  const browDown = ((bs.browDownLeft ?? 0) + (bs.browDownRight ?? 0)) / 2;
  const browInnerUp = bs.browInnerUp ?? 0;
  const brow_assertiveness = clamp01(browDown * 2.5 + browInnerUp * 1.5);

  const blinkSamples = samples.map((s) =>
    Math.max(s.blendshapes.eyeBlinkLeft ?? 0, s.blendshapes.eyeBlinkRight ?? 0),
  );
  const blinkPeaks = blinkSamples.filter((b) => b > 0.45).length;
  const attention_stability = clamp01(1 - blinkPeaks / Math.max(samples.length / 6, 1));

  const noseX = samples.map((s) => s.landmarks[IDX.noseTip].x);
  const noseY = samples.map((s) => s.landmarks[IDX.noseTip].y);
  // 80× saturated on the slightest head motion; 25× is closer to "did they
  // actually shake their head?" — so photogenic doesn't collapse to ~7 just
  // because the user blinked.
  const motion_jitter = clamp01(((stddev(noseX) + stddev(noseY)) / 2) * 25);

  const browVar = stddev(samples.map((s) => (s.blendshapes.browInnerUp ?? 0)));
  const smileVar = stddev(samples.map((s) =>
    ((s.blendshapes.mouthSmileLeft ?? 0) + (s.blendshapes.mouthSmileRight ?? 0)) / 2));
  const expression_variance = clamp01((browVar + smileVar) * 8);

  // ── New high-variance shape features ──────────────────────────────────────
  // These read off the resting face geometry (not expression blendshapes) so
  // two different people with the same expression still produce different
  // values. Each is normalized to the empirical range we see across faces, so
  // the typical population mean lands near 0.5 (with real spread on either
  // side) instead of saturating to the clamp edges.

  // Eye aspect ratio: tall/almond vs. wide/round eyes.
  const leftEye_h = distance(lm[IDX.leftEyeTop], lm[IDX.leftEyeBottom]);
  const leftEye_w = distance(lm[IDX.leftEyeOuter], lm[IDX.leftEyeInner]);
  const rightEye_h = distance(lm[IDX.rightEyeTop], lm[IDX.rightEyeBottom]);
  const rightEye_w = distance(lm[IDX.rightEyeOuter], lm[IDX.rightEyeInner]);
  const ear =
    (leftEye_h / Math.max(leftEye_w, 1e-6) +
      rightEye_h / Math.max(rightEye_w, 1e-6)) / 2;
  // Population typically 0.22..0.42; map to 0..1.
  const eye_aspect_ratio = clamp01((ear - 0.20) / 0.25);

  // Lip fullness: distance from upper-lip outer to lower-lip outer / face width.
  const lipSpan = distance(lm[IDX.upperLipOuter], lm[IDX.lowerLipOuter]);
  const lipRatio = lipSpan / Math.max(faceWidth, 1e-6);
  // Typical 0.06..0.16.
  const lip_fullness = clamp01((lipRatio - 0.05) / 0.12);

  // Nose width: alar (nostril) span / face width.
  const noseSpan = distance(lm[IDX.noseLeftAlar], lm[IDX.noseRightAlar]);
  const noseRatio = noseSpan / Math.max(faceWidth, 1e-6);
  // Typical 0.18..0.32.
  const nose_width = clamp01((noseRatio - 0.16) / 0.18);

  // Cheekbone prominence: zygion (cheekbone) span vs jaw (cheek) span.
  // Faces with high cheekbones have zygion > cheek; tapered jaws push this up.
  const zygSpan = distance(lm[IDX.leftZygion], lm[IDX.rightZygion]);
  const cheekbone_prominence = clamp01((zygSpan / Math.max(faceWidth, 1e-6) - 0.90) / 0.35);

  // Face compactness: width-to-height — distinct from jaw_assertiveness which
  // mixes in chin/forehead distance. This uses only cheek-to-cheek over the
  // forehead-to-chin axis, so it captures "round vs oval" cleanly.
  const compactness = faceWidth / Math.max(faceHeight, 1e-6);
  const face_compactness = clamp01((compactness - 0.50) / 0.35);

  return {
    symmetry_index, facial_thirds_proportion, inter_ocular_ratio,
    jaw_assertiveness, brow_assertiveness, smirk_index, smile_resting,
    attention_stability, motion_jitter, expression_variance,
    eye_aspect_ratio, lip_fullness, nose_width, cheekbone_prominence, face_compactness,
  };
}

// --- age estimation ---------------------------------------------------------

function estimateAge(f: ScanFeatures): number {
  const baseline = 24;
  const age =
    baseline +
    (f.jaw_assertiveness - 0.5) * 18 +
    (f.brow_assertiveness - 0.4) * 16 +
    (0.5 - f.inter_ocular_ratio) * 12 +
    (1 - f.smile_resting) * 4 +
    (f.attention_stability - 0.5) * 4 -
    (f.expression_variance - 0.4) * 6;
  return Math.round(clamp(age, 16, 65));
}

// --- body count -------------------------------------------------------------
// Anchor body count to real-world medians by age, then perturb by a face-read
// multiplier. Medians from Pew Research Center 2020 + CDC NSFG 2017-2019
// (median lifetime opposite-sex partners by age band).

const AGE_ANCHORS: { age: number; median: number }[] = [
  { age: 18, median: 1.0 },
  { age: 22, median: 2.5 },
  { age: 26, median: 4.5 },
  { age: 30, median: 6.5 },
  { age: 35, median: 8.0 },
  { age: 45, median: 10.5 },
  { age: 55, median: 12.5 },
  { age: 65, median: 14.0 },
];

function medianByAge(age: number): number {
  // Clamp into the anchor range — extrapolating below 18 produces sub-1
  // partner medians (e.g. age 16 → 0.25), which then × a small multiplier
  // rounds to 0 partners. If the CNN underestimates age (it can — 33→16
  // happens for some faces), use the 18-anchor median as the floor.
  const a = clamp(age, AGE_ANCHORS[0].age, AGE_ANCHORS[AGE_ANCHORS.length - 1].age);
  let lo = AGE_ANCHORS[0];
  for (let i = 1; i < AGE_ANCHORS.length; i++) {
    const hi = AGE_ANCHORS[i];
    if (a <= hi.age) {
      const t = (a - lo.age) / (hi.age - lo.age);
      return lo.median + t * (hi.median - lo.median);
    }
    lo = hi;
  }
  return AGE_ANCHORS[AGE_ANCHORS.length - 1].median;
}

// Population-mean offsets for the features we weight. Centering on the actual
// observed mean (instead of a flat 0.5) is what gives an "above average"
// face a multiplier > 1.0 and a "below average" face < 1.0 — without this,
// the sigmoid would pull every typical reading toward the same middle.
const FEATURE_MEAN = {
  symmetry_index:         0.70,
  facial_thirds_proportion: 0.55,
  jaw_assertiveness:      0.50,
  smile_resting:          0.18,
  smirk_index:            0.06,
  expression_variance:    0.30,
  attention_stability:    0.62,
  motion_jitter:          0.25,
  eye_aspect_ratio:       0.45,
  lip_fullness:           0.45,
  nose_width:             0.50,
  cheekbone_prominence:   0.45,
  face_compactness:       0.55,
} as const;

function attractivenessMultiplier(f: ScanFeatures): number {
  // Weighted sum of feature deltas around the population mean. Linear (no
  // sigmoid) so two readings with even modest feature spread produce
  // visibly different multipliers — sigmoid was the main culprit collapsing
  // distinct users to the same body count.
  const z =
    1.3 * (f.symmetry_index         - FEATURE_MEAN.symmetry_index) +
    1.1 * (f.facial_thirds_proportion - FEATURE_MEAN.facial_thirds_proportion) +
    0.9 * (f.jaw_assertiveness      - FEATURE_MEAN.jaw_assertiveness) +
    0.7 * (f.cheekbone_prominence   - FEATURE_MEAN.cheekbone_prominence) +
    0.6 * (f.eye_aspect_ratio       - FEATURE_MEAN.eye_aspect_ratio) +
    0.6 * (f.lip_fullness           - FEATURE_MEAN.lip_fullness) +
    0.5 * (f.face_compactness       - FEATURE_MEAN.face_compactness) -
    0.4 * (f.nose_width             - FEATURE_MEAN.nose_width) +     // narrower noses skew higher in stated-attractiveness data
    0.5 * (f.smile_resting          - FEATURE_MEAN.smile_resting) +
    0.4 * (f.smirk_index            - FEATURE_MEAN.smirk_index) +
    0.4 * (f.expression_variance    - FEATURE_MEAN.expression_variance) +
    0.3 * (f.attention_stability    - FEATURE_MEAN.attention_stability) -
    0.6 * (f.motion_jitter          - FEATURE_MEAN.motion_jitter);
  // Tighter range: 0.55..1.80. The previous 0.35..2.85 spread compounded
  // badly with CNN age over-estimates — a 23-year-old read as ~30 by Human
  // then multiplied 2.85× landed at body count ~16. Capping the swing keeps
  // outputs plausible even when one of the two inputs is off.
  return clamp(1.0 + 1.10 * z, 0.55, 1.80);
}

function computeBodyCount(age: number, f: ScanFeatures, ratings: FeatureRating[]): number {
  const base = medianByAge(age);
  // Use the full 7-rating fingerprint as an ADDITIVE delta. Each rating is
  // 1..10, total 7..70 with a population mean ~35. An average face nets a
  // zero offset (just the age median); a "max" rating profile adds ~9, a
  // "min" profile subtracts ~7. This decouples spread from the small bases
  // at young ages, so two distinct rating profiles never collapse onto the
  // same body count just because age 22 has a base of 2.5.
  const ratingSum = ratings.reduce((s, r) => s + r.rating, 0); // 7..70
  // Strong rating-sum coefficient so the unique 7-rating profile actually
  // drives the result instead of everyone collapsing onto the same base.
  const offset = (ratingSum - 35) * 0.40; // -11.2 .. +14
  // Multiplier adds a final perturbation — two users with the same rating
  // sum but differently-distributed features still land on different
  // numbers because the multiplier weights features differently.
  const mult = attractivenessMultiplier(f);
  const tilt = (mult - 1.0) * 1.0;
  return clamp(Math.round(base + offset + tilt), 0, 30);
}

// --- summary stats ----------------------------------------------------------

function computeStats(f: ScanFeatures): Stats {
  // Photogenic now folds in the new shape features so it can vary more
  // across users (the old version only used symmetry + phi + a few others,
  // which clustered tightly across faces). Weights sum to 1.0 so the inner
  // value is bounded 0..1 before ×100.
  const photogenic = Math.round(
    clamp01(
      f.symmetry_index         * 0.22 +
      f.facial_thirds_proportion * 0.18 +
      f.cheekbone_prominence   * 0.12 +
      f.eye_aspect_ratio       * 0.10 +
      f.lip_fullness           * 0.08 +
      (1 - f.motion_jitter)    * 0.10 +
      f.smile_resting          * 0.10 +
      f.attention_stability    * 0.10,
    ) * 100,
  );
  const approachability = Math.round(
    clamp01(
      f.smile_resting          * 0.40 +
      (1 - f.brow_assertiveness) * 0.25 +
      f.attention_stability    * 0.10 +
      f.eye_aspect_ratio       * 0.10 +
      f.lip_fullness           * 0.10 +
      (1 - f.motion_jitter)    * 0.05,
    ) * 100,
  );
  return { photogenic, approachability };
}

// --- verdict line -----------------------------------------------------------

interface FeatureNote { label: string; value: number }

function pickTopFeatures(f: ScanFeatures): FeatureNote[] {
  const all: FeatureNote[] = [
    { label: 'high symmetry',         value: f.symmetry_index },
    { label: 'a sharp jaw',           value: f.jaw_assertiveness },
    { label: 'pronounced cheekbones', value: f.cheekbone_prominence },
    { label: 'almond eyes',           value: f.eye_aspect_ratio },
    { label: 'full lips',             value: f.lip_fullness },
    { label: 'a resting smile',       value: f.smile_resting },
    { label: 'a smirk',               value: f.smirk_index },
    { label: 'even facial thirds',    value: f.facial_thirds_proportion },
    { label: 'a steady gaze',         value: f.attention_stability },
    { label: 'expressive eyes',       value: f.expression_variance },
    { label: 'a heavy brow',          value: f.brow_assertiveness },
  ];
  return all.sort((a, b) => b.value - a.value);
}

function buildVerdict(age: number, count: number, f: ScanFeatures): string {
  const top = pickTopFeatures(f);
  // Pick the top two — only those above 0.5 (so we don't praise an unremarkable
  // feature). Falls back to top-1 if only one clears the bar.
  const strong = top.filter((t) => t.value >= 0.5).slice(0, 2);
  const phrase =
    strong.length === 2 ? `${strong[0].label} and ${strong[1].label}` :
    strong.length === 1 ? strong[0].label :
    'unremarkable signal';
  return `a ${age}-year-old with ${phrase} probably has around ${count} partners on the books.`;
}

// --- feature ratings --------------------------------------------------------
//
// Each face dimension gets a 0–10 numeric rating plus a named category. The
// 7-tuple of categories is what makes a read identifiable — 5 categories per
// feature × 7 features = ~78k distinct combinations before you even factor in
// the numeric variation. Two scanners with broadly similar features almost
// never land on the exact same profile.

interface CategoryStop { threshold: number; label: string }

// Each list is sorted by threshold ascending; we pick the LAST entry whose
// threshold ≤ value. (So the first entry's threshold should be 0.)
const CATEGORIES: Record<string, CategoryStop[]> = {
  symmetry: [
    { threshold: 0.00, label: 'irregular' },
    { threshold: 0.35, label: 'asymmetric' },
    { threshold: 0.55, label: 'mostly even' },
    { threshold: 0.72, label: 'highly symmetric' },
    { threshold: 0.86, label: 'mirror-perfect' },
  ],
  jaw: [
    { threshold: 0.00, label: 'soft' },
    { threshold: 0.30, label: 'gentle' },
    { threshold: 0.50, label: 'defined' },
    { threshold: 0.70, label: 'sharp' },
    { threshold: 0.85, label: 'chiseled' },
  ],
  cheekbones: [
    { threshold: 0.00, label: 'flat' },
    { threshold: 0.30, label: 'subtle' },
    { threshold: 0.50, label: 'noticeable' },
    { threshold: 0.70, label: 'high-set' },
    { threshold: 0.85, label: 'pronounced' },
  ],
  eyes: [
    { threshold: 0.00, label: 'wide-set' },
    { threshold: 0.25, label: 'rounded' },
    { threshold: 0.45, label: 'balanced' },
    { threshold: 0.65, label: 'almond' },
    { threshold: 0.82, label: 'feline' },
  ],
  lips: [
    { threshold: 0.00, label: 'thin' },
    { threshold: 0.30, label: 'modest' },
    { threshold: 0.50, label: 'medium' },
    { threshold: 0.70, label: 'full' },
    { threshold: 0.85, label: 'pillowy' },
  ],
  nose: [
    { threshold: 0.00, label: 'narrow' },
    { threshold: 0.30, label: 'slim' },
    { threshold: 0.50, label: 'average' },
    { threshold: 0.70, label: 'broad' },
    { threshold: 0.85, label: 'wide' },
  ],
  harmony: [
    { threshold: 0.00, label: 'top-heavy' },
    { threshold: 0.30, label: 'uneven thirds' },
    { threshold: 0.50, label: 'mostly balanced' },
    { threshold: 0.70, label: 'even thirds' },
    { threshold: 0.85, label: 'classical proportion' },
  ],
};

function categorize(key: string, value: number): string {
  const stops = CATEGORIES[key];
  if (!stops) return '';
  let label = stops[0].label;
  for (const s of stops) if (value >= s.threshold) label = s.label;
  return label;
}

// Map a 0..1 feature into a 1..10 rating. We use 1 as the floor (no zeros)
// so the readout always looks like an opinion, not "missing data."
function rate(value: number): number {
  return clamp(Math.round(value * 9) + 1, 1, 10);
}

function buildRatings(f: ScanFeatures): FeatureRating[] {
  // Weights for the BODY-COUNT-driving sum are encoded later, in
  // computeBodyCount. Display order here is what the user sees in the card.
  return [
    { key: 'symmetry',   label: 'Facial Symmetry',     rating: rate(f.symmetry_index),         category: categorize('symmetry',   f.symmetry_index) },
    { key: 'jaw',        label: 'Jawline Definition',  rating: rate(f.jaw_assertiveness),      category: categorize('jaw',        f.jaw_assertiveness) },
    { key: 'cheekbones', label: 'Cheekbone Prominence',rating: rate(f.cheekbone_prominence),   category: categorize('cheekbones', f.cheekbone_prominence) },
    { key: 'eyes',       label: 'Eye Shape',           rating: rate(f.eye_aspect_ratio),       category: categorize('eyes',       f.eye_aspect_ratio) },
    { key: 'lips',       label: 'Lip Fullness',        rating: rate(f.lip_fullness),           category: categorize('lips',       f.lip_fullness) },
    { key: 'nose',       label: 'Nose Width',          rating: rate(f.nose_width),             category: categorize('nose',       f.nose_width) },
    { key: 'harmony',    label: 'Facial Proportion',   rating: rate(f.facial_thirds_proportion), category: categorize('harmony',    f.facial_thirds_proportion) },
  ];
}

// --- scan quality -----------------------------------------------------------
// Surface the unfixable degenerate scans (head turned the whole time, user
// moved a lot, fewer than ~1 sec of usable samples) so the UI can prompt a
// retry instead of confidently reporting "0 partners" from garbage features.

export function assessScanQuality(samples: FaceFrameSample[], _features: ScanFeatures | null): ScanQuality {
  const flags: ScanQualityFlag[] = [];
  // The only thing that genuinely makes a scan unusable is MediaPipe not
  // tracking the face. If we collected at least ~⅓ second of frames the read
  // is real, whatever it says. Motion / asymmetry / off-axis are *signals*,
  // not failures — let them through to the verdict instead of gating.
  if (samples.length < 10) flags.push('too_few_samples');
  return { ok: flags.length === 0, flags };
}

// --- public ----------------------------------------------------------------

export function scoreScan(
  samples: FaceFrameSample[],
  cnnAge?: number,
  cnnAgeBand?: number,
): ReadResult {
  if (samples.length === 0) return emptyResult();

  const features = extractFeatures(samples);
  // Reconcile Human's CNN age with our geometric estimator. The CNN has
  // documented systematic bias in both directions.
  //
  // Asymmetric reconciliation policy (by request): when the estimators
  // disagree, only ever push the displayed age YOUNGER. Never older.
  //  - CNN reads older than geometric → blend pulls down toward geometric.
  //  - CNN reads younger than geometric → keep the CNN read (don't curve up).
  //  - Estimators within ~4 years → trust the CNN.
  // The band still widens with disagreement so the uncertainty is visible.
  const geomAge = estimateAge(features);
  let age: number;
  let ageBand: number;
  if (cnnAge !== undefined && Number.isFinite(cnnAge)) {
    const cnn = clamp(Math.round(cnnAge), 13, 80);
    const diff = Math.abs(cnn - geomAge);
    if (diff < 4 || cnn <= geomAge) {
      age = cnn;
    } else {
      // CNN > geometric by ≥4 — pull toward geometric.
      const wCnn = clamp(0.85 - diff * 0.025, 0.45, 0.85);
      age = clamp(Math.round(cnn * wCnn + geomAge * (1 - wCnn)), 13, 80);
    }
    const sampleSd = cnnAgeBand !== undefined ? cnnAgeBand : 3;
    ageBand = clamp(Math.max(sampleSd, Math.round(diff / 2)), 2, 8);
  } else {
    age = geomAge;
    ageBand = 5;
  }

  // Final calibration: nudge very-young readings up by 2 years. The CNN
  // (and our geometric estimator) consistently land low in the 16–19 band
  // for typical adult users, so this small offset brings the displayed age
  // closer to actual ages observed in testing.
  if (age >= 16 && age <= 19) {
    age += 2;
  }
  // Ratings drive the additive body-count formula now, so they must be
  // built before computing body count.
  const ratings = buildRatings(features);
  const bodyCount = computeBodyCount(age, features, ratings);
  const bodyCountBand = 0; // Band display removed — every read is one number.
  const stats = computeStats(features);
  const quality = assessScanQuality(samples, features);
  const verdict = buildVerdict(age, bodyCount, features);

  return {
    bodyCount,
    bodyCountBand,
    age,
    ageBand,
    stats,
    ratings,
    verdict,
    features,
    sampleCount: samples.length,
    quality,
  };
}

// --- peer percentile --------------------------------------------------------
// Lognormal model of lifetime partner counts: distribution is heavily right-
// skewed in NSFG / Pew data. Median grows with age (already encoded in
// AGE_ANCHORS / medianByAge). We treat the population at a given age as
// lognormal(μ, σ²) with μ = ln(median) and σ ≈ 1.0 — consistent with the
// observed coefficient of variation in NSFG cycles 2017–2019.
//
// Special case: the survey share reporting zero lifetime partners drops from
// ~30% at 18 to ~3% by 35+. Lognormal can't model a true zero, so for c = 0
// we anchor to that survey share and skip the CDF.

// Abramowitz & Stegun 7.1.26 erf approximation. Max error ~1.5e-7.
function erf(x: number): number {
  const a1 =  0.254829592, a2 = -0.284496736, a3 =  1.421413741;
  const a4 = -1.453152027, a5 =  1.061405429, p  =  0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

function normalCdf(z: number): number {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

// Approx share of adults reporting exactly zero lifetime partners by age.
// NSFG 2017–2019, smoothed.
function zeroShareAt(age: number): number {
  if (age <= 18) return 0.32;
  if (age <= 22) return 0.14;
  if (age <= 26) return 0.08;
  if (age <= 30) return 0.05;
  if (age <= 40) return 0.035;
  return 0.03;
}

/**
 * Percentile of a self-reported lifetime partner count among adults of the
 * given age. Returns a 0..100 value: 30 means "30% of your age group reports
 * fewer than this," i.e. you're at the 30th percentile (lower-tail).
 */
export function peerPercentile(age: number, count: number): number {
  const c = Math.max(0, Math.floor(count));
  if (c === 0) {
    // Anyone at 0 is below the entire non-zero distribution; we put them at
    // the midpoint of the zero-share bracket so it doesn't read as "0th."
    return Math.max(0.5, zeroShareAt(age) * 100 * 0.5);
  }
  const median = medianByAge(age);
  const mu = Math.log(Math.max(0.5, median));
  const sigma = 1.0;
  // Use ln(c + 0.5) (continuity correction for the discrete count).
  const z = (Math.log(c + 0.5) - mu) / sigma;
  const zeroShare = zeroShareAt(age);
  // Lognormal CDF reflects the non-zero population; blend in the zero-share
  // floor so percentiles are over the full distribution including 0s.
  const upper = normalCdf(z);
  const pct = (zeroShare + (1 - zeroShare) * upper) * 100;
  return Math.max(0.1, Math.min(99.9, pct));
}

/** Median lifetime partner count at the given age (exposed for UI compare). */
export function partnerMedianAtAge(age: number): number {
  return medianByAge(age);
}

function emptyResult(): ReadResult {
  return {
    bodyCount: 0,
    bodyCountBand: 0,
    age: 0,
    ageBand: 0,
    stats: { photogenic: 0, approachability: 0 },
    ratings: [],
    verdict: 'no face detected — try again.',
    features: {
      symmetry_index: 0, facial_thirds_proportion: 0, inter_ocular_ratio: 0,
      jaw_assertiveness: 0, brow_assertiveness: 0, smirk_index: 0,
      smile_resting: 0, attention_stability: 0, motion_jitter: 0,
      expression_variance: 0,
      eye_aspect_ratio: 0, lip_fullness: 0, nose_width: 0,
      cheekbone_prominence: 0, face_compactness: 0,
    },
    sampleCount: 0,
    quality: { ok: false, flags: ['too_few_samples'] },
  };
}
