import { useEffect, useRef, type RefObject } from 'react';
import styles from './BouncingMannequin.module.css';

// A small flock of bouncing wooden artist mannequins — transparent PNG cutouts
// (sourced from Wikimedia Commons, CC0) ricocheting around the viewport while
// avoiding the central content card.
//
// We have one cutout image and instantiate it eight times with varied size,
// rotation, and horizontal mirroring so the page reads as a crowd rather than
// xeroxed copies. Touch / click on any one applies a kick impulse to just it.

interface Props {
  // Rect that the mannequins must not enter (the content card / <main>).
  avoidRef: RefObject<HTMLElement | null>;
}

interface CharInst {
  src: string;
  h: number;       // rendered height (px) — width derived from intrinsic ratio
  flip: boolean;   // mirror horizontally
}

// Intrinsic aspect (from the cutout PNG: 317 × 1154 ≈ 0.275 w/h).
const IMG_ASPECT = 317 / 1154;

const MANNEQUIN_SRC = '/assets/mannequin-wood-1.png';

const CHARS: CharInst[] = [
  { src: MANNEQUIN_SRC, h: 150, flip: false },
  { src: MANNEQUIN_SRC, h: 130, flip: true  },
  { src: MANNEQUIN_SRC, h: 170, flip: false },
  { src: MANNEQUIN_SRC, h: 110, flip: true  },
  { src: MANNEQUIN_SRC, h: 160, flip: true  },
  { src: MANNEQUIN_SRC, h: 120, flip: false },
  { src: MANNEQUIN_SRC, h: 145, flip: true  },
  { src: MANNEQUIN_SRC, h: 135, flip: false },
];

// Physics constants.
const MAX_SPEED = 1100;
const MIN_AMBIENT = 70;
const RESTITUTION = 0.92;
const AIR_DRAG = 0.18;
const KICK_BASE = 900;
const KICK_FALLOFF = 70;
const ANG_DRAG = 1.6;
const HIT_DURATION_MS = 260;

const MIN_LANE_WIDTH = 140;

interface StickerState {
  x: number; y: number;
  vx: number; vy: number;
  rot: number; rotVel: number;
  hitUntil: number;
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

function widthFor(h: number) { return Math.round(h * IMG_ASPECT); }

export default function BouncingMannequin({ avoidRef }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const figRefs = useRef<(HTMLDivElement | null)[]>([]);
  const statesRef = useRef<StickerState[]>([]);
  const rafRef = useRef(0);
  const initializedRef = useRef(false);

  const reducedMotion = useRef(
    typeof window !== 'undefined' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;

    // Bail entirely on very narrow viewports — no open zone to bounce through.
    const checkRender = () => {
      const rect = avoidRef.current?.getBoundingClientRect();
      if (!rect) return false;
      const leftLane = rect.left;
      const rightLane = window.innerWidth - rect.right;
      return Math.max(leftLane, rightLane) >= MIN_LANE_WIDTH;
    };

    if (!checkRender()) {
      wrap.style.display = 'none';
      return;
    }
    wrap.style.display = 'block';

    // Seed initial positions on first mount.
    if (!initializedRef.current) {
      const rect = avoidRef.current!.getBoundingClientRect();
      const W = window.innerWidth;
      const H = window.innerHeight;
      statesRef.current = CHARS.map((c, i) => {
        const w = widthFor(c.h);
        // Alternate lanes around the content card.
        const onLeft = i % 2 === 0;
        const xMin = onLeft ? 12 : rect.right + 12;
        const xMax = onLeft ? rect.left - w - 12 : W - w - 12;
        const x = clamp(xMin + Math.random() * Math.max(20, xMax - xMin), 8, W - w - 8);
        const yMin = 60;
        const yMax = H - c.h - 30;
        const y = clamp(yMin + (i / CHARS.length) * (yMax - yMin) + (Math.random() - 0.5) * 80, yMin, yMax);
        return {
          x, y,
          vx: (Math.random() < 0.5 ? -1 : 1) * (90 + Math.random() * 110),
          vy: (Math.random() < 0.5 ? -1 : 1) * (70 + Math.random() * 90),
          rot: (Math.random() - 0.5) * 18,
          rotVel: (Math.random() - 0.5) * 16,
          hitUntil: 0,
        };
      });
      initializedRef.current = true;
    }

    // Reduced-motion: park each mannequin in a static layout and skip rAF.
    if (reducedMotion.current) {
      const rect = avoidRef.current!.getBoundingClientRect();
      const onLeft = rect.left >= window.innerWidth - rect.right;
      const W = window.innerWidth;
      const H = window.innerHeight;
      figRefs.current.forEach((el, i) => {
        if (!el) return;
        const c = CHARS[i];
        const w = widthFor(c.h);
        const xLo = onLeft ? 8 : rect.right + 8;
        const xHi = onLeft ? rect.left - w - 8 : W - w - 8;
        const x = clamp(xLo + ((i + 1) / (CHARS.length + 1)) * (xHi - xLo) - w / 2, 8, W - w - 8);
        const y = clamp(80 + (i / CHARS.length) * (H - c.h - 160), 8, H - c.h - 8);
        el.style.transform = `translate3d(${x}px, ${y}px, 0) rotate(${(Math.random() - 0.5) * 10}deg)`;
      });
      return;
    }

    let lastT = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - lastT) / 1000);
      lastT = now;

      const ar = avoidRef.current?.getBoundingClientRect();
      const W = window.innerWidth;
      const H = window.innerHeight;
      const margin = 6;

      for (let i = 0; i < statesRef.current.length; i++) {
        const s = statesRef.current[i];
        const el = figRefs.current[i];
        const c = CHARS[i];
        if (!el || !c) continue;

        const figH = c.h;
        const figW = widthFor(c.h);

        // Integrate
        s.x += s.vx * dt;
        s.y += s.vy * dt;
        s.rot += s.rotVel * dt;

        // Drag
        const drag = Math.exp(-AIR_DRAG * dt);
        s.vx *= drag;
        s.vy *= drag;
        s.rotVel *= Math.exp(-ANG_DRAG * dt);

        // Viewport walls
        if (s.x < 0)            { s.x = 0;          s.vx =  Math.abs(s.vx) * RESTITUTION; }
        if (s.x + figW > W)     { s.x = W - figW;   s.vx = -Math.abs(s.vx) * RESTITUTION; }
        if (s.y < 0)            { s.y = 0;          s.vy =  Math.abs(s.vy) * RESTITUTION; }
        if (s.y + figH > H)     { s.y = H - figH;   s.vy = -Math.abs(s.vy) * RESTITUTION; }

        // Avoid the content card.
        if (ar) {
          const figL = s.x;
          const figR = s.x + figW;
          const figT = s.y;
          const figB = s.y + figH;
          const oxL = figR - (ar.left - margin);
          const oxR = (ar.right + margin) - figL;
          const oyT = figB - (ar.top - margin);
          const oyB = (ar.bottom + margin) - figT;
          if (oxL > 0 && oxR > 0 && oyT > 0 && oyB > 0) {
            const minOverlap = Math.min(oxL, oxR, oyT, oyB);
            if (minOverlap === oxL) {
              s.x = (ar.left - margin) - figW;
              if (s.vx > 0) s.vx = -s.vx * RESTITUTION;
            } else if (minOverlap === oxR) {
              s.x = ar.right + margin;
              if (s.vx < 0) s.vx = -s.vx * RESTITUTION;
            } else if (minOverlap === oyT) {
              s.y = (ar.top - margin) - figH;
              if (s.vy > 0) s.vy = -s.vy * RESTITUTION;
            } else {
              s.y = ar.bottom + margin;
              if (s.vy < 0) s.vy = -s.vy * RESTITUTION;
            }
          }
        }

        // Speed cap / floor
        const speed = Math.hypot(s.vx, s.vy);
        if (speed > MAX_SPEED) {
          s.vx = (s.vx / speed) * MAX_SPEED;
          s.vy = (s.vy / speed) * MAX_SPEED;
        } else if (speed < MIN_AMBIENT) {
          const ang = Math.random() * Math.PI * 2;
          s.vx += Math.cos(ang) * 14 * dt * 60;
          s.vy += Math.sin(ang) * 14 * dt * 60;
        }

        const hit = now < s.hitUntil;
        let squash = '';
        if (hit) {
          const t = 1 - (s.hitUntil - now) / HIT_DURATION_MS;
          const k = Math.sin(t * Math.PI);
          squash = ` scale(${1 + k * 0.12}, ${1 - k * 0.18})`;
        }
        const flip = c.flip ? ' scaleX(-1)' : '';
        el.className = `${styles.figure} ${hit ? styles.hit : ''}`;
        el.style.transform =
          `translate3d(${s.x}px, ${s.y}px, 0) rotate(${s.rot}deg)${flip}${squash}`;
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    const onResize = () => {
      if (!checkRender()) {
        wrap.style.display = 'none';
        cancelAnimationFrame(rafRef.current);
      } else if (wrap.style.display === 'none') {
        wrap.style.display = 'block';
        lastT = performance.now();
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', onResize);
    };
  }, [avoidRef]);

  const handlePointerDown = (idx: number) => (e: React.PointerEvent<HTMLDivElement>) => {
    if (reducedMotion.current) return;
    const el = figRefs.current[idx];
    const s = statesRef.current[idx];
    if (!el || !s) return;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = cx - e.clientX;
    const dy = cy - e.clientY;
    const dist = Math.max(6, Math.hypot(dx, dy));
    const mag = KICK_BASE * (KICK_FALLOFF / (dist + KICK_FALLOFF));
    const nx = dx / dist;
    const ny = dy / dist;
    s.vx += nx * mag;
    s.vy += ny * mag;
    s.rotVel += (Math.random() < 0.5 ? -1 : 1) * (200 + Math.random() * 260);
    s.hitUntil = performance.now() + HIT_DURATION_MS;
    e.preventDefault();
  };

  return (
    <div ref={wrapRef} className={styles.wrap} aria-hidden="true">
      {CHARS.map((c, i) => (
        <div
          key={i}
          ref={(el) => { figRefs.current[i] = el; }}
          className={styles.figure}
          style={{ width: widthFor(c.h), height: c.h }}
          onPointerDown={handlePointerDown(i)}
        >
          <img
            className={styles.img}
            src={c.src}
            alt=""
            draggable={false}
          />
        </div>
      ))}
    </div>
  );
}
