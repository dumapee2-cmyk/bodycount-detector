// Renders a 1080×1350 (4:5, Instagram-feed friendly) PNG of the user's reading
// directly via the Canvas 2D API — no html2canvas / dependency required. We
// draw a clean reproduction of the verdict layout, scaled for sharing.
//
// Returns a PNG Blob plus a File constructed from it, suitable for
// navigator.share({ files: [...] }) on browsers that support it.

import type { ReadResult } from './scoring';

const W = 1080;
const H = 1350;

const COLORS = {
  paper:    '#fff8e8',
  paperDim: '#f5e9c8',
  ink:      '#0a0a0a',
  ink2:     '#3a3a3a',
  ink3:     '#7a7a7a',
  hot:      '#ed1c24',
  rule:     'rgba(10, 10, 10, 0.16)',
  fill:     'rgba(10, 10, 10, 0.08)',
} as const;

const FONT_DISPLAY = '"Archivo Black", "Inter", sans-serif';
const FONT_MONO    = '"JetBrains Mono", ui-monospace, monospace';
const FONT_BODY    = '"Inter", system-ui, sans-serif';

/**
 * Render the share card and return the PNG bytes plus a File wrapping them.
 * Browsers vary widely on Web Share with files — caller decides whether to
 * `navigator.share()` it or fall back to a download.
 */
export async function renderShareCard(
  read: ReadResult,
  readingId: string,
): Promise<{ blob: Blob; file: File }> {
  // Best-effort: wait for custom fonts to be ready so the canvas doesn't
  // render the headline number in a fallback face. Bounded so we never hang.
  await fontsReady(800);

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas 2d context unavailable');

  drawCard(ctx, read, readingId);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('toBlob failed'))),
      'image/png',
    );
  });
  const file = new File([blob], `bodycount-${readingId.replace(/[^a-z0-9-]/gi, '')}.png`, {
    type: 'image/png',
  });
  return { blob, file };
}

async function fontsReady(timeoutMs: number): Promise<void> {
  // document.fonts.ready resolves once the layout has settled. We add a
  // hard cap so a missing font never blocks the share path.
  try {
    await Promise.race([
      document.fonts?.ready,
      new Promise((r) => setTimeout(r, timeoutMs)),
    ]);
  } catch {
    /* ignore */
  }
}

// ─────────────────────────────────────────────────────────────────────────────

function drawCard(ctx: CanvasRenderingContext2D, read: ReadResult, readingId: string) {
  // Background — soft paper with a subtle vertical gradient + grain dots.
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, COLORS.paper);
  grad.addColorStop(1, COLORS.paperDim);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);
  drawGrain(ctx, 0.03);

  // Outer hairline border (subtle "specimen card" feel).
  ctx.strokeStyle = 'rgba(10, 10, 10, 0.22)';
  ctx.lineWidth = 2;
  ctx.strokeRect(28, 28, W - 56, H - 56);

  const PAD = 90;
  let y = 110;

  // ── Header bar ──────────────────────────────────────────────────────────
  ctx.fillStyle = COLORS.ink;
  ctx.font = `900 56px ${FONT_DISPLAY}`;
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.fillText("Ditto's Verdict", PAD, y);

  ctx.fillStyle = COLORS.hot;
  ctx.font = `700 22px ${FONT_MONO}`;
  ctx.textAlign = 'right';
  ctx.fillText(`READING ${readingId}`, W - PAD, y);

  y += 24;
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 24px ${FONT_MONO}`;
  ctx.textAlign = 'left';
  ctx.fillText('BODYCOUNT  DETECTOR', PAD, y);

  ctx.textAlign = 'right';
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `500 22px ${FONT_MONO}`;
  ctx.fillText(`${read.sampleCount} samples · 5.0s scan`, W - PAD, y);

  // Top rule
  y += 36;
  drawRule(ctx, PAD, y, W - PAD, COLORS.rule);

  // ── Hero ───────────────────────────────────────────────────────────────
  y += 90;
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 28px ${FONT_MONO}`;
  ctx.textAlign = 'center';
  fillTracked(ctx, 'ESTIMATED BODY COUNT', W / 2, y, 0.20);

  y += 280;
  ctx.fillStyle = COLORS.ink;
  ctx.font = `900 360px ${FONT_DISPLAY}`;
  ctx.textAlign = 'center';
  ctx.fillText(String(read.bodyCount), W / 2, y);

  // Band removed — every reading is shown as a single number now.

  // ── Divider ────────────────────────────────────────────────────────────
  y += 90;
  drawRule(ctx, PAD, y, W - PAD, COLORS.rule);

  // ── Meta rows ──────────────────────────────────────────────────────────
  y += 60;
  drawMetaRow(ctx, PAD, y, 'predicted age',  String(read.age),               `±${read.ageBand} yrs`,  undefined);
  y += 80;
  drawMetaRow(ctx, PAD, y, 'photogenic',     String(read.stats.photogenic),   '/ 100', read.stats.photogenic / 100);
  y += 80;
  drawMetaRow(ctx, PAD, y, 'approachable',   String(read.stats.approachability),'/ 100', read.stats.approachability / 100);

  // ── Feature ratings ────────────────────────────────────────────────────
  if (read.ratings.length > 0) {
    y += 60;
    drawRule(ctx, PAD, y, W - PAD, COLORS.rule);
    y += 50;
    ctx.fillStyle = COLORS.ink3;
    ctx.font = `700 22px ${FONT_MONO}`;
    ctx.textAlign = 'left';
    fillTracked(ctx, 'FEATURE  PROFILE', PAD, y, 0.20);
    y += 36;
    for (const r of read.ratings.slice(0, 7)) {
      drawRatingRow(ctx, PAD, y, r.label, r.category, r.rating);
      y += 38;
    }
  }

  // ── Verdict quote ──────────────────────────────────────────────────────
  y += 28;
  drawRule(ctx, PAD, y, W - PAD, COLORS.rule);
  y += 50;
  ctx.fillStyle = COLORS.ink2;
  ctx.font = `italic 500 26px ${FONT_BODY}`;
  ctx.textAlign = 'left';
  wrapText(ctx, `“${read.verdict}”`, PAD, y, W - PAD * 2, 36);

  // ── Footer ─────────────────────────────────────────────────────────────
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 22px ${FONT_MONO}`;
  ctx.textAlign = 'left';
  ctx.fillText('ditto · bodycount', PAD, H - 64);
  ctx.textAlign = 'right';
  ctx.fillText('on-device · no upload', W - PAD, H - 64);
}

// ── primitives ──────────────────────────────────────────────────────────────

function drawRule(ctx: CanvasRenderingContext2D, x1: number, y: number, x2: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.stroke();
}

function drawMetaRow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  label: string,
  value: string,
  suffix: string,
  bar: number | undefined,
) {
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 26px ${FONT_MONO}`;
  ctx.textAlign = 'left';
  fillTracked(ctx, label.toUpperCase(), x, y, 0.16);

  ctx.fillStyle = COLORS.ink;
  ctx.font = `900 56px ${FONT_DISPLAY}`;
  ctx.textAlign = 'left';
  const xVal = x + 360;
  ctx.fillText(value, xVal, y + 6);

  const valueWidth = ctx.measureText(value).width;
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 22px ${FONT_MONO}`;
  ctx.fillText(suffix, xVal + valueWidth + 14, y + 4);

  if (bar !== undefined) {
    const barX = x + 600;
    const barW = (W - 90) - barX;
    const barH = 12;
    const barY = y - 8;
    ctx.fillStyle = COLORS.fill;
    roundedRect(ctx, barX, barY, barW, barH, 6);
    ctx.fill();

    ctx.fillStyle = COLORS.ink;
    const fillW = Math.max(6, Math.min(1, bar) * barW);
    roundedRect(ctx, barX, barY, fillW, barH, 6);
    ctx.fill();
  }
}

function drawRatingRow(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  label: string,
  category: string,
  rating: number,
) {
  // Label (mono caps)
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 18px ${FONT_MONO}`;
  ctx.textAlign = 'left';
  fillTracked(ctx, label.toUpperCase(), x, y, 0.12);

  // Category (italic body)
  ctx.fillStyle = COLORS.ink;
  ctx.font = `italic 600 22px ${FONT_BODY}`;
  ctx.fillText(category, x + 360, y);

  // Rating value (display) — render with /10 suffix as a left-aligned pair
  // anchored to the right edge so the column still lines up.
  ctx.font = `700 14px ${FONT_MONO}`;
  const suffixW = ctx.measureText('/10').width;
  ctx.font = `900 30px ${FONT_DISPLAY}`;
  const numW = ctx.measureText(`${rating}`).width;
  const numEndX = W - 90 - suffixW - 4;
  ctx.fillStyle = COLORS.ink;
  ctx.textAlign = 'left';
  ctx.fillText(`${rating}`, numEndX - numW, y);
  ctx.fillStyle = COLORS.ink3;
  ctx.font = `700 14px ${FONT_MONO}`;
  ctx.fillText('/10', numEndX + 4, y - 4);

  // Track bar (under the row)
  const barX = x;
  const barY = y + 10;
  const barW = W - 90 - x;
  const barH = 4;
  ctx.fillStyle = COLORS.fill;
  roundedRect(ctx, barX, barY, barW, barH, 2);
  ctx.fill();
  ctx.fillStyle = COLORS.hot;
  const fillW = Math.max(4, (rating / 10) * barW);
  roundedRect(ctx, barX, barY, fillW, barH, 2);
  ctx.fill();
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y,     x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x,     y + h, rr);
  ctx.arcTo(x,     y + h, x,     y,     rr);
  ctx.arcTo(x,     y,     x + w, y,     rr);
  ctx.closePath();
}

// Word-wrap a long verdict line into a max-width column, breaking on spaces.
function wrapText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
) {
  const words = text.split(/\s+/);
  let line = '';
  let cursorY = y;
  for (let i = 0; i < words.length; i++) {
    const tryLine = line ? line + ' ' + words[i] : words[i];
    if (ctx.measureText(tryLine).width > maxWidth && line) {
      ctx.fillText(line, x, cursorY);
      line = words[i];
      cursorY += lineHeight;
    } else {
      line = tryLine;
    }
  }
  if (line) ctx.fillText(line, x, cursorY);
}

// Letter-spacing for canvas (no built-in `letter-spacing` on ctx).
function fillTracked(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  tracking: number,
) {
  // Tracking is the em-fraction extra space between glyphs.
  // We measure single-char widths and advance manually.
  const fontSize = parseFloat(/(\d+(?:\.\d+)?)px/.exec(ctx.font)?.[1] ?? '16');
  const gap = fontSize * tracking;
  const totalWidth = [...text].reduce((s, ch) => s + ctx.measureText(ch).width + gap, -gap);

  let cursorX = x;
  if (ctx.textAlign === 'center') cursorX = x - totalWidth / 2;
  else if (ctx.textAlign === 'right') cursorX = x - totalWidth;

  const prevAlign = ctx.textAlign;
  ctx.textAlign = 'left';
  for (const ch of text) {
    ctx.fillText(ch, cursorX, y);
    cursorX += ctx.measureText(ch).width + gap;
  }
  ctx.textAlign = prevAlign;
}

function drawGrain(ctx: CanvasRenderingContext2D, alpha: number) {
  ctx.fillStyle = `rgba(10, 10, 10, ${alpha})`;
  // Sparse dot grain — deterministic so two renders look the same.
  let seed = 1;
  const rand = () => {
    // tiny xorshift PRNG so we don't pull a dep
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    return ((seed >>> 0) % 10000) / 10000;
  };
  const count = 2200;
  for (let i = 0; i < count; i++) {
    const x = rand() * W;
    const y = rand() * H;
    ctx.fillRect(x, y, 1, 1);
  }
}

/**
 * Try to share via the Web Share API with a file attached; fall back to a
 * direct download. Returns a literal kind tag so the caller can show the
 * right confirmation copy.
 */
export async function shareOrDownload(file: File, blob: Blob): Promise<'shared' | 'downloaded' | 'cancelled'> {
  const nav = navigator as Navigator & {
    canShare?: (data: ShareData) => boolean;
    share?: (data: ShareData) => Promise<void>;
  };
  if (nav.canShare && nav.share && nav.canShare({ files: [file] })) {
    try {
      await nav.share({ files: [file], title: "Ditto's Verdict", text: 'my bodycount detector reading' });
      return 'shared';
    } catch (err) {
      // User cancelled the share sheet — treat as cancellation, not failure.
      if (err instanceof Error && err.name === 'AbortError') return 'cancelled';
      // otherwise fall through to download
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = file.name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return 'downloaded';
}
