import { useMemo, useState } from 'react';
import type { ReadResult } from './scoring';
import PeerCheck from './PeerCheck';
import { renderShareCard, shareOrDownload } from './shareCard';
import styles from './Verdict.module.css';

interface Props {
  read: ReadResult;
  onReset: () => void;
}

export default function Verdict({ read, onReset }: Props) {
  const [shareState, setShareState] = useState<'idle' | 'rendering' | 'shared' | 'downloaded' | 'failed'>('idle');

  const readingId = useMemo(
    () => `#${String(Math.floor(Math.random() * 9000) + 1000)}`,
    [],
  );

  const share = async () => {
    if (shareState === 'rendering') return;
    setShareState('rendering');
    try {
      const { blob, file } = await renderShareCard(read, readingId);
      const result = await shareOrDownload(file, blob);
      setShareState(result === 'cancelled' ? 'idle' : result);
      if (result === 'shared' || result === 'downloaded') {
        window.setTimeout(() => setShareState('idle'), 1800);
      }
    } catch (err) {
      console.error('share card failed', err);
      setShareState('failed');
      window.setTimeout(() => setShareState('idle'), 1800);
    }
  };

  const shareLabel =
    shareState === 'rendering'  ? 'rendering…' :
    shareState === 'shared'     ? 'shared ✓' :
    shareState === 'downloaded' ? 'saved to downloads ✓' :
    shareState === 'failed'     ? 'try again' :
    'Share results';

  const {
    bodyCount, age, ageBand,
    stats, ratings, verdict, sampleCount,
  } = read;

  return (
    <>
    <section className={styles.experiment}>
      <header className={styles.head}>
        <div className={styles.headLeft}>
          <span className={styles.readingTag}>reading {readingId}</span>
        </div>
        <div className={styles.headRight}>
          <span className={styles.samples}>{sampleCount} samples · 5.0s scan</span>
        </div>
      </header>

      <div className={styles.heroBlock}>
        <div className={styles.heroLabel}>estimated body count</div>
        <div className={styles.heroNumber}>
          <span className={styles.heroValue}>{bodyCount}</span>
        </div>
      </div>

      <div className={styles.rule} aria-hidden />

      <div className={styles.metaGrid}>
        <MetaRow label="predicted age" value={`${age}`} suffix={`±${ageBand} yrs`} />
        <MetaRow label="photogenic"    value={`${stats.photogenic}`} suffix="/ 100" bar={stats.photogenic / 100} />
        <MetaRow label="approachable"  value={`${stats.approachability}`} suffix="/ 100" bar={stats.approachability / 100} />
      </div>

      <div className={styles.inputsHeader}>
        <span>feature profile</span>
        <span className={styles.inputsHeaderRule} aria-hidden />
      </div>

      <div className={styles.ratings}>
        {ratings.map((r, i) => (
          <RatingRow key={r.key} label={r.label} category={r.category} rating={r.rating} delay={i * 60} />
        ))}
      </div>

      <blockquote className={styles.verdict}>
        <span className={styles.verdictMark}>›</span> {verdict}
      </blockquote>

      <div className={styles.actions}>
        <button
          type="button"
          className={styles.shareBtn}
          onClick={share}
          disabled={shareState === 'rendering'}
        >
          {shareLabel}
        </button>
        <button type="button" className={styles.againBtn} onClick={onReset}>
          ↺ scan again
        </button>
      </div>

      <p className={styles.fineprint}>
        an experiment. body count anchored to Pew &amp; CDC NSFG median lifetime
        partners by age band, then perturbed by face read. take it personally
        anyway.
      </p>
    </section>
    <PeerCheck age={age} predictedBodyCount={bodyCount} />
    </>
  );
}

function MetaRow({
  label, value, suffix, bar,
}: { label: string; value: string; suffix: string; bar?: number }) {
  return (
    <div className={styles.metaRow}>
      <span className={styles.metaLabel}>{label}</span>
      <span className={styles.metaValue}>
        {value}
        <span className={styles.metaSuffix}> {suffix}</span>
      </span>
      {bar !== undefined && (
        <span className={styles.metaTrack}>
          <span className={styles.metaFill} style={{ width: `${Math.round(bar * 100)}%` }} />
        </span>
      )}
    </div>
  );
}

function RatingRow({
  label, category, rating, delay,
}: { label: string; category: string; rating: number; delay: number }) {
  const pct = Math.round((rating / 10) * 100);
  return (
    <div className={styles.ratingRow}>
      <span className={styles.ratingLabel}>{label}</span>
      <span className={styles.ratingCategory}>{category}</span>
      <span className={styles.ratingValue}>
        {rating}<span className={styles.ratingMax}>/10</span>
      </span>
      <span className={styles.ratingTrack}>
        <span
          className={styles.ratingFill}
          style={{ width: `${pct}%`, animationDelay: `${delay}ms` }}
        />
      </span>
    </div>
  );
}
