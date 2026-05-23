import { useEffect, useState } from 'react';
import { peerPercentile, partnerMedianAtAge } from './scoring';
import styles from './PeerCheck.module.css';

interface Props {
  age: number;
  predictedBodyCount: number;
}

// localStorage keys — purely client-side counter for "Nth submission" social
// proof. No data ever leaves the device.
const COUNT_KEY = 'bcd:peer:submissions';
const LAST_KEY  = 'bcd:peer:lastCount';

function ordinalSuffix(n: number): string {
  const v = Math.round(n) % 100;
  if (v >= 11 && v <= 13) return 'th';
  switch (v % 10) {
    case 1: return 'st';
    case 2: return 'nd';
    case 3: return 'rd';
    default: return 'th';
  }
}

function readSubmissionCount(): number {
  try {
    const v = parseInt(localStorage.getItem(COUNT_KEY) ?? '0', 10);
    return Number.isFinite(v) && v >= 0 ? v : 0;
  } catch {
    return 0;
  }
}

function recordSubmission(value: number): number {
  try {
    const next = readSubmissionCount() + 1;
    localStorage.setItem(COUNT_KEY, String(next));
    localStorage.setItem(LAST_KEY, String(value));
    return next;
  } catch {
    return 0;
  }
}

export default function PeerCheck({ age, predictedBodyCount }: Props) {
  const [raw, setRaw] = useState<string>('');
  const [submitted, setSubmitted] = useState<number | null>(null);
  const [submissionRank, setSubmissionRank] = useState<number>(0);

  // On mount: pre-populate from last-submitted value (if any) and show the
  // running submission count as a passive social-proof line.
  useEffect(() => {
    try {
      const last = localStorage.getItem(LAST_KEY);
      if (last !== null) setRaw(last);
    } catch { /* ignore */ }
    setSubmissionRank(readSubmissionCount());
  }, []);

  const parsed = parseInt(raw, 10);
  const valid = Number.isFinite(parsed) && parsed >= 0 && parsed <= 999;

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    const rank = recordSubmission(parsed);
    setSubmissionRank(rank);
    setSubmitted(parsed);
  };

  const bumpUp = () => {
    setRaw((curr) => {
      const n = parseInt(curr, 10);
      const next = (Number.isFinite(n) ? n : 0) + 1;
      return String(Math.min(999, next));
    });
  };

  const onEdit = () => setSubmitted(null);

  // ─── result view ────────────────────────────────────────────────────────
  if (submitted !== null) {
    const pct = peerPercentile(age, submitted);
    const median = Math.round(partnerMedianAtAge(age));
    const delta = submitted - predictedBodyCount;
    const deltaSign = delta > 0 ? '+' : '';
    const pctRounded = Math.round(pct);
    const tickPos = `${Math.max(0, Math.min(100, pct))}%`;
    const fillTarget = `${Math.max(0, Math.min(100, pct))}%`;

    return (
      <section className={styles.section}>
        <header className={styles.head}>
          <span className={styles.headTag}>peer check</span>
          <span className={styles.headPriv}>anonymous · on-device</span>
        </header>

        <div className={styles.resultBlock}>
          <div className={styles.resultLabel}>your percentile vs. peers age {age}</div>
          <div>
            <span className={styles.percentile}>{pctRounded}</span>
            <span className={styles.percentileSuffix}>{ordinalSuffix(pctRounded)}</span>
          </div>
        </div>

        <div className={styles.gauge}>
          <span className={styles.gaugeFill} style={{ width: fillTarget }} />
          <span className={styles.gaugeTick} style={{ left: tickPos }} />
        </div>
        <div className={styles.gaugeLegend}>
          <span>0th</span><span>median</span><span>100th</span>
        </div>

        <div className={styles.rule} aria-hidden />

        <div className={styles.factRow}>
          <span className={styles.factLabel}>you reported</span>
          <span className={styles.factValue}>{submitted}</span>
        </div>
        <div className={styles.factRow}>
          <span className={styles.factLabel}>scan predicted</span>
          <span className={styles.factValue}>{predictedBodyCount}</span>
        </div>
        <div className={styles.factRow}>
          <span className={styles.factLabel}>
            {delta === 0 ? 'right on the nose' : delta > 0 ? 'over-performed scan' : 'under scan estimate'}
          </span>
          <span className={`${styles.factValue} ${delta > 0 ? styles.deltaUp : delta < 0 ? styles.deltaDown : ''}`}>
            {delta === 0 ? '—' : `${deltaSign}${delta}`}
          </span>
        </div>
        <div className={styles.factRow}>
          <span className={styles.factLabel}>NSFG median age {age}</span>
          <span className={styles.factValue}>{median}</span>
        </div>

        <button type="button" className={styles.editBtn} onClick={onEdit}>
          ↺ edit my number
        </button>

        <p className={styles.peerNote}>
          you're the {submissionRank.toLocaleString()}{ordinalSuffix(submissionRank)} number
          shared from this browser · nothing leaves the device
        </p>
      </section>
    );
  }

  // ─── input view ─────────────────────────────────────────────────────────
  return (
    <section className={styles.section}>
      <header className={styles.head}>
        <span className={styles.headTag}>peer check</span>
        <span className={styles.headPriv}>anonymous · on-device</span>
      </header>

      <h3 className={styles.title}>tell us your real number</h3>
      <p className={styles.lede}>
        Peer check, tell us your real bodycount to see where you stand amongst
        similar peers.
      </p>

      <form className={styles.form} onSubmit={onSubmit}>
        <div className={styles.inputWrap}>
          <input
            type="number"
            inputMode="numeric"
            min={0}
            max={999}
            placeholder="0"
            value={raw}
            onChange={(e) => setRaw(e.target.value.replace(/\D/g, '').slice(0, 3))}
            className={styles.input}
            aria-label="lifetime partners"
          />
          <span className={styles.inputUnit}>partners</span>
        </div>
        <button type="submit" className={styles.submit} disabled={!valid}>
          check ↗
        </button>
      </form>

      <button type="button" className={styles.bumpBtn} onClick={bumpUp}>
        + increase body count
      </button>

      <p className={styles.privacy}>
        · nothing transmitted · no account · only a local counter ({submissionRank.toLocaleString()} so far)
      </p>
    </section>
  );
}
