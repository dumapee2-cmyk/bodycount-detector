import { useCallback, useRef, useState } from 'react';
import WebcamScan from './WebcamScan';
import Verdict from './Verdict';
import BouncingMannequin from './BouncingMannequin';
import type { ReadResult } from './scoring';
import styles from './FaceAnalysis.module.css';

type Phase = 'scan' | 'done';

export default function FaceAnalysis() {
  const [phase, setPhase] = useState<Phase>('scan');
  const [read, setRead] = useState<ReadResult | null>(null);
  const mainRef = useRef<HTMLElement>(null);

  const onComplete = useCallback((r: ReadResult) => {
    setRead(r);
    setPhase('done');
  }, []);

  const reset = useCallback(() => {
    setRead(null);
    setPhase('scan');
  }, []);

  return (
    <div className={styles.page}>
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <span className={styles.brandWord}>Bodycount Detector</span>
          <span className={styles.brandKind}>How many bodies you look like you have</span>
        </div>
        <div className={styles.statusPill} data-state={phase}>
          {phase === 'scan' ? 'standing by' : 'reading complete'}
        </div>
      </header>

      <main ref={mainRef} className={styles.main}>
        {phase === 'scan' && <WebcamScan onComplete={onComplete} />}
        {phase === 'done' && read && <Verdict read={read} onReset={reset} />}
      </main>

      {phase === 'scan' && <BouncingMannequin avoidRef={mainRef} />}

      <footer className={styles.foot}>
        runs on your device · no photo taken · no frame uploaded · 478 landmarks
      </footer>
    </div>
  );
}
