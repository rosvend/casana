import { useEffect, useState } from "react";

const WORDS = ["arriendo", "apartamento", "hogar", "refugio", "lugar"] as const;
const INTERVAL_MS = 2200;

export function RotatingWord() {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<"enter" | "exit">("enter");

  useEffect(() => {
    const cycle = window.setInterval(() => {
      setPhase("exit");
      window.setTimeout(() => {
        setIndex((i) => (i + 1) % WORDS.length);
        setPhase("enter");
      }, 280);
    }, INTERVAL_MS);
    return () => window.clearInterval(cycle);
  }, []);

  // The widest word reserves layout width so the surrounding sentence
  // doesn't jitter as words swap.
  const widest = WORDS.reduce((a, b) => (a.length >= b.length ? a : b));

  return (
    <span className="word-slot">
      <span className="word-slot__measure">{widest}</span>
      <span className="word-slot__visible" data-state={phase}>
        {WORDS[index]}
      </span>
    </span>
  );
}
