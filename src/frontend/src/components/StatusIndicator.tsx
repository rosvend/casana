import type { StreamingStatus } from "../types";

interface StatusIndicatorProps {
  status: StreamingStatus | null;
  visible: boolean;
}

export function StatusIndicator({ status, visible }: StatusIndicatorProps) {
  if (!visible || !status) {
    return <div aria-hidden className="h-6" />;
  }
  return (
    <div
      key={status.message}
      role="status"
      aria-live="polite"
      className="status-line flex items-center gap-2.5 px-1 py-1 text-sm italic text-ink-55"
    >
      <span className="pulse-dot inline-block h-2 w-2 rounded-full bg-accent" aria-hidden />
      <span>{status.message}</span>
    </div>
  );
}
