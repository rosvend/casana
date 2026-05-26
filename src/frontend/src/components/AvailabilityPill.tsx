interface AvailabilityPillProps {
  confirmed: boolean;
}

export function AvailabilityPill({ confirmed }: AvailabilityPillProps) {
  if (confirmed) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-accent px-2.5 py-0.5 text-xs font-medium text-bg">
        <span aria-hidden>✓</span>
        Disponibilidad confirmada por WhatsApp
      </span>
    );
  }
  return (
    <span className="inline-flex items-center rounded-full border border-ink-20 px-2.5 py-0.5 text-xs font-medium text-ink-70">
      Disponibilidad no confirmada
    </span>
  );
}
