import { useEffect, useRef, useState, type KeyboardEvent } from "react";

interface InputAreaProps {
  onSend: (text: string) => void;
  disabled: boolean;
  whatsappEnabled: boolean;
  onToggleWhatsapp: (value: boolean) => void;
  placeholder?: string;
}

export function InputArea({
  onSend,
  disabled,
  whatsappEnabled,
  onToggleWhatsapp,
  placeholder = "Escribe tu búsqueda…",
}: InputAreaProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    if (disabled) return;
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div
      className="enter"
      style={{ "--delay": "280ms" } as React.CSSProperties}
    >
      <div
        className={`flex items-end gap-2 rounded-[28px] border-2 border-ink-30 bg-bg px-4 py-3 transition-opacity ${
          disabled ? "opacity-60" : ""
        }`}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          disabled={disabled}
          className="flex-1 resize-none bg-transparent px-2 py-2 text-[15.5px] leading-relaxed text-ink placeholder:text-ink-55 focus:outline-none disabled:cursor-not-allowed"
          aria-label="Escribe tu mensaje"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          aria-label="Enviar mensaje"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-accent text-bg transition disabled:cursor-not-allowed disabled:opacity-40 hover:opacity-90"
        >
          <SendGlyph />
        </button>
      </div>

      <WhatsappToggle enabled={whatsappEnabled} onChange={onToggleWhatsapp} disabled={disabled} />
    </div>
  );
}

interface WhatsappToggleProps {
  enabled: boolean;
  onChange: (value: boolean) => void;
  disabled: boolean;
}

function WhatsappToggle({ enabled, onChange, disabled }: WhatsappToggleProps) {
  return (
    <label
      className={`mt-3 flex items-center gap-3 text-sm text-ink-70 ${
        disabled ? "opacity-60" : "cursor-pointer"
      }`}
    >
      <button
        type="button"
        role="switch"
        aria-checked={enabled}
        onClick={() => !disabled && onChange(!enabled)}
        disabled={disabled}
        className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition ${
          enabled ? "bg-accent" : "bg-ink-20"
        }`}
      >
        <span
          className={`inline-block h-5 w-5 transform rounded-full bg-bg shadow-sm transition ${
            enabled ? "translate-x-[22px]" : "translate-x-[2px]"
          }`}
        />
      </button>
      <span>Contactar corredores por WhatsApp automáticamente</span>
    </label>
  );
}

function SendGlyph() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <path
        d="M8 13V3M3 8l5-5 5 5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
