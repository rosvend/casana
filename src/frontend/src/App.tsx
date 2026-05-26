import { useEffect, useRef } from "react";
import { Header } from "./components/Header";
import { InputArea } from "./components/InputArea";
import { MessageBubble } from "./components/MessageBubble";
import { StatusIndicator } from "./components/StatusIndicator";
import { useChat } from "./hooks/useChat";

function App() {
  const {
    messages,
    status,
    activeInterrupt,
    isStreaming,
    errorText,
    whatsappEnabled,
    setWhatsappEnabled,
    sendMessage,
  } = useChat();

  const scrollAnchorRef = useRef<HTMLDivElement | null>(null);
  const hasConversation = messages.length > 0 || isStreaming;

  useEffect(() => {
    if (hasConversation) {
      scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages.length, status?.message, hasConversation]);

  const placeholder = activeInterrupt ? "Tu respuesta…" : "Escribe tu búsqueda…";

  if (!hasConversation) {
    return (
      <div className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center gap-10 px-6 py-10">
        <Header variant="hero" />
        <div className="w-full">
          <InputArea
            onSend={sendMessage}
            disabled={isStreaming}
            whatsappEnabled={whatsappEnabled}
            onToggleWhatsapp={setWhatsappEnabled}
            placeholder={placeholder}
          />
          {errorText && (
            <p className="mt-4 text-center text-sm text-ink-70">{errorText}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-3xl flex-col px-6 pb-10">
      <Header variant="compact" />

      <main className="flex flex-1 flex-col gap-6 pb-6">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}

        {errorText && (
          <div className="rounded-2xl border border-ink-20 px-4 py-3 text-sm text-ink-70">
            <span className="font-medium text-ink">Algo salió mal. </span>
            {errorText}
          </div>
        )}

        <div ref={scrollAnchorRef} />
      </main>

      <div className="sticky bottom-0 bg-bg/95 backdrop-blur-sm pt-2">
        <StatusIndicator status={status} visible={isStreaming} />
        <InputArea
          onSend={sendMessage}
          disabled={isStreaming}
          whatsappEnabled={whatsappEnabled}
          onToggleWhatsapp={setWhatsappEnabled}
          placeholder={placeholder}
        />
      </div>
    </div>
  );
}

export default App;
