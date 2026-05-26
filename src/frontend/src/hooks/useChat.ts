import { useCallback, useEffect, useRef, useState } from "react";
import { getHistory, streamChat, submitResume } from "../api/client";
import type {
  BackendMessage,
  DoneEvent,
  InterruptPayload,
  Message,
  StreamingStatus,
} from "../types";
import { useThreadId } from "./useThreadId";

interface UseChatResult {
  threadId: string;
  messages: Message[];
  status: StreamingStatus | null;
  activeInterrupt: InterruptPayload | null;
  isStreaming: boolean;
  errorText: string | null;
  whatsappEnabled: boolean;
  setWhatsappEnabled: (value: boolean) => void;
  sendMessage: (text: string) => Promise<void>;
}

const WHATSAPP_KEY = "estatia_whatsapp_enabled";

export function useChat(): UseChatResult {
  const threadId = useThreadId();
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<StreamingStatus | null>(null);
  const [activeInterrupt, setActiveInterrupt] = useState<InterruptPayload | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [whatsappEnabled, setWhatsappEnabledState] = useState<boolean>(() => {
    if (typeof window === "undefined") return true;
    const raw = window.localStorage.getItem(WHATSAPP_KEY);
    return raw === null ? true : raw === "true";
  });

  const historyLoadedRef = useRef(false);

  // whatsappEnabled is UI-only for now; the backend has no field for it yet.
  // We still persist it so the toggle survives reloads.
  const setWhatsappEnabled = useCallback((value: boolean) => {
    setWhatsappEnabledState(value);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(WHATSAPP_KEY, String(value));
    }
  }, []);

  useEffect(() => {
    if (historyLoadedRef.current) return;
    historyLoadedRef.current = true;
    getHistory(threadId)
      .then((response) => {
        if (!response.messages?.length) return;
        const restored = response.messages
          .map(toMessage)
          .filter((m): m is Message => m !== null);
        if (restored.length) setMessages(restored);
      })
      .catch(() => {
        /* fresh thread or backend unreachable — ignore */
      });
  }, [threadId]);

  const applyDone = useCallback((done: DoneEvent) => {
    const lastAi = [...done.messages].reverse().find((m) => m.type === "ai");
    const aiContent = lastAi?.content?.trim();
    // When the graph pauses on an interrupt() without a preceding AIMessage,
    // surface the clarification_question itself as the assistant turn.
    const fallback = done.interrupt?.clarification_question?.trim();
    const content = aiContent || fallback;
    if (content) {
      setMessages((prev) => [
        ...prev,
        {
          id: lastAi?.id || `ai-${Date.now()}`,
          role: "assistant",
          content,
          timestamp: Date.now(),
        },
      ]);
    }
    setActiveInterrupt(done.interrupt ?? null);
    setStatus(null);
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;

      setErrorText(null);
      setMessages((prev) => [
        ...prev,
        {
          id: `user-${Date.now()}`,
          role: "user",
          content: trimmed,
          timestamp: Date.now(),
        },
      ]);
      setIsStreaming(true);

      try {
        if (activeInterrupt) {
          setStatus({ node: "requirements_agent", message: "Procesando tu respuesta…" });
          setActiveInterrupt(null);
          const done = await submitResume(threadId, trimmed);
          applyDone(done);
        } else {
          const done = await streamChat(threadId, trimmed, (event) => {
            if (event.type === "status") {
              setStatus({ node: event.data.node, message: event.data.message });
            } else if (event.type === "error") {
              setErrorText(event.data.message);
            }
          });
          applyDone(done);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Ocurrió un error inesperado.";
        setErrorText(message);
        setStatus(null);
      } finally {
        setIsStreaming(false);
      }
    },
    [activeInterrupt, applyDone, isStreaming, threadId],
  );

  return {
    threadId,
    messages,
    status,
    activeInterrupt,
    isStreaming,
    errorText,
    whatsappEnabled,
    setWhatsappEnabled,
    sendMessage,
  };
}

function toMessage(backend: BackendMessage): Message | null {
  if (backend.type !== "human" && backend.type !== "ai") return null;
  if (!backend.content?.trim()) return null;
  return {
    id: backend.id || `${backend.type}-${Math.random().toString(36).slice(2, 10)}`,
    role: backend.type === "human" ? "user" : "assistant",
    content: backend.content,
    timestamp: Date.now(),
  };
}
