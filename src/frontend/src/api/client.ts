import type { BackendMessage, DoneEvent, StreamEvent } from "../types";

const JSON_HEADERS = { "Content-Type": "application/json" } as const;

export interface ChatRequestBody {
  thread_id: string;
  user_message: string;
}

export interface ResumeRequestBody {
  thread_id: string;
  resume_payload: string;
}

export interface HistoryResponse {
  thread_id: string;
  messages: BackendMessage[];
}

/**
 * POST /api/chat/stream — consumes the SSE stream and invokes `onEvent`
 * for every framed event. Resolves with the terminal `done` payload, or
 * rejects on an `error` event / network failure.
 */
export async function streamChat(
  threadId: string,
  userMessage: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<DoneEvent> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { ...JSON_HEADERS, Accept: "text/event-stream" },
    body: JSON.stringify({ thread_id: threadId, user_message: userMessage } satisfies ChatRequestBody),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await safeText(response);
    throw new Error(`stream chat failed (${response.status}): ${detail}`);
  }

  return consumeSse(response.body, onEvent);
}

/**
 * POST /api/resume — non-streaming. Returns the same shape as a `done`
 * event so the caller can update state uniformly.
 */
export async function submitResume(threadId: string, replyText: string): Promise<DoneEvent> {
  const response = await fetch("/api/resume", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ thread_id: threadId, resume_payload: replyText } satisfies ResumeRequestBody),
  });

  if (!response.ok) {
    throw new Error(`resume failed (${response.status}): ${await safeText(response)}`);
  }

  return (await response.json()) as DoneEvent;
}

/**
 * GET /api/history/{thread_id} — restores prior conversation on mount.
 */
export async function getHistory(threadId: string): Promise<HistoryResponse> {
  const response = await fetch(`/api/history/${encodeURIComponent(threadId)}`);
  if (!response.ok) {
    throw new Error(`history fetch failed (${response.status})`);
  }
  return (await response.json()) as HistoryResponse;
}

async function consumeSse(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: StreamEvent) => void,
): Promise<DoneEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let final: DoneEvent | null = null;
  let streamError: Error | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let separatorIndex: number;
    while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      const parsed = parseFrame(frame);
      if (!parsed) continue;
      onEvent(parsed);
      if (parsed.type === "done") {
        final = parsed.data;
      } else if (parsed.type === "error") {
        streamError = new Error(parsed.data.message || "stream error");
      }
    }
  }

  if (streamError) throw streamError;
  if (!final) throw new Error("stream ended without a `done` event");
  return final;
}

function parseFrame(frame: string): StreamEvent | null {
  let eventName = "message";
  const dataLines: string[] = [];
  for (const rawLine of frame.split("\n")) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    const colon = rawLine.indexOf(":");
    const field = colon === -1 ? rawLine : rawLine.slice(0, colon);
    const value = colon === -1 ? "" : rawLine.slice(colon + 1).replace(/^ /, "");
    if (field === "event") eventName = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }
  switch (eventName) {
    case "status":
    case "message":
    case "error":
    case "done":
      return { type: eventName, data } as StreamEvent;
    default:
      return null;
  }
}

async function safeText(response: Response): Promise<string> {
  try {
    return await response.text();
  } catch {
    return "<no body>";
  }
}
