export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
}

export interface InterruptPayload {
  clarification_question: string;
}

export interface StreamingStatus {
  node: string;
  message: string;
}

export interface BackendMessage {
  type: "human" | "ai" | "system" | "tool";
  content: string;
  id?: string | null;
}

export interface DoneEvent {
  messages: BackendMessage[];
  final_results: unknown[];
  evaluation: unknown | null;
  is_best_effort: boolean;
  interrupt: InterruptPayload | null;
}

export type StreamEvent =
  | { type: "status"; data: StreamingStatus & { ts: number } }
  | { type: "message"; data: { node: string; ts: number } }
  | { type: "error"; data: { message: string } }
  | { type: "done"; data: DoneEvent };
