import { useEffect, useState } from "react";

const STORAGE_KEY = "estatia_thread_id";

function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `t-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function useThreadId(): string {
  const [threadId] = useState<string>(() => {
    if (typeof window === "undefined") return newId();
    const existing = window.localStorage.getItem(STORAGE_KEY);
    if (existing) return existing;
    const fresh = newId();
    window.localStorage.setItem(STORAGE_KEY, fresh);
    return fresh;
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY, threadId);
  }, [threadId]);

  return threadId;
}
