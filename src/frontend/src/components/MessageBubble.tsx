import { Children, isValidElement, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import { AvailabilityPill } from "./AvailabilityPill";

interface MessageBubbleProps {
  message: Message;
}

const AVAILABILITY_RE =
  /(?:✓\s*)?disponibilidad\s*:?\s*(confirmada por whatsapp|no confirmada(?:\s+por whatsapp)?)/gi;

function transformText(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let pillKey = 0;
  for (const match of text.matchAll(AVAILABILITY_RE)) {
    const start = match.index ?? 0;
    if (start > lastIndex) parts.push(text.slice(lastIndex, start));
    const confirmed = match[1].toLowerCase().startsWith("confirmada");
    parts.push(<AvailabilityPill key={`pill-${pillKey++}`} confirmed={confirmed} />);
    lastIndex = start + match[0].length;
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex));
  return parts;
}

function transformChildren(children: ReactNode): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child === "string") return transformText(child);
    if (typeof child === "number") return child;
    if (isValidElement(child)) return child;
    return child;
  });
}

const ASSISTANT_COMPONENTS: Components = {
  ul: ({ children }) => <ul className="prose-list space-y-2 my-3">{children}</ul>,
  ol: ({ children }) => (
    <ol className="list-decimal pl-5 space-y-2 my-3 marker:text-ink-55">{children}</ol>
  ),
  li: ({ children }) => <li className="leading-relaxed">{transformChildren(children)}</li>,
  p: ({ children }) => <p className="my-3 leading-relaxed">{transformChildren(children)}</p>,
  strong: ({ children }) => (
    <strong className="font-display italic font-medium text-ink">
      {transformChildren(children)}
    </strong>
  ),
  em: ({ children }) => <em className="italic text-ink-70">{children}</em>,
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="text-accent underline-offset-2 hover:underline"
    >
      {children}
    </a>
  ),
  h1: ({ children }) => (
    <h2 className="font-display text-2xl mt-5 mb-2">{children}</h2>
  ),
  h2: ({ children }) => (
    <h2 className="font-display text-xl mt-5 mb-2">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="font-display text-lg mt-4 mb-2">{children}</h3>
  ),
  hr: () => <hr className="my-5 border-ink-10" />,
  code: ({ children }) => (
    <code className="rounded bg-ink-10 px-1.5 py-0.5 text-[0.92em]">{children}</code>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-accent pl-4 my-3 italic text-ink-70">
      {children}
    </blockquote>
  ),
};

export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-[20px] border border-ink-10 px-5 py-3 text-[15.5px] leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="max-w-[95%] text-[15.5px] leading-relaxed text-ink">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={ASSISTANT_COMPONENTS}>
        {message.content}
      </ReactMarkdown>
    </div>
  );
}
