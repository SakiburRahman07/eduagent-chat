"use client";

import { cn } from "@/app/lib/utils";
import ReactMarkdown from "react-markdown";

export function ChatBubble({
  message,
  isUser,
  className,
  ...props
}) {
  return (
    <div
      className={cn(
        "flex flex-col space-y-2 p-4 rounded-lg max-w-[85%] text-sm",
        isUser
          ? "bg-primary text-primary-foreground self-end rounded-br-none"
          : "bg-muted self-start rounded-bl-none",
        className
      )}
      {...props}
    >
      {isUser ? (
        message
      ) : (
        <ReactMarkdown 
          className="prose dark:prose-invert prose-sm max-w-none"
          components={{
            pre: ({ node, ...props }) => (
              <div className="overflow-auto w-full my-2 bg-black/10 dark:bg-white/10 p-2 rounded-md">
                <pre {...props} />
              </div>
            ),
            code: ({ node, inline, ...props }) => 
              inline ? (
                <code className="bg-black/10 dark:bg-white/10 p-1 rounded" {...props} />
              ) : (
                <code {...props} />
              ),
            a: ({ node, ...props }) => (
              <a className="text-primary underline" target="_blank" rel="noopener noreferrer" {...props} />
            ),
          }}
        >
          {message}
        </ReactMarkdown>
      )}
    </div>
  );
} 