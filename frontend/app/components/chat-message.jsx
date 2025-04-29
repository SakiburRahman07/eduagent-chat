import { cn } from "@/app/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "./avatar";
import { ChatBubble } from "./chat-bubble";

export function ChatMessage({
  message,
  isUser,
  reasoning,
  className,
  ...props
}) {
  return (
    <div
      className={cn(
        "flex items-end gap-2 mb-4",
        isUser ? "flex-row-reverse" : "flex-row",
        className
      )}
      {...props}
    >
      <Avatar className={cn("h-8 w-8", isUser ? "order-2" : "order-1")}>
        <AvatarFallback className={isUser ? "bg-primary/25" : "bg-secondary/25"}>
          {isUser ? "U" : "AI"}
        </AvatarFallback>
      </Avatar>
      
      <div className={cn("flex flex-col", isUser ? "items-end" : "items-start")}>
        {!isUser && reasoning && reasoning.length > 0 && (
          <div className="mb-2 text-xs text-muted-foreground">
            {reasoning.map((step, index) => (
              <div key={index} className="flex items-start gap-1 mb-1">
                <span className={cn(
                  "px-1.5 py-0.5 rounded text-[10px] uppercase font-medium",
                  step.type === "thought" ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300" :
                  step.type === "action" ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300" :
                  "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                )}>
                  {step.type}
                </span>
                <span>{step.content}</span>
              </div>
            ))}
          </div>
        )}
        <ChatBubble isUser={isUser} message={message} />
      </div>
    </div>
  );
} 