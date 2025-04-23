import { cn } from "@/app/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "./avatar";
import { ChatBubble } from "./chat-bubble";

export function ChatMessage({
  message,
  isUser,
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
        <ChatBubble isUser={isUser} message={message} />
      </div>
    </div>
  );
} 