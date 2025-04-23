"use client";

import { useState, useRef, useEffect } from "react";
import { Send, RefreshCw } from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/app/components/ui/avatar";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/app/components/ui/tooltip";
import ReactMarkdown from "react-markdown";
import { cn } from "@/app/lib/utils";

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(Date.now().toString());
  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);
  
  // Focus the input field on load and after response
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [messages, isLoading]);

  // Auto-scroll to the bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleExampleClick = (question) => {
    setInput(question);
    // Wait a brief moment then submit the form
    setTimeout(() => {
      handleSubmit();
    }, 100);
  };

  const startNewConversation = () => {
    setMessages([]);
    setConversationId(Date.now().toString());
    setInput("");
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  async function handleSubmit(e) {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    // Add user message to the chat
    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    
    // Save the current input before clearing it
    const currentInput = input;
    setInput("");
    setIsLoading(true);

    try {
      // Log the request being sent
      console.log("Sending request:", {
        message: currentInput,
        conversation_id: conversationId
      });
      
      // Send message to API using the expected format
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: currentInput,
          conversation_id: conversationId,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API error:", errorText);
        throw new Error(`Failed to send message: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received response:", data);
      
      // Add AI response to the chat
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  // Chat message component
  const ChatMessage = ({ message, isUser }) => (
    <div className={cn("flex items-start gap-3 mb-4", isUser && "flex-row-reverse")}>
      <Avatar className="h-8 w-8 mt-0.5">
        <AvatarFallback className={isUser ? "bg-primary/20" : "bg-secondary/20"}>
          {isUser ? "U" : "SB"}
        </AvatarFallback>
      </Avatar>
      
      <div className={cn(
        "flex flex-col max-w-[80%]", 
        isUser ? "items-end" : "items-start"
      )}>
        <div className={cn(
          "px-4 py-2.5 rounded-lg",
          isUser 
            ? "bg-primary text-primary-foreground rounded-tr-none" 
            : "bg-secondary/30 text-foreground rounded-tl-none"
        )}>
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap break-words">{message}</p>
          ) : (
            <ReactMarkdown 
              className="prose dark:prose-invert prose-sm max-w-none"
              components={{
                pre: ({ node, ...props }) => (
                  <div className="overflow-auto w-full my-2 bg-background/80 p-2 rounded-md border">
                    <pre {...props} />
                  </div>
                ),
                code: ({ node, inline, ...props }) => 
                  inline ? (
                    <code className="bg-background/80 px-1 py-0.5 rounded border" {...props} />
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
      </div>
    </div>
  );

  return (
    <Card className="flex flex-col h-full border-none shadow-none bg-transparent">
      <CardHeader className="px-0 pt-0">
        <CardTitle className="sr-only">Chat with Study Buddy</CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full pr-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-4">
              <h2 className="text-2xl font-bold mb-2">Welcome to Study Buddy</h2>
              <p className="text-muted-foreground mb-6">
                Your AI research assistant for learning and studying. Ask me anything or try one of these examples:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full max-w-md">
                {[
                  "Explain photosynthesis with examples",
                  "What are the key themes in Romeo and Juliet?",
                  "Find recent research papers on climate change",
                  "Explain quantum computing for beginners",
                ].map((question) => (
                  <Button 
                    key={question}
                    variant="outline"
                    className="justify-start h-auto py-3 px-4 text-left"
                    onClick={() => handleExampleClick(question)}
                  >
                    {question}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="py-4 px-2">
              {messages.map((message, i) => (
                <ChatMessage 
                  key={i} 
                  message={message.content} 
                  isUser={message.role === "user"} 
                />
              ))}
              {isLoading && (
                <div className="flex items-start gap-3 mb-4">
                  <Avatar className="h-8 w-8 mt-0.5">
                    <AvatarFallback className="bg-secondary/20">SB</AvatarFallback>
                  </Avatar>
                  <div className="bg-secondary/30 text-foreground px-4 py-2.5 rounded-lg rounded-tl-none max-w-[80%]">
                    <div className="flex items-center gap-1.5">
                      <div className="h-2 w-2 rounded-full bg-foreground/40 animate-pulse"></div>
                      <div className="h-2 w-2 rounded-full bg-foreground/40 animate-pulse delay-150"></div>
                      <div className="h-2 w-2 rounded-full bg-foreground/40 animate-pulse delay-300"></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </ScrollArea>
      </CardContent>
      
      <CardFooter className="pt-6 px-0 pb-0">
        <div className="w-full space-y-3">
          {messages.length > 0 && (
            <div className="flex justify-between items-center">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={startNewConversation}
                      className="text-xs"
                    >
                      <RefreshCw className="h-3 w-3 mr-2" />
                      New Topic
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Start a new research topic</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="flex space-x-2">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about any subject..."
              className="min-h-[60px] max-h-[200px] flex-1 resize-none"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            <Button
              type="submit"
              size="icon"
              disabled={isLoading || !input.trim()}
              className="shrink-0"
            >
              <Send className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </form>
        </div>
      </CardFooter>
    </Card>
  );
} 