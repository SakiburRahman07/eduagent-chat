"use client";

import { useState, useRef, useEffect } from "react";
import { Send, RefreshCw, ThumbsUp, ThumbsDown, Book, Lightbulb, Settings } from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { Textarea } from "@/app/components/ui/textarea";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/app/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/app/components/ui/avatar";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/app/components/ui/tooltip";
import { 
  Dialog, 
  DialogContent, 
  DialogDescription, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle, 
  DialogTrigger 
} from "@/app/components/ui/dialog";
import { Label } from "@/app/components/ui/label";
import { Input } from "@/app/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/app/components/ui/select";
import { Popover, PopoverContent, PopoverTrigger } from "@/app/components/ui/popover";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import ReactMarkdown from "react-markdown";
import { cn } from "@/app/lib/utils";

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState(Date.now().toString());
  const [userContext, setUserContext] = useState({
    academic_level: "undergraduate",
    interests: [],
    preferred_style: "balanced"
  });
  const [feedbackState, setFeedbackState] = useState({});
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
    setFeedbackState({});
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };
  
  const handleContextUpdate = (newContext) => {
    setUserContext(prev => ({
      ...prev,
      ...newContext
    }));
  };
  
  const provideFeedback = async (messageId, rating, feedbackText = "") => {
    try {
      // Store feedback locally
      setFeedbackState(prev => ({
        ...prev,
        [messageId]: { rating, feedbackText }
      }));
      
      // Send feedback to backend
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          message_id: messageId,
          rating: rating,
          feedback_text: feedbackText
        }),
      });
      
      if (!response.ok) {
        console.error("Failed to send feedback:", response.status);
      }
    } catch (error) {
      console.error("Error sending feedback:", error);
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
      // Send message to API with context
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: currentInput,
          conversation_id: conversationId,
          context: userContext
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("API error:", errorText);
        throw new Error(`Failed to send message: ${response.status}`);
      }

      const data = await response.json();
      console.log("Received response:", data);
      
      // Add AI response to the chat including the message_id for feedback
      setMessages((prev) => [
        ...prev,
        { 
          role: "assistant", 
          content: data.response,
          message_id: data.message_id,
          reasoning: data.reasoning
        },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Sorry, I encountered an error. Please try again.",
          reasoning: [{ type: "error", content: error.message }]
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  // Enhanced chat message component with feedback options
  const ChatMessage = ({ message, isUser, reasoning, messageId, feedbackState, onFeedback }) => {
    // Check if message content is a string or object
    const messageContent = typeof message === 'string' ? message : message.content;
    
    return (
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
          {/* Message content */}
          <div className={cn(
            "px-4 py-2.5 rounded-lg w-full",
            isUser 
              ? "bg-primary text-primary-foreground rounded-tr-none" 
              : "bg-secondary/30 text-foreground rounded-tl-none"
          )}>
            {isUser ? (
              <p className="text-sm whitespace-pre-wrap break-words">{messageContent}</p>
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
                  h2: ({ node, ...props }) => (
                    <h2 className="text-lg font-bold mt-4 mb-2" {...props} />
                  ),
                  // Enhanced citation rendering
                  h3: ({ node, ...props }) => {
                    if (props.children[0]?.toLowerCase().includes('references')) {
                      return <h3 className="text-base font-semibold mt-4 mb-2 border-t pt-2" {...props} />;
                    }
                    return <h3 className="text-base font-semibold mt-3 mb-1" {...props} />;
                  }
                }}
              >
                {messageContent}
              </ReactMarkdown>
            )}
          </div>
          
          {/* Feedback controls for AI messages */}
          {!isUser && typeof message === 'object' && message.message_id && (
            <div className="flex items-center mt-1 space-x-1 text-xs text-muted-foreground">
              {feedbackState[message.message_id] ? (
                <span className="italic">
                  {feedbackState[message.message_id].rating > 3 
                    ? "Thanks for the positive feedback!" 
                    : "Thanks for your feedback. I'll try to improve."}
                </span>
              ) : (
                <>
                  <span className="mr-1">Was this helpful?</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-6 w-6" 
                          onClick={() => onFeedback(message.message_id, 5)}
                        >
                          <ThumbsUp className="h-3 w-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Helpful</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button 
                          variant="ghost" 
                          size="icon" 
                          className="h-6 w-6" 
                          onClick={() => onFeedback(message.message_id, 2)}
                        >
                          <ThumbsDown className="h-3 w-3" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Not helpful</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Preferences Dialog Component
  const PreferencesDialog = () => (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="ml-auto">
          <Settings className="h-4 w-4 mr-2" />
          Preferences
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Study Preferences</DialogTitle>
          <DialogDescription>
            Customize how Study Buddy responds to your questions.
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="academic-level">Academic Level</Label>
            <Select 
              value={userContext.academic_level} 
              onValueChange={(value) => handleContextUpdate({ academic_level: value })}
            >
              <SelectTrigger id="academic-level">
                <SelectValue placeholder="Select level" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="elementary">Elementary School</SelectItem>
                <SelectItem value="middle">Middle School</SelectItem>
                <SelectItem value="high">High School</SelectItem>
                <SelectItem value="undergraduate">Undergraduate</SelectItem>
                <SelectItem value="graduate">Graduate</SelectItem>
                <SelectItem value="professional">Professional</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="grid gap-2">
            <Label htmlFor="learning-style">Learning Style</Label>
            <Select 
              value={userContext.preferred_style} 
              onValueChange={(value) => handleContextUpdate({ preferred_style: value })}
            >
              <SelectTrigger id="learning-style">
                <SelectValue placeholder="Select style" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="concise">Concise (Brief explanations)</SelectItem>
                <SelectItem value="balanced">Balanced (Moderate detail)</SelectItem>
                <SelectItem value="detailed">Detailed (In-depth explanations)</SelectItem>
                <SelectItem value="visual">Visual (Favor diagrams when possible)</SelectItem>
                <SelectItem value="practical">Practical (Focus on applications)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          <div className="grid gap-2">
            <Label htmlFor="interests">Topics of Interest (comma separated)</Label>
            <Input
              id="interests"
              placeholder="e.g., physics, literature, computer science"
              value={userContext.interests.join(", ")}
              onChange={(e) => handleContextUpdate({ 
                interests: e.target.value.split(",").map(i => i.trim()).filter(i => i) 
              })}
            />
          </div>
        </div>
        
        <DialogFooter>
          <Button type="button" onClick={() => {}}>Save Changes</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  return (
    <Card className="flex flex-col h-full border-none shadow-none bg-transparent">
      <CardHeader className="px-0 pt-0 flex flex-row items-center">
        <CardTitle className="sr-only">Chat with Study Buddy</CardTitle>
        <PreferencesDialog />
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
                  reasoning={message.reasoning}
                  messageId={message.message_id}
                  feedbackState={feedbackState}
                  onFeedback={provideFeedback}
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