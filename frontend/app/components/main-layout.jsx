"use client";

import { BookOpen, Github } from "lucide-react";
import { ThemeToggle } from "./ui/theme-toggle";
import Link from "next/link";
import { Button } from "./ui/button";
import { Separator } from "./ui/separator";
import { cn } from "@/app/lib/utils";

export function MainLayout({ children }) {
  return (
    <div className="flex flex-col min-h-screen bg-background">
      <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-14 items-center">
          <div className="flex items-center gap-2 font-semibold">
            <BookOpen className="h-5 w-5 text-primary" />
            <span>Study Buddy</span>
          </div>
          <div className="ml-auto flex items-center gap-4">
            <span className="text-xs text-muted-foreground hidden md:inline">
              Powered by Groq & LangGraph
            </span>
            <div className="flex items-center gap-1">
              <Link href="https://github.com" target="_blank">
                <Button variant="ghost" size="icon" className="h-9 w-9">
                  <Github className="h-5 w-5" />
                  <span className="sr-only">GitHub</span>
                </Button>
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>
      <main className="flex-1 container py-6 md:py-8 flex">
        <div className={cn("w-full max-w-4xl mx-auto flex-1 overflow-hidden")}>
          {children}
        </div>
      </main>
    </div>
  );
} 