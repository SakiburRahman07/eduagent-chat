"use client";

import { MainLayout } from "./components/main-layout";
import ChatInterface from "./components/chat-interface";

export default function Home() {
  return (
    <MainLayout>
      <ChatInterface />
    </MainLayout>
  );
}
