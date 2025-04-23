import { NextResponse } from "next/server";

export async function POST(request) {
  try {
    const body = await request.json();
    
    // Forward the request to the Flask backend
    const response = await fetch("http://localhost:5000/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: body.message,
        conversation_id: body.conversation_id,
      }),
    });

    if (!response.ok) {
      console.error(`Backend error: ${response.status} ${response.statusText}`);
      const errorText = await response.text();
      console.error(`Error details: ${errorText}`);
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error forwarding request to backend:", error);
    return NextResponse.json(
      { error: "Failed to communicate with the AI service" },
      { status: 500 }
    );
  }
} 