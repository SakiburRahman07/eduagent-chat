import { NextResponse } from "next/server";

export async function POST(request) {
  try {
    const body = await request.json();
    
    // Forward the feedback to the Flask backend
    const response = await fetch("http://localhost:5000/api/feedback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        conversation_id: body.conversation_id,
        message_id: body.message_id,
        rating: body.rating,
        feedback_text: body.feedback_text || ""
      }),
    });

    if (!response.ok) {
      console.error(`Backend feedback error: ${response.status} ${response.statusText}`);
      const errorText = await response.text();
      console.error(`Feedback error details: ${errorText}`);
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    const data = await response.json();
    
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error forwarding feedback to backend:", error);
    return NextResponse.json(
      { error: "Failed to submit feedback" },
      { status: 500 }
    );
  }
} 