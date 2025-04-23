# Study Buddy - AI Learning Assistant

This project implements an educational AI assistant using LangGraph for the backend and Next.js for the frontend. The assistant helps students with research and studying by combining information from multiple sources including Wikipedia, ArXiv, and web search.

## Project Structure

- `backend/`: Flask API with LangGraph-powered educational assistant
- `frontend/`: Next.js web application for the chat interface

## Features

- **Multi-Source Research**: Combines information from Wikipedia, ArXiv, and web searches
- **URL Content Extraction**: Can extract and summarize content from specific webpages
- **Educational Focus**: Designed specifically to help students with academic topics
- **Conversation History**: Preserves context throughout the learning session
- **User-Friendly Interface**: Modern, responsive design with dark mode support

## Requirements

- Python 3.8+
- Node.js 18+
- Groq API key (required)
- Tavily API key (optional but recommended for web search)

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

4. Install the dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file from the example:
```bash
cp .env.example .env
```

6. Add your API keys to the `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=tvly-your_tavily_api_key_here
```

7. Start the Flask server:
```bash
python app.py
```

The backend server will run on `http://localhost:5000`.

You can access the Swagger UI documentation at `http://localhost:5000/swagger/` to test the API endpoints directly from your browser.

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install the dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend application will run on `http://localhost:3000`.

## Example Questions

You can ask the Study Buddy a variety of academic questions, such as:
- "Can you explain quantum mechanics in simple terms?"
- "Help me understand the causes of World War II"
- "What are the latest research papers on machine learning?"
- "Explain the process of photosynthesis with examples"
- "What is the significance of Hemingway's writing style in literature?"

## API Endpoints

- `POST /api/chat` - Send a message to the Study Buddy
  - Request body:
    ```json
    {
      "message": "Your question here",
      "conversation_id": "optional_conversation_id"
    }
    ```
  - Response:
    ```json
    {
      "response": "AI response here",
      "conversation_id": "conversation_id"
    }
    ```

- `GET /health` - Check if the API is running
  - Response:
    ```json
    {
      "status": "OK",
      "message": "Study Buddy AI Assistant is running"
    }
    ```

## Tools and Technologies

- **LangGraph**: For creating the agent workflow
- **LangChain**: For connecting to various knowledge sources
- **Groq**: LLM provider for fast, accurate responses
- **Tavily**: For web search and content extraction
- **Next.js**: Frontend framework
- **TailwindCSS & shadcn/ui**: For styling the interface
- **Flask & Flask-RESTX**: Backend API framework

## Troubleshooting

### Package Compatibility Issues

This project uses specific versions of packages to ensure compatibility:

- **Flask 2.0.3** and **Werkzeug 2.0.3**: These specific versions work together and with Flask-RESTX 1.1.0. Newer versions of Flask require newer versions of Werkzeug, which are not compatible with the current version of Flask-RESTX.

- **Flask-RESTX 1.1.0**: This version requires Werkzeug's `__version__` attribute, which was removed in newer Werkzeug versions.

If you encounter dependency conflicts or import errors, make sure you're using the exact package versions specified in `requirements.txt`. It's recommended to create a fresh virtual environment to avoid conflicts with previously installed packages.

## Credits

- LangGraph: https://github.com/langchain-ai/langgraph
- Groq: https://groq.com/
- Next.js: https://nextjs.org/
- Flask: https://flask.palletsprojects.com/
- Flask-RESTX: https://flask-restx.readthedocs.io/ 