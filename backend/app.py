import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime

# LangGraph and LangChain imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import Tavily Search
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from tavily import TavilyClient
    tavily_available = True
except ImportError:
    print("Warning: Tavily packages not found. Web search functionality will be disabled.")
    tavily_available = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS to allow requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Flask-RESTX
api = Api(
    app,
    version='1.0',
    title='Study Buddy - AI Learning Assistant',
    description='An AI agent to help students with research and studying using knowledge tools and web search',
    doc='/swagger/'
)

# Create namespaces for API organization
chat_ns = Namespace('chat', description='Chat operations')
api.add_namespace(chat_ns, path='/api')

# Define API models
chat_input = api.model('ChatInput', {
    'message': fields.String(required=True, description='User message'),
    'conversation_id': fields.String(required=False, description='Conversation identifier')
})

chat_output = api.model('ChatOutput', {
    'response': fields.String(description='AI response'),
    'conversation_id': fields.String(description='Conversation identifier')
})

# Get API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable not set. Please set it in your .env file.")
    print("The application will not function correctly without this key.")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY and tavily_available:
    print("Warning: TAVILY_API_KEY environment variable not set. Web search functionality will be limited.")

# Set up Tavily search tool if API key is available
tavily_tools = []
if tavily_available and TAVILY_API_KEY:
    try:
        # General web search tool
        tavily_search_tool = TavilySearchResults(
            max_results=3,
            api_key=TAVILY_API_KEY,
            description="Search the web for current information on academic topics, general knowledge, and recent events."
        )
        tavily_tools.append(tavily_search_tool)
        
        # Create a Tavily extraction tool
        class TavilyExtraction:
            def __init__(self, api_key):
                self.client = TavilyClient(api_key=api_key)
                
            def run(self, url):
                """Extract the content from a specific URL."""
                try:
                    result = self.client.extract(url)
                    return result.get("content", "Unable to extract content from the URL.")
                except Exception as e:
                    return f"Error extracting content: {str(e)}"
        
        tavily_extraction_tool = TavilyExtraction(TAVILY_API_KEY)
        
        # Add the extraction tool to LangChain tool format
        from langchain.tools import Tool
        tavily_extract_tool = Tool(
            name="tavily_extract",
            func=tavily_extraction_tool.run,
            description="Extract and summarize content from a specific URL. Use this when you want to get detailed information from a webpage."
        )
        
        tavily_tools.append(tavily_extract_tool)
        print("Tavily search tools initialized successfully.")
    except Exception as e:
        print(f"Error initializing Tavily tools: {str(e)}")
        print("Web search functionality will be disabled.")
        tavily_tools = []

# Set up Wikipedia tool
try:
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=4000)
    wiki_tool = WikipediaQueryRun(
        api_wrapper=wiki_wrapper,
        description="Search Wikipedia for explanations of academic concepts, historical events, scientific theories, and general knowledge topics."
    )
    print("Wikipedia tool initialized successfully.")
except Exception as e:
    print(f"Error initializing Wikipedia tool: {str(e)}")
    print("Wikipedia search functionality will be disabled.")
    wiki_tool = None

# Set up ArXiv tool for academic papers
try:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=4000)
    arxiv_tool = ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        description="Search for academic papers and scientific research on arXiv. Use this for finding scholarly information on advanced topics."
    )
    print("ArXiv tool initialized successfully.")
except Exception as e:
    print(f"Error initializing ArXiv tool: {str(e)}")
    print("ArXiv search functionality will be disabled.")
    arxiv_tool = None

# Combine all working tools
tools = []
if wiki_tool:
    tools.append(wiki_tool)
if arxiv_tool:
    tools.append(arxiv_tool)
tools.extend(tavily_tools)

if not tools:
    print("ERROR: No search tools are available. The assistant will have very limited functionality.")
else:
    print(f"Successfully initialized {len(tools)} search tools.")

# Define state type for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# A student-friendly system prompt
system_prompt = """You are **Study Buddy**, a knowledgeable and friendly AI assistant built to help students excel in learning and research.

🎯 Your goals:
- Provide **clear, well-structured, and academically sound** responses.
- Support students across **all education levels**, adapting complexity accordingly.
- Use **available tools** (Wikipedia, ArXiv, Tavily search, URL extractor) to back up answers with verified sources.

🧠 When answering:
1. Break down complex concepts into **digestible parts**.
2. Synthesize insights from **multiple tools** and **cross-reference** if needed.
3. Include **diagrams, tables, or step-by-step explanations** if helpful.
4. When using search tools, explain briefly **what was searched** and **why the tool was selected**.
5. Add **insightful follow-up questions** or suggestions to encourage deeper thinking.

📚 Formatting Guidelines:
- Use **headings**, **bullets**, and **short paragraphs** to improve readability.
- Always end with a **"References" section** using proper academic citation formats.

🛠 Tool Usage Rules:
- Use Wikipedia for well-established concepts.
- Use ArXiv for **scientific depth** or cutting-edge research.
- Use Tavily for **recent/current events**, or when broader sources are needed.
- Use the URL extractor when a **specific page link** is provided.

🤝 Tone:
- Always be **encouraging and patient**.
- Avoid sounding like a search engine — add context, summaries, and helpful commentary.

FORMAT FOR REFERENCES:
Always end your response with a "References" section that lists your sources using one of these citation styles:

For websites:
- [Website Title]. (Year if available). Retrieved from [URL]
  Example: Khan Academy. (2022). Retrieved from https://www.khanacademy.org/science/biology/photosynthesis

For books:
- [Author Last Name, Initials]. (Year). [Book Title]. [Publisher]
  Example: Campbell, N.A. & Reece, J.B. (2005). Biology (7th ed.). Benjamin Cummings

For academic papers (when using ArXiv):
- [Author(s)]. (Year). [Paper Title]. arXiv:[ID]
  Example: Smith, J. & Jones, T. (2021). Advances in Machine Learning. arXiv:2101.12345

For Wikipedia:
- Wikipedia. (n.d.). [Article Title]. Retrieved [current date]
  Example: Wikipedia. (n.d.). Photosynthesis. Retrieved April 23, 2023

If tool outputs are used directly:
- [Tool Name] search results for "[query]"
  Example: Wikipedia search results for "quantum mechanics"

🧪 Example Approach:
If the student asks, "Explain quantum entanglement," you should:
- Define it in simple terms.
- Mention real-world applications (e.g., quantum cryptography).
- Use Wikipedia and ArXiv together.
- End with: "Would you like to see a visual diagram or read a specific paper on this topic?"

Remember, your job is not just to **answer**, but to **empower students to learn deeply**.
EVERY RESPONSE MUST INCLUDE A REFERENCES SECTION.
"""

# Initialize LLM - FIXED: Don't use system_message parameter in bind_tools
try:
    # Initialize Groq model
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="Gemma2-9b-It"
    )
    
    # Bind tools without system_message parameter
    llm_with_tools = llm.bind_tools(tools=tools)
    
    print("LLM initialized successfully with Groq's Gemma2-9b-It model.")
except Exception as e:
    print(f"ERROR: Failed to initialize the LLM: {str(e)}")
    print("The application will not function correctly. Please check your GROQ_API_KEY.")

# Define the chatbot node function
def chatbot(state: State):
    try:
        # Convert messages to proper message types with better error handling
        lc_messages = []
        for message in state["messages"]:
            # Check if message is a tuple (expected format: (role, content))
            if isinstance(message, tuple) and len(message) == 2:
                msg_type, content = message
                if msg_type == "user":
                    # Check if topic might benefit from up-to-date information
                    current_topics = ["recent", "latest", "new", "current", "today", "2023", "2024", "2025"]
                    should_search = any(topic in content.lower() for topic in current_topics)
                    
                    # If it's a follow-up question in an existing conversation, don't modify
                    if len(state["messages"]) > 1:
                        lc_messages.append(HumanMessage(content=content))
                    # For new conversations on current topics, encourage using search
                    elif should_search:
                        enhanced_prompt = f"{content}\n\nPlease use your search tools to find the most up-to-date information on this topic."
                        lc_messages.append(HumanMessage(content=enhanced_prompt))
                    else:
                        lc_messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    lc_messages.append(AIMessage(content=content))
            # Check if message is already a LangChain message object
            elif hasattr(message, 'type'):
                lc_messages.append(message)
            else:
                print(f"Warning: Skipping message with unexpected format: {message}")
        
        # Add system message at the beginning
        system_message = SystemMessage(content=system_prompt)
        lc_messages.insert(0, system_message)
        
        # Call the LLM with properly formatted messages
        result = llm_with_tools.invoke(lc_messages)
        
        # Ensure the response has references
        if hasattr(result, 'content') and result.content:
            # Check for references section
            ref_indicators = ["References", "REFERENCES", "Reference:", "Sources:", "SOURCES"]
            has_references = any(indicator in result.content for indicator in ref_indicators)
            
            if not has_references:
                # Add a more comprehensive references section if missing
                current_date = datetime.now().strftime("%B %d, %Y")
                additional_content = "\n\n## References\n"
                
                # If we can determine which tools were used (simplified example)
                if "wikipedia" in result.content.lower():
                    additional_content += "- Wikipedia. (n.d.). Retrieved " + current_date + "\n"
                
                if "arxiv" in result.content.lower():
                    additional_content += "- Various academic papers from arXiv database. Retrieved " + current_date + "\n"
                    
                # Always add a general reference
                additional_content += "- Study Buddy AI Assistant. (2023). Educational content synthesis.\n"
                additional_content += "\nNote: In future responses, I'll provide more specific references for each piece of information."
                
                result.content += additional_content
        
        return {"messages": [result]}
    except Exception as e:
        print(f"Error in chatbot node: {str(e)}")
        # Return a generic error message as a fallback
        return {"messages": [AIMessage(content="I'm sorry, I encountered an error processing your request. Please try again.")]}

# Create and compile the graph
def create_graph():
    try:
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        return graph_builder.compile()
    except Exception as e:
        print(f"Error creating graph: {str(e)}")
        return None

graph = create_graph()
if not graph:
    print("ERROR: Failed to initialize the graph. The application will not function correctly.")

# Conversation history storage
# In a production environment, this should be replaced with a database
conversations = {}

@chat_ns.route('/chat')
class ChatResource(Resource):
    @chat_ns.expect(chat_input)
    @chat_ns.marshal_with(chat_output)
    @chat_ns.doc(
        responses={
            200: "Success",
            400: "Invalid input",
            500: "Server error"
        },
        description="Ask a question to the Study Buddy AI assistant"
    )
    def post(self):
        """Chat with the Study Buddy AI assistant"""
        if not graph:
            return {
                'response': "I'm sorry, the Study Buddy service is not initialized properly. Please contact support.",
                'conversation_id': 'error'
            }, 500
            
        try:
            data = request.json
            user_input = data.get('message', '')
            conversation_id = data.get('conversation_id', 'default')
            
            print(f"Received request with message: '{user_input[:50]}...' and conversation_id: {conversation_id}")
            
            # Get or initialize conversation history
            if conversation_id not in conversations:
                conversations[conversation_id] = []
            
            # Add user message to history
            conversations[conversation_id].append(("user", user_input))
            
            # Process with LangGraph
            events = graph.stream(
                {"messages": conversations[conversation_id]},
                stream_mode="values"
            )
            
            # Extract the AI response from the last event
            ai_responses = []
            for event in events:
                # Add each message to the history
                message = event["messages"][-1]
                if message.type == "ai":
                    if hasattr(message, 'content') and message.content:
                        ai_responses.append(message.content)
                    
                    # Add AI message to conversation history
                    conversations[conversation_id].append(("ai", message.content if hasattr(message, 'content') else ""))
            
            # Return the last AI response
            response = ai_responses[-1] if ai_responses else "I couldn't generate a response"
            print(f"Sending response: '{response[:50]}...'")
            return {
                'response': response,
                'conversation_id': conversation_id
            }
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return {
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'conversation_id': conversation_id if 'conversation_id' in locals() else 'error'
            }, 500

# Add a health check endpoint
@api.route('/health')
class HealthResource(Resource):
    def get(self):
        """Check if the API is running"""
        status = "DEGRADED" if not graph else "OK"
        message = "Study Buddy AI Assistant is running, but with degraded functionality." if not graph else "Study Buddy AI Assistant is running"
        return {'status': status, 'message': message}, 200

if __name__ == '__main__':
    print("Starting Study Buddy AI Assistant...")
    app.run(debug=True, host='0.0.0.0', port=5000) 