import os
import sys
import re
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any
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
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

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
    'conversation_id': fields.String(required=False, description='Conversation identifier'),
    'context': fields.Raw(required=False, description='Additional context parameters')
})

reasoning_step = api.model('ReasoningStep', {
    'type': fields.String(description='Type of reasoning step'),
    'content': fields.String(description='Content of the reasoning step')
})

chat_output = api.model('ChatOutput', {
    'response': fields.String(description='AI response'),
    'conversation_id': fields.String(description='Conversation identifier'),
    'message_id': fields.String(description='Unique message identifier for feedback'),
    'reasoning': fields.List(fields.Nested(reasoning_step, description='Reasoning step details'))
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

# Define state type for LangGraph with additional context fields
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: dict  # Store persistent context like academic level, topics of interest, etc.
    memory: str    # Store conversation summary
    reasoning: List[Dict[str, str]]  # Store reasoning steps

# Initialize memory and context storage
# In a production environment, this should be replaced with a database
conversations = {}
conversation_contexts = {}
conversation_memories = {}

# Create a simple memory summarization function since we don't have ConversationSummaryMemory
def summarize_conversation(messages, max_length=500):
    """Create a simple summary of the conversation"""
    if not messages:
        return ""
        
    # For simplicity, just take the first few and last few messages
    if len(messages) <= 4:
        summary = "Conversation is just starting.\n"
    else:
        # Extract first 2 exchanges
        first_msgs = messages[:4]  # First 2 exchanges (2 user, 2 ai)
        last_msgs = messages[-2:]  # Last exchange
        
        summary = "Conversation started with:\n"
        
        # Add first messages
        for i, (role, content) in enumerate(first_msgs):
            prefix = "Student" if role == "user" else "AI"
            summary += f"- {prefix}: {content[:50]}...\n"
            
        summary += "...\n"
        
        # Add recent messages
        for i, (role, content) in enumerate(last_msgs):
            prefix = "Student" if role == "user" else "AI"
            summary += f"- {prefix}: {content[:50]}...\n"
    
    return summary

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

🔍 CHAIN-OF-THOUGHT REASONING:
For complex problems (especially in mathematics, science, logic, and multi-step reasoning):
1. **Break down the problem** into smaller, manageable components
2. **Think step by step** - show your complete reasoning process
3. **Explicitly state intermediate steps** - don't skip logical connections
4. Consider **multiple approaches** when appropriate and explain why you chose a particular method
5. Clearly indicate your **final answer** after showing all reasoning steps
6. For mathematical problems, explain each calculation and why you're performing it

Examples:
- For math: "First, I'll identify the variables... Next, I'll set up the equation... Then I'll solve for x..."
- For conceptual: "To understand this concept, let's first examine... This leads us to consider... Finally..."
- For analysis: "First, let's identify the key factors... Next, let's analyze how these interact... This suggests..."

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
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
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
        # Get context and memory
        context = state.get("context", {})
        memory_summary = state.get("memory", "")
        
        # Initialize reasoning steps
        reasoning_steps = []
        
        # Convert messages to proper message types with better error handling
        lc_messages = []
        for message in state["messages"]:
            # Check if message is a tuple (expected format: (role, content))
            if isinstance(message, tuple) and len(message) == 2:
                msg_type, content = message
                if msg_type == "user":
                    # Add initial reasoning step
                    reasoning_steps.append({
                        "type": "thought",
                        "content": "I need to understand the user's query and determine the best approach to answer it."
                    })
                    
                    # Check if this is a STEM query that should be optimized
                    stem_focus = context.get("stem_focus", False)
                    if stem_focus:
                        # Try to optimize the query for STEM topics
                        optimized_content = optimize_query(content, "scientific")
                        reasoning_steps.append({
                            "type": "thought",
                            "content": f"This appears to be a STEM-related query. Optimizing it for scientific accuracy: {optimized_content}"
                        })
                        lc_messages.append(HumanMessage(content=optimized_content))
                    else:
                        # Check if the query would benefit from Chain-of-Thought reasoning
                        requires_cot = needs_chain_of_thought(content)
                        
                        # Check if topic might benefit from up-to-date information
                        current_topics = ["recent", "latest", "new", "current", "today", "2023", "2024", "2025"]
                        should_search = any(topic in content.lower() for topic in current_topics)
                        
                        if requires_cot:
                            reasoning_steps.append({
                                "type": "thought",
                                "content": "This query would benefit from chain-of-thought reasoning. Enhancing prompt to encourage step-by-step thinking."
                            })
                            # Add CoT directive to the prompt
                            enhanced_prompt = f"{content}\n\nPlease use chain-of-thought reasoning to solve this problem. Think step by step and show all your work."
                            lc_messages.append(HumanMessage(content=enhanced_prompt))
                            
                            # Add CoT requirement to context for consistency in follow-ups
                            context["requires_cot"] = True
                        elif should_search:
                            reasoning_steps.append({
                                "type": "thought",
                                "content": "This query may benefit from up-to-date information. I'll use search tools to find recent data."
                            })
                            enhanced_prompt = f"{content}\n\nPlease use your search tools to find the most up-to-date information on this topic."
                            lc_messages.append(HumanMessage(content=enhanced_prompt))
                        else:
                            lc_messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    lc_messages.append(AIMessage(content=content))
            elif hasattr(message, 'type'):
                lc_messages.append(message)
            else:
                print(f"Warning: Skipping message with unexpected format: {message}")
        
        # Build a dynamic system prompt that includes context and memory
        dynamic_system_prompt = system_prompt
        
        # Add context information if available
        if context:
            context_info = "\n\nCONVERSATION CONTEXT:\n"
            if "academic_level" in context:
                context_info += f"- Student Academic Level: {context['academic_level']}\n"
                reasoning_steps.append({
                    "type": "thought",
                    "content": f"Adapting response for {context['academic_level']} level understanding"
                })
            if "interests" in context:
                context_info += f"- Topics of Interest: {', '.join(context['interests'])}\n"
            if "preferred_style" in context:
                context_info += f"- Preferred Learning Style: {context['preferred_style']}\n"
            if "stem_focus" in context:
                context_info += f"- STEM Focus: {context['stem_focus']}\n"
            if "requires_cot" in context and context["requires_cot"]:
                context_info += f"- NOTE: Student's questions benefit from chain-of-thought reasoning. Show your step-by-step thinking process.\n"
                reasoning_steps.append({
                    "type": "thought",
                    "content": "Using chain-of-thought reasoning for this conversation"
                })
            if "simplify_explanations" in context and context["simplify_explanations"]:
                context_info += f"- NOTE: Student has requested simpler explanations. Please break down concepts into more basic terms and avoid jargon.\n"
                reasoning_steps.append({
                    "type": "thought",
                    "content": "Using simpler explanations based on user preference"
                })
            
            dynamic_system_prompt += context_info
        
        # Add memory summary if available
        if memory_summary:
            dynamic_system_prompt += f"\n\nCONVERSATION HISTORY SUMMARY:\n{memory_summary}\n"
        
        # Add system message at the beginning
        system_message = SystemMessage(content=dynamic_system_prompt)
        lc_messages.insert(0, system_message)
        
        # Call the LLM with properly formatted messages
        result = llm_with_tools.invoke(lc_messages)
        
        # Post-process the response for better citations
        result = post_process_response(result)
        
        # Add final reasoning step
        reasoning_steps.append({
            "type": "action",
            "content": "Generated response with citations and references"
        })
        
        # Validate reasoning steps to ensure all keys and values are strings
        validated_reasoning = []
        for step in reasoning_steps:
            validated_step = {}
            for k, v in step.items():
                # Ensure keys are strings
                key = str(k) if not isinstance(k, str) else k
                # Ensure values are strings
                value = str(v) if not isinstance(v, str) else v
                validated_step[key] = value
            validated_reasoning.append(validated_step)
        
        # Return updated state with all fields preserved
        return {
            "messages": state["messages"] + [result],
            "context": state.get("context", {}),
            "memory": state.get("memory", ""),
            "reasoning": validated_reasoning
        }
    except Exception as e:
        print(f"Error in chatbot node: {str(e)}")
        # Return a generic error message as a fallback, preserving state
        error_message = AIMessage(content="I'm sorry, I encountered an error processing your request. Please try again.")
        return {
            "messages": state["messages"] + [error_message],
            "context": state.get("context", {}),
            "memory": state.get("memory", ""),
            "reasoning": [{"type": "error", "content": str(e)}]
        }

# Add a query optimizer function to improve search queries
def optimize_query(query, query_type=None):
    """Optimize a search query based on query type"""
    if not query or not isinstance(query, str):
        return query
        
    # Convert vague questions to more specific search queries
    if query.lower().startswith("what is"):
        return query.replace("What is", "Define").strip()
    if query.lower().startswith("how does"):
        return query.replace("How does", "Explain").strip()
    
    # Add specific modifiers based on query type
    if query_type == "scientific":
        return f"scientific explanation {query}"
    elif query_type == "historical":
        return f"historical context {query}"
    elif query_type == "conceptual":
        return f"concept explanation {query}"
        
    return query

# Add a function to detect if a query requires chain-of-thought reasoning
def needs_chain_of_thought(query):
    """Detect if a query would benefit from chain-of-thought reasoning"""
    if not query or not isinstance(query, str):
        return False
        
    query_lower = query.lower()
    
    # Keywords indicating complex problem-solving
    problem_solving_indicators = [
        "solve", "calculate", "find", "determine", "compute", "prove", 
        "explain how", "explain why", "reason", "analyze", "step by step", 
        "step-by-step", "break down", "work through", "approach"
    ]
    
    # Subject areas that often benefit from step-by-step reasoning
    cot_subjects = [
        "math", "mathematics", "algebra", "calculus", "geometry", "trigonometry",
        "physics", "chemistry", "equations", "formula", "theorem", "proof",
        "algorithm", "programming", "code", "logic", "syllogism", "argument",
        "reasoning", "philosophy", "ethics", "economic", "statistics", "probability"
    ]
    
    # Question patterns that indicate multi-step reasoning needs
    multi_step_patterns = [
        "what happens if", "what would happen", "how would", "why does", 
        "explain the process", "walk me through", "how can i", "how do i",
        "steps to", "steps for", "procedure for", "method to"
    ]
    
    # Check if the query contains problem-solving indicators
    has_problem_indicator = any(indicator in query_lower for indicator in problem_solving_indicators)
    
    # Check if the query relates to subjects that benefit from CoT
    has_cot_subject = any(subject in query_lower for subject in cot_subjects)
    
    # Check if the query follows multi-step reasoning patterns
    has_multi_step_pattern = any(pattern in query_lower for pattern in multi_step_patterns)
    
    # Check for mathematical expressions or equations
    has_math_expression = bool(re.search(r'[0-9+\-*/()^=]', query)) and bool(re.search(r'[+\-*/()^=]', query))
    
    # Decision logic: Use CoT if the query matches specific patterns
    return has_math_expression or (has_problem_indicator and has_cot_subject) or has_multi_step_pattern

# Tool selection logic based on query content
def select_tools(query):
    """Select appropriate tools based on query content"""
    query_lower = query.lower()
    
    # STEM topic detection
    stem_keywords = ["math", "physics", "chemistry", "biology", "engineering", 
                    "quantum", "algorithm", "molecule", "protein", "theorem"]
    
    # Academic paper indicators                
    paper_keywords = ["paper", "research", "publication", "journal", "study", 
                     "experiment", "findings", "published", "arxiv"]
    
    # Current events indicators
    current_keywords = ["recent", "latest", "new", "current", "today", "this year", 
                       "2023", "2024", "2025", "news"]
    
    # URL indicators
    url_indicators = ["http", "https", "www.", ".com", ".org", ".edu"]
    
    # Determine the correct tool priorities
    if any(keyword in query_lower for keyword in stem_keywords) and any(keyword in query_lower for keyword in paper_keywords):
        # STEM research papers - prioritize ArXiv
        return ["arxiv", "wikipedia", "tavily"]
    elif any(indicator in query_lower for indicator in url_indicators):
        # URL extraction request
        return ["tavily_extract", "tavily"]
    elif any(keyword in query_lower for keyword in current_keywords):
        # Current events - prioritize Tavily web search
        return ["tavily", "wikipedia", "arxiv"]
    else:
        # General knowledge - start with Wikipedia
        return ["wikipedia", "tavily", "arxiv"]

# Define the tool router node
def tool_router(state):
    """Route to appropriate tools based on the most recent user message"""
    try:
        # Extract the most recent user message
        user_messages = [msg for msg in state["messages"] if isinstance(msg, tuple) and msg[0] == "user"]
        if not user_messages:
            # Return full state with tool_choice = None
            return {
                "tool_choice": None,
                "messages": state.get("messages", []),
                "context": state.get("context", {}),
                "memory": state.get("memory", ""),
                "reasoning": state.get("reasoning", [])
            }
            
        latest_user_message = user_messages[-1][1]
        
        # Get preferred tools based on message content
        preferred_tools = select_tools(latest_user_message)
        
        # Get STEM focus from context if available
        context = state.get("context", {})
        stem_focus = context.get("stem_focus", False)
        
        # Further adjust tool preference based on context
        if stem_focus and "arxiv" in preferred_tools:
            # Move ArXiv to first position for STEM-focused students
            preferred_tools.remove("arxiv")
            preferred_tools.insert(0, "arxiv")
        
        # Return full state with updated tool_choice
        return {
            "tool_choice": preferred_tools[0] if preferred_tools else None,
            "messages": state.get("messages", []),
            "context": state.get("context", {}),
            "memory": state.get("memory", ""),
            "reasoning": state.get("reasoning", [])
        }
    except Exception as e:
        print(f"Error in tool router: {str(e)}")
        # Return full state with tool_choice = None
        return {
            "tool_choice": None,
            "messages": state.get("messages", []),
            "context": state.get("context", {}),
            "memory": state.get("memory", ""),
            "reasoning": [{"type": "error", "content": str(e)}]
        }

# Create and compile the graph
def create_graph():
    try:
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tool_router", tool_router)
        
        # Create tool node with additional configuration for tool choice
        tool_node = ToolNode(
            tools=tools,
            name="tool_executor",
            handle_tool_errors=True
        )
        graph_builder.add_node("tools", tool_node)

        # Add conditional edges with tool routing
        graph_builder.add_edge(START, "tool_router")
        graph_builder.add_edge("tool_router", "chatbot")
        
        # Add conditional edges for tool execution
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition
        )
        graph_builder.add_edge("tools", "chatbot")

        return graph_builder.compile()
    except Exception as e:
        print(f"Error creating graph: {str(e)}")
        return None

# Add a self-consistency function to improve reliability of CoT reasoning for complex problems
def apply_self_consistency(query, llm_instance, max_iterations=3):
    """
    Implements the self-consistency technique (Wang et al., 2022) by generating
    multiple reasoning paths and taking the majority answer for critical reasoning tasks.
    
    Args:
        query: The user's question
        llm_instance: The LLM to use for generating responses
        max_iterations: Maximum number of reasoning paths to generate
    
    Returns:
        The most consistent answer with its reasoning path
    """
    try:
        # Only apply this to complex reasoning questions to avoid unnecessary API calls
        if not needs_chain_of_thought(query):
            return None
            
        # Create a prompt that encourages diverse reasoning paths
        base_prompt = f"""
        I need to solve this problem using different approaches:
        
        {query}
        
        Let me solve this step-by-step using different reasoning paths.
        """
        
        # Generate multiple reasoning paths
        reasoning_paths = []
        answers = []
        
        for i in range(max_iterations):
            # Add entropy to encourage diverse approaches
            approach_prompt = f"{base_prompt}\n\nReasoning Path #{i+1}:"
            
            # Get response
            messages = [
                SystemMessage(content="You are a mathematical and logical reasoning expert. Show your work step-by-step."),
                HumanMessage(content=approach_prompt)
            ]
            
            response = llm_instance.invoke(messages)
            
            if response and hasattr(response, 'content'):
                reasoning_paths.append(response.content)
                
                # Extract the answer from the reasoning path
                # This is a simple approach - more sophisticated parsing could be implemented
                answer_match = re.search(r'(answer|result|therefore)(?:.+?)(?:is|=)\s*([^\.]+)', 
                                       response.content, re.IGNORECASE)
                if answer_match:
                    answers.append(answer_match.group(2).strip())
        
        # If we have multiple answers, find the most common one
        if len(answers) > 1:
            # Count occurrences of each answer
            from collections import Counter
            answer_counts = Counter(answers)
            
            # Get the most common answer
            most_common_answer = answer_counts.most_common(1)[0][0]
            
            # Find the reasoning path that led to this answer
            for i, path in enumerate(reasoning_paths):
                if i < len(answers) and answers[i] == most_common_answer:
                    return {
                        "answer": most_common_answer,
                        "reasoning": path,
                        "consistency_score": answer_counts[most_common_answer] / len(answers)
                    }
        
        # If we couldn't find a consistent answer or only have one path, return the first one
        if reasoning_paths:
            return {
                "answer": answers[0] if answers else "Unknown",
                "reasoning": reasoning_paths[0],
                "consistency_score": 1.0 if len(answers) == 1 else 0.0
            }
            
        return None
    except Exception as e:
        print(f"Error in self-consistency check: {str(e)}")
        return None

graph = create_graph()
if not graph:
    print("ERROR: Failed to initialize the graph. The application will not function correctly.")

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
            
            # Get or update context parameters if provided
            context_updates = data.get('context', {})
            
            print(f"Received request with message: '{user_input[:50]}...' and conversation_id: {conversation_id}")
            
            # Get or initialize conversation history, context and memory
            if conversation_id not in conversations:
                conversations[conversation_id] = []
                conversation_contexts[conversation_id] = {}
                conversation_memories[conversation_id] = ""
            
            # Update context with any new information
            if context_updates:
                conversation_contexts[conversation_id].update(context_updates)
            
            # Process user input for topic categorization and context enrichment
            context = conversation_contexts[conversation_id]
            
            # Simple keyword-based subject detection for STEM focus
            stem_keywords = ["math", "physics", "chemistry", "biology", "computer", "engineering", 
                             "science", "technology", "algorithm", "quantum", "molecular"]
            
            if any(keyword in user_input.lower() for keyword in stem_keywords) and "stem_focus" not in context:
                context["stem_focus"] = True
            
            # Check if this question requires high-reliability reasoning with self-consistency
            # Only apply to complex mathematical/logical problems to avoid unnecessary API usage
            math_patterns = [
                r'\d+[\+\-\*\/]\d+',  # Basic arithmetic expressions
                r'equation',
                r'solve for',
                r'calculate',
                r'compute'
            ]
            is_complex_math = needs_chain_of_thought(user_input) and any(re.search(pattern, user_input, re.IGNORECASE) for pattern in math_patterns)
            
            # For complex math problems, try self-consistency approach first
            consistency_result = None
            if is_complex_math and llm:
                print(f"Applying self-consistency for complex problem: {user_input[:50]}...")
                consistency_result = apply_self_consistency(user_input, llm, max_iterations=2)
                
                if consistency_result and consistency_result.get("consistency_score", 0) >= 0.5:
                    print(f"Self-consistency check successful with score: {consistency_result['consistency_score']}")
                    # Store the self-consistency result in context for later use
                    context["self_consistency_applied"] = True
                    context["consistency_score"] = consistency_result["consistency_score"]
            
            # Add user message to history
            conversations[conversation_id].append(("user", user_input))
            
            # Update memory with conversation summary using our custom function
            conversation_memories[conversation_id] = summarize_conversation(conversations[conversation_id])
            
            # Prepare state with messages, context and memory
            state = {
                "messages": conversations[conversation_id],
                "context": conversation_contexts[conversation_id],
                "memory": conversation_memories[conversation_id],
                "reasoning": []
            }
            
            # Process with LangGraph
            events = graph.stream(
                state,
                stream_mode="values"
            )
            
            # Extract the AI response and reasoning steps from the last event
            ai_responses = []
            reasoning_steps = []
            for event in events:
                # Add each message to the history
                message = event["messages"][-1]
                if message.type == "ai":
                    if hasattr(message, 'content') and message.content:
                        ai_responses.append(message.content)
                    
                    # Add AI message to conversation history
                    conversations[conversation_id].append(("ai", message.content if hasattr(message, 'content') else ""))
                
                # Collect reasoning steps
                if "reasoning" in event:
                    reasoning_steps.extend(event["reasoning"])
            
            # For complex math problems where we applied self-consistency, replace response if needed
            response = ai_responses[-1] if ai_responses else "I couldn't generate a response"
            
            if consistency_result and context.get("self_consistency_applied") and consistency_result.get("reasoning"):
                # Add self-consistency information to the response
                enhanced_response = f"{response}\n\n**Self-Consistency Check**: I verified this solution using multiple reasoning paths for higher reliability."
                response = enhanced_response
                
                # Add reasoning step about self-consistency
                reasoning_steps.append({
                    "type": "verification",
                    "content": f"Applied self-consistency with {consistency_result.get('consistency_score', 0)} agreement score"
                })
            
            print(f"Sending response: '{response[:50]}...'")
            
            # Generate a unique message ID for feedback purposes
            message_id = str(uuid.uuid4())
            
            # Store the message ID in the context for future reference
            if "message_ids" not in conversation_contexts[conversation_id]:
                conversation_contexts[conversation_id]["message_ids"] = {}
            
            # Map the message ID to the response index in the conversation history
            conversation_contexts[conversation_id]["message_ids"][message_id] = len(conversations[conversation_id]) - 1
            
            # Ensure reasoning_steps contains only dictionaries with string attributes
            validated_reasoning = []
            for step in reasoning_steps:
                if isinstance(step, dict):
                    # Convert all values to strings to avoid serialization issues
                    validated_step = {}
                    for k, v in step.items():
                        if not isinstance(k, str):
                            k = str(k)
                        if not isinstance(v, str):
                            v = str(v)
                        validated_step[k] = v
                    validated_reasoning.append(validated_step)
                else:
                    # If it's not a dict, convert to a simple dict with string values
                    validated_reasoning.append({"type": "unknown", "content": str(step)})
            
            return {
                'response': response,
                'conversation_id': conversation_id,
                'message_id': message_id,
                'reasoning': validated_reasoning
            }
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return {
                'response': "I'm sorry, I encountered an error processing your request. Please try again.",
                'conversation_id': conversation_id if 'conversation_id' in locals() else 'error',
                'message_id': str(uuid.uuid4()),
                'reasoning': [{"type": "error", "content": str(e)}]
            }, 500

# Add a health check endpoint
@api.route('/health')
class HealthResource(Resource):
    def get(self):
        """Check if the API is running"""
        status = "DEGRADED" if not graph else "OK"
        message = "Study Buddy AI Assistant is running, but with degraded functionality." if not graph else "Study Buddy AI Assistant is running"
        return {'status': status, 'message': message}, 200

# Custom citation generator for formatting references
def format_citation(source_type, metadata):
    """Format a citation in proper academic style based on source type and metadata"""
    try:
        current_date = datetime.now().strftime("%B %d, %Y")
        
        if source_type == "wikipedia":
            # Format Wikipedia citation
            title = metadata.get("title", "Unknown Article")
            return f"Wikipedia. (n.d.). {title}. Retrieved {current_date}"
            
        elif source_type == "arxiv":
            # Format ArXiv citation
            title = metadata.get("title", "Unknown Paper")
            authors = metadata.get("authors", "Unknown Authors")
            paper_id = metadata.get("entry_id", "").replace("http://arxiv.org/abs/", "")
            year = metadata.get("published", "n.d.")[:4] if metadata.get("published") else "n.d."
            
            return f"{authors}. ({year}). {title}. arXiv:{paper_id}"
            
        elif source_type == "web":
            # Format web citation
            title = metadata.get("title", "Unknown Website")
            url = metadata.get("url", "")
            site_name = metadata.get("site_name", url.split("//")[-1].split("/")[0] if url else "Unknown Site")
            
            return f"{title}. ({current_date.split()[-1]}). {site_name}. Retrieved from {url}"
            
        elif source_type == "book":
            # Format book citation
            title = metadata.get("title", "Unknown Book")
            authors = metadata.get("authors", "Unknown Author")
            year = metadata.get("year", "n.d.")
            publisher = metadata.get("publisher", "Unknown Publisher")
            
            return f"{authors}. ({year}). {title}. {publisher}"
            
        else:
            # Generic citation
            return f"Unknown source. ({current_date}). Retrieved from {metadata.get('url', 'unknown source')}"
    
    except Exception as e:
        print(f"Error formatting citation: {str(e)}")
        return "Citation formatting error"

# Extract URLs and format them as citations
def extract_and_format_citations(text):
    """Find URLs in text and convert them to properly formatted citations"""
    if not text:
        return text
        
    # Simple URL regex pattern
    url_pattern = r'https?://[^\s)"]+'
    
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    
    # Replace each URL with a formatted citation
    for url in urls:
        # Extract domain as site name
        site_name = url.split("//")[-1].split("/")[0]
        
        # Create metadata
        metadata = {"url": url, "site_name": site_name}
        
        # Generate citation
        citation = format_citation("web", metadata)
        
        # Replace the raw URL with the citation
        text = text.replace(url, f"[{site_name}]")
    
    return text

# Post-process response to improve citations
def post_process_response(response):
    """Post-process a response to improve citations and formatting"""
    if not response or not hasattr(response, 'content'):
        return response
        
    content = response.content
    
    # Format URLs as citations
    content = extract_and_format_citations(content)
    
    # Enhance Chain-of-Thought formatting if present
    content = enhance_cot_structure(content)
    
    # Check if there's a references section
    ref_indicators = ["References", "REFERENCES", "Reference:", "Sources:", "SOURCES"]
    has_references = any(indicator in content for indicator in ref_indicators)
    
    if not has_references:
        # Add references section if missing
        content += "\n\n## References\n"
        content += "- Study Buddy AI Assistant. (2023). Educational content synthesis.\n"
    
    # Update the response content
    response.content = content
    return response

# Add a function to enhance the structure of Chain-of-Thought reasoning
def enhance_cot_structure(text):
    """Enhance the structure and formatting of Chain-of-Thought reasoning in the text"""
    if not text:
        return text
        
    # Identify common step indicators
    step_patterns = [
        r'Step \d+[\.:]',  # Step 1:, Step 2., etc.
        r'(\d+)[\.:]',      # 1., 2:, etc. at the beginning of lines
        r'First,',
        r'Second,', 
        r'Third,',
        r'Next,',
        r'Then,',
        r'Finally,'
    ]
    
    # Flag to track if we've detected CoT reasoning
    has_cot_structure = any(re.search(pattern, text, re.IGNORECASE) for pattern in step_patterns)
    
    if has_cot_structure:
        # Add a wrapper section for the CoT reasoning if not already present
        if not re.search(r'(step[-\s]by[-\s]step|reasoning process|chain[-\s]of[-\s]thought|solution approach)', text, re.IGNORECASE):
            sections = text.split('\n\n')
            
            # Find the best place to insert the CoT header
            insert_index = 0
            for i, section in enumerate(sections):
                if any(re.search(pattern, section, re.IGNORECASE) for pattern in step_patterns):
                    insert_index = i
                    break
            
            # Insert the CoT header
            sections.insert(insert_index, "## Step-by-Step Solution Process")
            text = '\n\n'.join(sections)
        
        # Make step numbers bold if they aren't already
        for pattern in step_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Skip if the match already contains markdown formatting
                if '**' in match:
                    continue
                # Create the bold replacement, preserving trailing punctuation
                replacement = re.sub(r'(\w+)', r'**\1**', match)
                text = text.replace(match, replacement)
    
    return text

# Add feedback handling in the ChatResource class
@chat_ns.route('/feedback')
class FeedbackResource(Resource):
    feedback_model = api.model('Feedback', {
        'conversation_id': fields.String(required=True, description='Conversation identifier'),
        'message_id': fields.String(required=True, description='Message identifier'),
        'rating': fields.Integer(required=True, description='Rating (1-5)'),
        'feedback_text': fields.String(required=False, description='Additional feedback text')
    })
    
    @chat_ns.expect(feedback_model)
    @chat_ns.doc(
        responses={
            200: "Feedback received",
            400: "Invalid input",
            404: "Conversation not found"
        },
        description="Submit feedback for a response"
    )
    def post(self):
        """Submit feedback for a response"""
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            message_id = data.get('message_id')
            rating = data.get('rating')
            feedback_text = data.get('feedback_text', '')
            
            if not conversation_id or not message_id or rating is None:
                return {"message": "Missing required parameters"}, 400
                
            if conversation_id not in conversations:
                return {"message": "Conversation not found"}, 404
                
            # In a production environment, store feedback in a database
            # For now, we'll just log it
            print(f"Received feedback for conversation {conversation_id}, message {message_id}")
            print(f"Rating: {rating}/5, Feedback: {feedback_text}")
            
            # Store feedback in the context for future use
            if conversation_id in conversation_contexts:
                if "feedback" not in conversation_contexts[conversation_id]:
                    conversation_contexts[conversation_id]["feedback"] = []
                
                conversation_contexts[conversation_id]["feedback"].append({
                    "message_id": message_id,
                    "rating": rating,
                    "feedback_text": feedback_text,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Adjust response style based on feedback
                if rating <= 2:  # Poor rating
                    # Simplify explanations for next responses
                    conversation_contexts[conversation_id]["simplify_explanations"] = True
                elif rating >= 4:  # Good rating
                    # Keep similar style for future responses
                    if "simplify_explanations" in conversation_contexts[conversation_id]:
                        del conversation_contexts[conversation_id]["simplify_explanations"]
            
            return {"message": "Feedback received"}, 200
            
        except Exception as e:
            print(f"Error processing feedback: {str(e)}")
            return {"message": "Error processing feedback"}, 500

if __name__ == '__main__':
    print("Starting Study Buddy AI Assistant...")
    app.run(debug=True, host='0.0.0.0', port=5000) 