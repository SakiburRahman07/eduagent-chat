import os
import sys
import re
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from datetime import datetime
import json
from collections import Counter
import numpy as np

# LangGraph and LangChain imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, FunctionMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

# Import our custom CRITIC framework
from critic import CriticFramework

# Try to import sentence transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
    # Initialize the embedding model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence transformers initialized successfully for semantic memory search.")
    except Exception as e:
        print(f"Error initializing sentence transformer model: {str(e)}")
        embedding_model = None
        sentence_transformers_available = False
except ImportError:
    print("Sentence transformers not available. Semantic search will be disabled.")
    embedding_model = None
    sentence_transformers_available = False

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

# Hierarchical Memory System inspired by MemGPT
class MemoryPage:
    """A page of memory containing conversation exchanges and metadata"""
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        
    def access(self):
        """Mark this page as accessed, updating metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        return self

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary (deserialization)"""
        page = cls(data["content"], data["metadata"])
        page.created_at = datetime.fromisoformat(data["created_at"])
        page.last_accessed = datetime.fromisoformat(data["last_accessed"])
        page.access_count = data["access_count"]
        return page

class HierarchicalMemory:
    """MemGPT-inspired hierarchical memory system with main memory and external storage"""
    def __init__(self, config=None):
        # Configuration
        self.config = {
            "main_memory_capacity": 10,  # Number of exchanges to keep in immediate context
            "attention_sink_size": 2,     # Number of important memories to always include
            "recency_weight": 0.6,        # Weight for recency in scoring (vs relevance)
            "relevance_threshold": 0.3,   # Minimum relevance score to retrieve
        }
        if config:
            self.config.update(config)
            
        # Memory structures
        self.main_memory = []               # Short-term/working memory (token context window)
        self.external_memory = {}           # Long-term storage by topic
        self.attention_sinks = []           # Critical memories that should always be accessible
        self.user_profile = {}              # Persistent information about the user
        self.embeddings_cache = {}          # Cache for computed embeddings
        
        # Statistics
        self.stats = {
            "total_exchanges": 0,
            "pages": 0,
            "retrievals": 0,
            "page_ins": 0,
            "page_outs": 0
        }
    
    def add_exchange(self, user_message: str, ai_message: str, user_metadata: Dict = None, ai_metadata: Dict = None):
        """Add a new conversation exchange to memory"""
        # Create memory pages for user and AI messages
        user_page = MemoryPage(
            content=user_message,
            metadata={
                "type": "user_message",
                "timestamp": datetime.now().isoformat(),
                **(user_metadata or {})
            }
        )
        
        ai_page = MemoryPage(
            content=ai_message,
            metadata={
                "type": "ai_message",
                "timestamp": datetime.now().isoformat(),
                **(ai_metadata or {})
            }
        )
        
        # Add to main memory
        self.main_memory.append((user_page, ai_page))
        
        # Check if we need to page out to external memory
        if len(self.main_memory) > self.config["main_memory_capacity"]:
            self._page_out()
            
        # Update memory statistics
        self.stats["total_exchanges"] += 1
        
        # Extract and update user profile information
        self._update_user_profile(user_message, user_metadata)
        
        return len(self.main_memory)
    
    def _page_out(self):
        """Move oldest memory from main memory to external memory"""
        # Remove oldest exchange from main memory (except attention sinks)
        oldest_exchange = self.main_memory.pop(0)
        user_page, ai_page = oldest_exchange
        
        # Extract topics for memory organization
        topics = self._extract_topics(user_page.content)
        
        # Store in external memory under each topic
        for topic in topics:
            if topic not in self.external_memory:
                self.external_memory[topic] = []
            self.external_memory[topic].append(oldest_exchange)
        
        # Update statistics
        self.stats["pages"] += 1
        self.stats["page_outs"] += 1
        
        # Check if this exchange should be an attention sink
        self._check_attention_sink_candidate(oldest_exchange)
    
    def _check_attention_sink_candidate(self, exchange):
        """Evaluate if an exchange should become an attention sink"""
        user_page, ai_page = exchange
        
        # Criteria for attention sinks:
        # 1. Message contains personal information about student
        # 2. Message defines learning goals or preferences
        # 3. Message contains important context for the tutoring relationship
        
        important_keywords = [
            "my name is", "i am", "i'm", "my goal", "my learning", 
            "i want to", "i need to", "i prefer", "my background",
            "my major", "my field", "remember this", "important"
        ]
        
        is_important = any(keyword in user_page.content.lower() for keyword in important_keywords)
        
        if is_important:
            # Only keep top N attention sinks
            self.attention_sinks.append(exchange)
            if len(self.attention_sinks) > self.config["attention_sink_size"]:
                self.attention_sinks.pop(0)  # Remove oldest attention sink
    
    def _update_user_profile(self, message, metadata=None):
        """Extract and update information about the user"""
        # This would be more sophisticated in production
        # Simple extraction of education level
        education_patterns = [
            (r"i'?m in (elementary|middle|high) school", "education_level"),
            (r"i'?m a (freshman|sophomore|junior|senior|college|university|graduate|phd) student", "education_level"),
            (r"i'?m studying ([a-zA-Z\s]+) at ([a-zA-Z\s]+)", "field_of_study"),
            (r"i want to learn about ([a-zA-Z\s]+)", "learning_interests"),
            (r"i'?m interested in ([a-zA-Z\s]+)", "interests"),
            (r"my name is ([a-zA-Z\s]+)", "name"),
            (r"call me ([a-zA-Z\s]+)", "name")
        ]
        
        # Extract information using patterns
        message_lower = message.lower()
        for pattern, key in education_patterns:
            match = re.search(pattern, message_lower)
            if match:
                # For capturing specific fields identified in the patterns
                if key in ["field_of_study", "learning_interests", "interests"]:
                    if key not in self.user_profile:
                        self.user_profile[key] = []
                    # Add to list, avoiding duplicates
                    value = match.group(1).strip()
                    if value not in self.user_profile[key]:
                        self.user_profile[key].append(value)
                else:
                    # For single-value fields
                    self.user_profile[key] = match.group(1).strip()
        
        # Update from metadata if provided
        if metadata:
            for key, value in metadata.items():
                if key.startswith("user_"):  # Only store user-related metadata
                    profile_key = key[5:]  # Remove 'user_' prefix
                    self.user_profile[profile_key] = value
    
    def _extract_topics(self, message):
        """Extract topic keywords from a message for memory organization"""
        # Core academic subjects
        academic_subjects = [
            "math", "mathematics", "algebra", "calculus", "geometry", "statistics",
            "physics", "chemistry", "biology", "anatomy", "ecology", "genetics",
            "history", "geography", "civics", "political science", "economics",
            "literature", "writing", "grammar", "language", "linguistics",
            "computer science", "programming", "coding", "algorithms",
            "psychology", "sociology", "anthropology", "philosophy",
            "art", "music", "theater", "film", "design"
        ]
        
        # Find matches in the message
        message_lower = message.lower()
        found_topics = []
        
        # Match core academic subjects
        for subject in academic_subjects:
            if subject in message_lower:
                found_topics.append(subject)
        
        # If no specific topics found, use general categories
        if not found_topics:
            # Try to categorize into general areas
            if any(term in message_lower for term in ["math", "equation", "number", "calculation"]):
                found_topics.append("mathematics")
            elif any(term in message_lower for term in ["science", "experiment", "theory", "natural"]):
                found_topics.append("science")
            elif any(term in message_lower for term in ["history", "past", "century", "ancient", "war", "civilization"]):
                found_topics.append("history")
            elif any(term in message_lower for term in ["book", "novel", "story", "author", "write", "essay"]):
                found_topics.append("literature")
            else:
                found_topics.append("general")
        
        return found_topics
    
    def retrieve_relevant_context(self, query, limit=3):
        """Retrieve relevant context from memory using semantic search"""
        # Start with attention sinks (always included)
        relevant_exchanges = []
        
        # Add attention sinks
        attention_content = []
        for exchange in self.attention_sinks:
            user_page, ai_page = exchange
            attention_content.append(f"Attention Sink - Student: {user_page.content}")
            attention_content.append(f"Attention Sink - Response: {ai_page.content}")
        
        # Compute query embedding for semantic search
        query_embedding = self._get_embedding(query)
        
        # Search in main memory first
        from_main = self._search_memory_segment(query, query_embedding, self.main_memory)
        
        # Then search in external memory
        from_external = []
        query_topics = self._extract_topics(query)
        
        # Collect all potentially relevant exchanges from external memory
        candidate_exchanges = []
        for topic in query_topics:
            if topic in self.external_memory:
                candidate_exchanges.extend(self.external_memory[topic])
        
        if candidate_exchanges:
            from_external = self._search_memory_segment(query, query_embedding, candidate_exchanges)
        
        # Combine results - first attention sinks, then main memory, then external
        combined = []
        
        # Add attention sink content
        if attention_content:
            combined.append("\n## Important Context\n" + "\n".join(attention_content))
        
        # Format and add main memory exchanges
        if from_main:
            main_content = [f"## Recent Conversation\n"]
            for score, exchange in from_main[:limit]:
                user_page, ai_page = exchange
                main_content.append(f"Student: {user_page.content}")
                main_content.append(f"Study Buddy: {ai_page.content[:200]}...")
            combined.append("\n".join(main_content))
        
        # Format and add external memory exchanges
        if from_external:
            external_content = [f"## Related Previous Exchanges\n"]
            for score, exchange in from_external[:limit]:
                user_page, ai_page = exchange
                external_content.append(f"Student previously asked: {user_page.content}")
                external_content.append(f"You answered: {ai_page.content[:200]}...")
            combined.append("\n".join(external_content))
        
        # Update statistics
        self.stats["retrievals"] += 1
        
        return "\n\n".join(combined)
    
    def _search_memory_segment(self, query, query_embedding, memory_segment):
        """Search within a specific memory segment using semantic similarity"""
        if not memory_segment:
            return []
            
        scored_exchanges = []
        
        # If embeddings are available, use semantic search
        if query_embedding is not None and embedding_model is not None:
            for exchange in memory_segment:
                user_page, _ = exchange
                
                # Get or compute embedding for this exchange
                page_embedding = self._get_embedding(user_page.content)
                
                if page_embedding is not None:
                    # Compute semantic similarity
                    similarity = np.dot(query_embedding, page_embedding)
                    
                    # Skip if below relevance threshold
                    if similarity < self.config["relevance_threshold"]:
                        continue
                    
                    # Calculate recency score (normalized by time decay)
                    time_diff = (datetime.now() - user_page.last_accessed).total_seconds() / 3600  # hours
                    recency_score = 1.0 / (1.0 + time_diff/24)  # decay over days
                    
                    # Combined score (weighted sum of similarity and recency)
                    combined_score = (1-self.config["recency_weight"]) * similarity + self.config["recency_weight"] * recency_score
                    
                    scored_exchanges.append((combined_score, exchange))
        else:
            # Fallback to keyword matching if embeddings unavailable
            keywords = self._extract_keywords(query)
            
            for exchange in memory_segment:
                user_page, _ = exchange
                
                # Count keyword matches
                match_count = sum(1 for kw in keywords if kw in user_page.content.lower())
                if match_count == 0:
                    continue
                    
                # Normalize by total keywords
                similarity = match_count / len(keywords) if keywords else 0
                
                # Calculate recency score
                time_diff = (datetime.now() - user_page.last_accessed).total_seconds() / 3600
                recency_score = 1.0 / (1.0 + time_diff/24)
                
                # Combined score
                combined_score = (1-self.config["recency_weight"]) * similarity + self.config["recency_weight"] * recency_score
                
                scored_exchanges.append((combined_score, exchange))
        
        # Sort by score (descending)
        scored_exchanges.sort(reverse=True, key=lambda x: x[0])
        
        # Mark accessed pages
        for _, exchange in scored_exchanges:
            user_page, ai_page = exchange
            user_page.access()
            ai_page.access()
        
        return scored_exchanges
    
    def _get_embedding(self, text):
        """Get embedding for text, using cache to avoid recomputation"""
        if not embedding_model or not sentence_transformers_available:
            return None
            
        # Simple cache key - in production would use a hashing function
        cache_key = text[:100]  # First 100 chars as key
        
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
            
        try:
            embedding = embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            self.embeddings_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return None
    
    def _extract_keywords(self, text):
        """Extract keywords from text for basic relevance matching"""
        # Remove common stopwords
        stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "in", "to", "for", "with", "on", "at"}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords
    
    def get_memory_summary(self):
        """Generate a summary of memory state and user profile"""
        summary = []
        
        # User profile summary
        if self.user_profile:
            summary.append("## Student Profile")
            for key, value in self.user_profile.items():
                if isinstance(value, list):
                    summary.append(f"- {key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    summary.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Memory statistics
        summary.append("## Memory Statistics")
        summary.append(f"- Current session exchanges: {len(self.main_memory)}")
        summary.append(f"- Total knowledge areas: {len(self.external_memory)}")
        summary.append(f"- Total stored exchanges: {self.stats['total_exchanges']}")
        
        # Topics with stored knowledge
        if self.external_memory:
            topics = list(self.external_memory.keys())
            topics_str = ", ".join(topics[:5])
            if len(topics) > 5:
                topics_str += f" and {len(topics) - 5} more"
            summary.append(f"- Knowledge areas: {topics_str}")
        
        return "\n".join(summary)
    
    def save_to_file(self, filepath):
        """Serialize memory to file"""
        memory_data = {
            "main_memory": [
                (user.to_dict(), ai.to_dict())
                for user, ai in self.main_memory
            ],
            "external_memory": {
                topic: [
                    (user.to_dict(), ai.to_dict())
                    for user, ai in exchanges
                ]
                for topic, exchanges in self.external_memory.items()
            },
            "attention_sinks": [
                (user.to_dict(), ai.to_dict())
                for user, ai in self.attention_sinks
            ],
            "user_profile": self.user_profile,
            "stats": self.stats,
            "config": self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(memory_data, f)
    
    @classmethod
    def load_from_file(cls, filepath):
        """Load memory from serialized file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        memory = cls(config=data.get("config"))
        
        # Restore main memory
        memory.main_memory = [
            (MemoryPage.from_dict(user), MemoryPage.from_dict(ai))
            for user, ai in data.get("main_memory", [])
        ]
        
        # Restore external memory
        memory.external_memory = {
            topic: [
                (MemoryPage.from_dict(user), MemoryPage.from_dict(ai))
                for user, ai in exchanges
            ]
            for topic, exchanges in data.get("external_memory", {}).items()
        }
        
        # Restore attention sinks
        memory.attention_sinks = [
            (MemoryPage.from_dict(user), MemoryPage.from_dict(ai))
            for user, ai in data.get("attention_sinks", [])
        ]
        
        # Restore user profile and stats
        memory.user_profile = data.get("user_profile", {})
        memory.stats = data.get("stats", {})
        
        return memory

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

# Initialize hierarchical memory system
hierarchical_memories = {}

# Define memory serialization directory
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
# Create directory if it doesn't exist
os.makedirs(MEMORY_DIR, exist_ok=True)

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
- Apply the **CRITIC framework** to verify information and self-correct when necessary.

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

🔬 THE CRITIC FRAMEWORK:
When providing factual information, I follow the CRITIC framework to ensure accuracy:
1. **Claim Identification**: I identify specific claims that need verification
2. **Research & Information Gathering**: I use appropriate tools to verify each claim
3. **Information Triangulation**: I cross-check information from multiple sources when possible
4. **Truth Assessment**: I evaluate the reliability and consistency of the information
5. **Integration & Correction**: I integrate verified information and correct any inaccuracies
6. **Confidence Communication**: I clearly state my confidence level in the information provided

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
        # Get context and memory information
        context = state.get("context", {})
        memory_content = state.get("memory", "")
        memory_summary = state.get("memory_summary", "")
        
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
                    
                    # Check if we have relevant memory content
                    if memory_content:
                        reasoning_steps.append({
                            "type": "memory",
                            "content": "Retrieving relevant context from hierarchical memory system"
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
        
        # Add memory content if available - this includes retrieved context from hierarchical memory
        if memory_content:
            dynamic_system_prompt += f"\n\n{memory_content}\n"
            
        # Add memory summary if available - this includes statistics from the memory system
        if memory_summary:
            dynamic_system_prompt += f"\n\nMEMORY SUMMARY:\n{memory_summary}\n"
        
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
            if "self_consistency_applied" in context and context["self_consistency_applied"]:
                context_info += f"- NOTE: Using self-consistency verification for enhanced accuracy.\n"
                reasoning_steps.append({
                    "type": "thought",
                    "content": "Applying self-consistency checks for greater reliability"
                })
            
            dynamic_system_prompt += context_info
        
        # Add specific instructions for memory utilization
        memory_instructions = """
MEMORY UTILIZATION INSTRUCTIONS:
1. The MEMORY SUMMARY section contains information about the student that has been automatically extracted.
2. The "Important Context" and "Related Previous Exchanges" sections contain relevant information from past conversations.
3. Use this information to personalize your response and maintain continuity in the tutoring relationship.
4. If the student refers to something from a previous conversation, try to connect it with the information provided.
5. If you need additional information that isn't in memory, you can use your search tools.
"""
        dynamic_system_prompt += memory_instructions
        
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
            "memory_summary": state.get("memory_summary", ""),
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
            "memory_summary": state.get("memory_summary", ""),
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
                "memory_summary": state.get("memory_summary", ""),
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
            "memory_summary": state.get("memory_summary", ""),
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
            "memory_summary": state.get("memory_summary", ""),
            "reasoning": [{"type": "error", "content": str(e)}]
        }

# Add a function to implement the CRITIC framework for tool output reflection
def reflect_on_tool_output(state: State):
    """
    Implement the CRITIC framework for LLM to critique and verify information using tools.
    
    This function allows the agent to:
    1. Identify potential errors or inaccuracies in tool outputs
    2. Verify information against other sources
    3. Determine if additional tools are needed for complete verification
    4. Produce more accurate and transparent responses
    
    Args:
        state: The current state including messages and tool outputs
        
    Returns:
        Updated state with reflection results and next tool selection if needed
    """
    try:
        # Initialize reasoning steps to track the reflection process
        reasoning_steps = state.get("reasoning", [])
        
        # Extract the most recent user query
        user_messages = [msg for msg in state["messages"] if isinstance(msg, tuple) and msg[0] == "user"]
        if not user_messages:
            reasoning_steps.append({
                "type": "critic",
                "content": "No user query found to reflect on"
            })
            return {
                "messages": state.get("messages", []),
                "tool_reflection": None,
                "need_additional_tools": False,
                "context": state.get("context", {}),
                "memory": state.get("memory", ""),
                "memory_summary": state.get("memory_summary", ""),
                "reasoning": reasoning_steps
            }
            
        latest_user_message = user_messages[-1][1]
        
        # Get the most recent tool outputs
        tool_outputs = [msg for msg in state["messages"] if hasattr(msg, 'type') and msg.type == "function"]
        if not tool_outputs:
            reasoning_steps.append({
                "type": "critic",
                "content": "No tool outputs to reflect on"
            })
            return {
                "messages": state.get("messages", []),
                "tool_reflection": None,
                "need_additional_tools": False,
                "context": state.get("context", {}),
                "memory": state.get("memory", ""),
                "memory_summary": state.get("memory_summary", ""),
                "reasoning": reasoning_steps
            }
            
        latest_tool_output = tool_outputs[-1].content if hasattr(tool_outputs[-1], 'content') else str(tool_outputs[-1])
        current_tool = state.get("tool_choice", "unknown_tool")
        
        # Add a reasoning step about starting reflection
        reasoning_steps.append({
            "type": "critic",
            "content": f"Reflecting on output from {current_tool} to verify information accuracy"
        })
        
        # Generate reflection prompt following CRITIC framework
        reflection_prompt = f"""
        # CRITIC Framework Analysis
        
        ## User Query
        {latest_user_message}
        
        ## Tool Used
        {current_tool}
        
        ## Tool Output
        {latest_tool_output}
        
        Please analyze this tool output following the CRITIC framework:
        
        1. **Identify Claims**: What factual claims are made in this output?
        2. **Assess Verification Needs**: Which claims require further verification?
        3. **Identify Potential Errors**: Are there any potential inaccuracies, contradictions, or uncertainties?
        4. **Evaluate Completeness**: Does this output fully answer the user's question?
        5. **Tool Selection**: Would another tool provide complementary or corrective information?
        
        Based on your analysis:
        - Is the information adequate and accurate? (Yes/No/Partial)
        - Should we use another tool to verify or complement? (Yes/No)
        - If yes, what specific tool should we use? (wikipedia/arxiv/tavily)
        - What specific query should we use with that tool?
        """
        
        # Call LLM to generate reflection
        reflection_messages = [
            SystemMessage(content="You are a critical thinking assistant evaluating information quality and accuracy."), 
            HumanMessage(content=reflection_prompt)
        ]
        
        # Use the same LLM that's used throughout the application
        reflection_result = llm.invoke(reflection_messages)
        
        # Extract reflection content
        reflection_text = reflection_result.content if hasattr(reflection_result, 'content') else str(reflection_result)
        
        # Add detailed reflection to reasoning steps
        reasoning_steps.append({
            "type": "critic_reflection",
            "content": reflection_text[:300] + "..." if len(reflection_text) > 300 else reflection_text
        })
        
        # Parse reflection to determine next action
        need_more_info = "yes" in reflection_text.lower() and (
            "should we use another tool" in reflection_text.lower() or 
            "use another tool" in reflection_text.lower()
        )
        
        # Extract recommended tool if mentioned
        next_tool = None
        if need_more_info:
            # Look for tool recommendations in the reflection
            for tool_name in ["wikipedia", "arxiv", "tavily"]:
                if tool_name.lower() in reflection_text.lower():
                    next_tool = tool_name
                    break
                    
            # If no specific tool was recommended but verification is needed,
            # select a tool different from the current one
            if not next_tool:
                available_tools = [t for t in ["wikipedia", "arxiv", "tavily"] if t != current_tool]
                if available_tools:
                    next_tool = available_tools[0]
            
            # Try to extract a specific query recommendation
            query_match = re.search(r"query.*?[\"\'](.*?)[\"\']", reflection_text, re.IGNORECASE)
            specific_query = query_match.group(1) if query_match else latest_user_message
            
            # Add reasoning step for tool selection
            reasoning_steps.append({
                "type": "tool_selection",
                "content": f"Based on CRITIC analysis, using {next_tool} to verify information with query: '{specific_query}'"
            })
            
            # Create a human message that will be processed by the chatbot to use the tool
            verification_message = (
                "user", 
                f"[SYSTEM: VERIFICATION QUERY] {specific_query}"
            )
            state["messages"].append(verification_message)
        else:
            reasoning_steps.append({
                "type": "conclusion",
                "content": "Information appears sufficient and accurate; no additional verification needed"
            })
        
        # Update state with reflection and action
        return {
            "messages": state.get("messages", []),
            "tool_reflection": reflection_text,
            "need_additional_tools": need_more_info,
            "next_tool": next_tool,
            "context": state.get("context", {}),
            "memory": state.get("memory", ""),
            "memory_summary": state.get("memory_summary", ""),
            "reasoning": reasoning_steps,
            "tool_choice": next_tool if need_more_info else None
        }
        
    except Exception as e:
        print(f"Error in CRITIC reflection: {str(e)}")
        return {
            "messages": state.get("messages", []),
            "tool_reflection": f"Error during reflection: {str(e)}",
            "need_additional_tools": False,
            "context": state.get("context", {}),
            "memory": state.get("memory", ""),
            "memory_summary": state.get("memory_summary", ""),
            "reasoning": state.get("reasoning", []) + [{"type": "error", "content": f"Error in CRITIC reflection: {str(e)}"}]
        }

# Create and compile the graph
def create_graph():
    try:
        # Initialize the CRITIC framework with our LLM
        critic_framework = CriticFramework(llm)
        
        # Define a wrapper function to integrate with LangGraph
        def critic_node(state):
            return critic_framework.reflect_on_tool_output(state)
        
        # Build the graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tool_router", tool_router)
        graph_builder.add_node("critic", critic_node)  # Add CRITIC reflection node
        
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
        
        # Add CRITIC framework to the workflow
        graph_builder.add_edge("tools", "critic")
        
        # Conditional edge from CRITIC reflection
        graph_builder.add_conditional_edges(
            "critic",
            lambda x: x.get("need_additional_tools", False),
            {
                True: "tool_router",  # If more info needed, route to another tool
                False: "chatbot"      # Otherwise proceed to response
            }
        )

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
            
            # Get or initialize hierarchical memory
            memory_file = os.path.join(MEMORY_DIR, f"{conversation_id}.json")
            
            if conversation_id not in hierarchical_memories:
                # Try to load from file first
                if os.path.exists(memory_file):
                    try:
                        hierarchical_memories[conversation_id] = HierarchicalMemory.load_from_file(memory_file)
                        print(f"Loaded memory from {memory_file}")
                    except Exception as e:
                        print(f"Error loading memory from {memory_file}: {str(e)}")
                        hierarchical_memories[conversation_id] = HierarchicalMemory()
                else:
                    hierarchical_memories[conversation_id] = HierarchicalMemory()
            
            # Get memory instance
            memory = hierarchical_memories[conversation_id]
            
            # Get or initialize conversation history and context
            if conversation_id not in conversations:
                conversations[conversation_id] = []
                conversation_contexts[conversation_id] = {}
            
            # Update context with any new information
            if context_updates:
                conversation_contexts[conversation_id].update(context_updates)
                
                # Also update user profile in memory with relevant context
                user_metadata = {}
                for key, value in context_updates.items():
                    if key in ["academic_level", "interests", "preferred_style", "learning_goal"]:
                        user_metadata[f"user_{key}"] = value
                
                if user_metadata and memory.main_memory:
                    # Update metadata for the most recent user message
                    recent_exchange = memory.main_memory[-1]
                    user_page, _ = recent_exchange
                    user_page.metadata.update(user_metadata)
            
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
            
            # Get relevant context from hierarchical memory
            relevant_context = memory.retrieve_relevant_context(user_input)
            memory_summary = memory.get_memory_summary()
            
            # Add user message to history
            conversations[conversation_id].append(("user", user_input))
            
            # Prepare state with messages, context and memory
            state = {
                "messages": conversations[conversation_id],
                "context": conversation_contexts[conversation_id],
                "memory": relevant_context,       # Retrieved relevant context
                "memory_summary": memory_summary, # Memory system summary
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
            critic_verification_happened = False
            
            for event in events:
                # Add each message to the history (except internal verification queries)
                if "messages" in event and event["messages"]:
                    message = event["messages"][-1]
                    
                    # Check if this is an AI message to add to responses
                    if hasattr(message, 'type') and message.type == "ai":
                        if hasattr(message, 'content') and message.content:
                            ai_responses.append(message.content)
                        
                        # Add AI message to conversation history only if it's not a verification response
                        if not critic_verification_happened:
                            conversations[conversation_id].append(("ai", message.content if hasattr(message, 'content') else ""))
                    
                    # Skip adding SYSTEM verification queries to the conversation history
                    if isinstance(message, tuple) and message[0] == "user" and "[SYSTEM: VERIFICATION QUERY]" in message[1]:
                        critic_verification_happened = True
                        # Add reasoning step about using CRITIC for verification
                        reasoning_steps.append({
                            "type": "critic",
                            "content": f"Applying CRITIC framework to verify information using additional tool queries"
                        })
                
                # Collect reasoning steps
                if "reasoning" in event:
                    for step in event["reasoning"]:
                        if "critic" in step.get("type", "").lower():
                            # Mark that we've used the CRITIC framework
                            critic_verification_happened = True
                        reasoning_steps.append(step)
            
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
            
            # Add explanation about CRITIC verification if applied
            if critic_verification_happened:
                # Add a brief explanation about the CRITIC framework being used for greater accuracy
                critic_explanation = (
                    "\n\n**Verification Process**: I used the CRITIC (Critique and Reflection for Information Validation) framework "
                    "to verify the information provided and ensure accuracy."
                )
                response += critic_explanation
            
            print(f"Sending response: '{response[:50]}...'")
            
            # Add the exchange to hierarchical memory
            memory.add_exchange(
                user_message=user_input,
                ai_message=response,
                user_metadata=context
            )
            
            # Save memory to file periodically (every 3 exchanges)
            if memory.stats["total_exchanges"] % 3 == 0:
                try:
                    memory.save_to_file(memory_file)
                    print(f"Saved memory to {memory_file}")
                except Exception as e:
                    print(f"Error saving memory to {memory_file}: {str(e)}")
            
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

# Add a memory statistics endpoint
memory_stats_model = api.model('MemoryStats', {
    'conversation_id': fields.String(required=True, description='Conversation identifier')
})

memory_stats_response = api.model('MemoryStatsResponse', {
    'stats': fields.Raw(description='Memory statistics'),
    'user_profile': fields.Raw(description='User profile information'),
    'topics': fields.List(fields.String(description='Knowledge topics')),
    'main_memory_size': fields.Integer(description='Number of exchanges in main memory'),
    'external_memory_size': fields.Integer(description='Number of exchanges in external memory'),
    'attention_sinks': fields.Integer(description='Number of attention sink memories')
})

@chat_ns.route('/memory_stats')
class MemoryStatsResource(Resource):
    @chat_ns.expect(memory_stats_model)
    @chat_ns.marshal_with(memory_stats_response)
    @chat_ns.doc(
        responses={
            200: "Success",
            404: "Conversation not found",
            500: "Server error"
        },
        description="Get memory statistics for a conversation"
    )
    def post(self):
        """Get memory statistics for a conversation"""
        try:
            data = request.json
            conversation_id = data.get('conversation_id')
            
            if not conversation_id:
                return {"message": "Missing conversation_id parameter"}, 400
                
            if conversation_id not in hierarchical_memories:
                return {"message": "Memory not found for conversation"}, 404
                
            memory = hierarchical_memories[conversation_id]
            
            # Count total exchanges in external memory
            external_count = sum(len(exchanges) for exchanges in memory.external_memory.values())
            
            return {
                'stats': memory.stats,
                'user_profile': memory.user_profile,
                'topics': list(memory.external_memory.keys()),
                'main_memory_size': len(memory.main_memory),
                'external_memory_size': external_count,
                'attention_sinks': len(memory.attention_sinks)
            }
            
        except Exception as e:
            print(f"Error retrieving memory stats: {str(e)}")
            return {"message": "Error retrieving memory statistics"}, 500

if __name__ == '__main__':
    print("Starting Study Buddy AI Assistant...")
    app.run(debug=True, host='0.0.0.0', port=5000) 