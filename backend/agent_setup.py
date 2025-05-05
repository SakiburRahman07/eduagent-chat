import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun

# Import Tavily Search (conditionally)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from tavily import TavilyClient
    tavily_available = True
except ImportError:
    print("Warning: Tavily packages not found. Web search functionality will be disabled.")
    tavily_available = False

# Load environment variables
load_dotenv()

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

# Initialize LLM
try:
    # Initialize Groq model
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    
    # Bind tools
    llm_with_tools = llm.bind_tools(tools=tools)
    
    print("LLM initialized successfully with Groq's Llama-4-Maverick-17B-Instruct model.")
except Exception as e:
    print(f"ERROR: Failed to initialize the LLM: {str(e)}")
    print("The application will not function correctly. Please check your GROQ_API_KEY.")
    llm = None
    llm_with_tools = None 