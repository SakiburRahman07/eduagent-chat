import os
import sys
import re
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
from dotenv import load_dotenv
import json
from datetime import datetime

# Import the agent functionality
from agent import HierarchicalMemory, State, create_graph, summarize_conversation, apply_self_consistency
from agent_setup import tools, llm, llm_with_tools

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

# Define memory serialization directory
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_store")
# Create directory if it doesn't exist
os.makedirs(MEMORY_DIR, exist_ok=True)

# Initialize conversation tracking
conversations = {}
conversation_contexts = {}
conversation_memories = {}

# Initialize hierarchical memory system
hierarchical_memories = {}

# Create graph from imported components
graph = create_graph(tools, llm)
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
            is_complex_math = any(re.search(pattern, user_input, re.IGNORECASE) for pattern in math_patterns)
            
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