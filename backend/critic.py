import re
from typing import Dict, List, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, FunctionMessage

class CriticFramework:
    """
    Implementation of the CRITIC framework for LLM self-correction through tool-interactive critiquing.
    
    Based on: "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing"
    (Gou et al., NeurIPS 2023)
    """
    
    def __init__(self, llm):
        """
        Initialize the CRITIC framework with the LLM to use for critiquing.
        
        Args:
            llm: LangChain-compatible LLM for generating reflections
        """
        self.llm = llm
    
    def reflect_on_tool_output(self, state):
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
            tool_outputs = [msg for msg in state["messages"] if isinstance(msg, FunctionMessage)]
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
            
            # Use the provided LLM
            reflection_result = self.llm.invoke(reflection_messages)
            
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
            specific_query = latest_user_message
            
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
                if query_match:
                    specific_query = query_match.group(1)
                else:
                    # Try alternative patterns for query extraction
                    alt_match = re.search(r"should search for [\"\'](.*?)[\"\']", reflection_text, re.IGNORECASE)
                    if alt_match:
                        specific_query = alt_match.group(1)
                
                # Add reasoning step for tool selection
                reasoning_steps.append({
                    "type": "tool_selection",
                    "content": f"Based on CRITIC analysis, using {next_tool} to verify information with query: '{specific_query}'"
                })
                
                # Create a verification message for follow-up
                verification_message = (
                    "user", 
                    f"[SYSTEM: VERIFICATION QUERY] {specific_query}"
                )
                
                # Append the verification message to continue the conversation
                new_messages = state.get("messages", []).copy()
                new_messages.append(verification_message)
                
                # Update state with reflection and action
                return {
                    "messages": new_messages,
                    "tool_reflection": reflection_text,
                    "need_additional_tools": True,
                    "next_tool": next_tool,
                    "context": state.get("context", {}),
                    "memory": state.get("memory", ""),
                    "memory_summary": state.get("memory_summary", ""),
                    "reasoning": reasoning_steps,
                    "tool_choice": next_tool
                }
            else:
                # No additional verification needed
                reasoning_steps.append({
                    "type": "conclusion",
                    "content": "Information appears sufficient and accurate; no additional verification needed"
                })
                
                # Update state without requesting additional tools
                return {
                    "messages": state.get("messages", []),
                    "tool_reflection": reflection_text,
                    "need_additional_tools": False,
                    "context": state.get("context", {}),
                    "memory": state.get("memory", ""),
                    "memory_summary": state.get("memory_summary", ""),
                    "reasoning": reasoning_steps,
                    "tool_choice": None
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

    def generate_self_correction(self, state, reflection_result):
        """
        Generate a corrected response based on the reflection and tool outputs.
        
        Args:
            state: The current state
            reflection_result: The result of the reflection analysis
            
        Returns:
            A corrected response
        """
        try:
            # Extract relevant information from the state
            user_messages = [msg for msg in state["messages"] if isinstance(msg, tuple) and msg[0] == "user"]
            if not user_messages:
                return "I don't have enough context to provide a correction."
                
            original_query = user_messages[0][1]
            tool_outputs = [msg for msg in state["messages"] if isinstance(msg, FunctionMessage)]
            reflection_text = reflection_result.get("tool_reflection", "")
            
            # Create a prompt for self-correction
            correction_prompt = f"""
            # Self-Correction Based on Tool Verification
            
            ## Original Query
            {original_query}
            
            ## Tool Outputs
            {[t.content if hasattr(t, 'content') else str(t) for t in tool_outputs]}
            
            ## Reflection Analysis
            {reflection_text}
            
            Based on the analysis above, please generate a comprehensive, accurate response that:
            1. Incorporates verified information from the tools
            2. Explains any corrections or clarifications made
            3. Acknowledges any remaining uncertainties
            4. Provides proper citations for all factual claims
            
            Your response should be educational in tone, well-structured, and focused on helping the student.
            """
            
            # Call LLM to generate the corrected response
            correction_messages = [
                SystemMessage(content="You are a precise educational assistant committed to accuracy."),
                HumanMessage(content=correction_prompt)
            ]
            
            # Get the corrected response
            corrected_response = self.llm.invoke(correction_messages)
            
            return corrected_response.content if hasattr(corrected_response, 'content') else str(corrected_response)
            
        except Exception as e:
            print(f"Error generating self-correction: {str(e)}")
            return f"I apologize, but I encountered an error while trying to verify the information: {str(e)}" 