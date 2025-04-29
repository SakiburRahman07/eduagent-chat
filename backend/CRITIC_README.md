# CRITIC Framework Implementation in Study Buddy

## Overview

This implementation adds the CRITIC (Large Language Models Can Self-Correct with Tool-Interactive Critiquing) framework to the Study Buddy educational assistant. Based on the research paper by Gou et al. (NeurIPS 2023), this framework enables the LLM to critique its own outputs by using tools to verify information and self-correct.

## Key Components

1. **critic.py**: Contains the `CriticFramework` class that implements the core CRITIC functionality
   - `reflect_on_tool_output`: Analyzes tool outputs and determines if additional verification is needed
   - `generate_self_correction`: Creates improved responses based on verified information

2. **LangGraph Integration**: The workflow has been modified to:
   - Send tool outputs through the CRITIC node for reflection
   - Conditionally route to additional tools for verification based on CRITIC analysis
   - Return to the chatbot for final response generation once verification is complete

## How CRITIC Works

1. **Initial Tool Usage**: When a tool (Wikipedia, ArXiv, etc.) is used to answer a query, its output is sent to the CRITIC framework

2. **Self-Critiquing**: The system analyzes the tool output by:
   - Identifying factual claims that need verification
   - Assessing completeness of the information
   - Detecting potential inaccuracies or contradictions
   - Evaluating if the output adequately answers the query

3. **Tool Selection**: If verification is needed, CRITIC selects an appropriate tool and generates a specific verification query

4. **Self-Correction**: After gathering all necessary information, the system integrates verified facts and corrects inaccuracies

5. **Transparency**: The final response includes an explanation that the CRITIC framework was used for verification

## Benefits

- **Improved Accuracy**: ~22.5% relative improvement on factual query accuracy (per original paper)
- **Reduced Hallucinations**: Significantly fewer unverified claims in responses
- **Educational Value**: Shows students the importance of verifying information from multiple sources
- **Transparent Reasoning**: Helps students understand the process of scientific/academic verification

## Example Process Flow

```
User Query: "When was the Eiffel Tower completed?"

Initial Tool: Wikipedia search for "Eiffel Tower"

Tool Output: "The Eiffel Tower was built for the 1889 World's Fair..."

CRITIC Analysis: 
- Claim: Eiffel Tower was built for 1889 World's Fair
- Verification: Is this the completion date?
- Assessment: Tool output mentions the fair date but not explicit completion date

Verification Query: "What exact date was the Eiffel Tower completed?"

Secondary Tool: Tavily search for specific completion date

Tool Output: "The Eiffel Tower was completed on March 31, 1889..."

Final Response: "The Eiffel Tower was completed on March 31, 1889, in time for the 1889 World's Fair in Paris..."
```

## Reference

```
@inproceedings{gou2023critic,
  title={CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing},
  author={Gou, Zhibin and Shao, Zhihong and Gong, Yeyun and Shen, Yelong and Yang, Yujiu and Duan, Nan and Chen, Weizhu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
``` 