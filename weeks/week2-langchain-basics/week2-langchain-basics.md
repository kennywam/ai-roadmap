# Week 2: LangChain Basics

## Learning Objectives
- Understand the LangChain framework and its components
- Learn to build chains and agents using LangChain
- Implement memory systems for conversational AI
- Create custom tools and integrations

## Topics Covered

### 1. Introduction to LangChain
- What is LangChain?
- Core concepts and architecture
- Installation and setup

### 2. LangChain Components
- **Models**: LLMs, Chat Models, Text Embedding Models
- **Prompts**: Prompt Templates, Example Selectors
- **Chains**: Sequential chains, Router chains
- **Agents**: Agent types and tools
- **Memory**: Conversation memory, Entity memory

### 3. Building Your First Chain
- Simple LLM chain
- Sequential chains
- Router chains
- Custom chain creation

### 4. Working with Agents
- Agent types (Zero-shot, Conversational, etc.)
- Built-in tools (Calculator, Search, etc.)
- Creating custom tools
- Agent execution and debugging

### 5. Memory Systems
- Conversation buffer memory
- Conversation summary memory
- Entity memory
- Custom memory implementations

## Exercises

1. **Basic Chain Implementation**
   - Create a simple question-answering chain
   - Implement a multi-step reasoning chain
   - Add prompt templates with variables

2. **Agent Development**
   - Build an agent with calculator and search tools
   - Create a custom tool for your specific use case
   - Implement conversational agent with memory

3. **Integration Project**
   - Connect LangChain to external APIs
   - Build a chain that processes files
   - Create a chatbot with persistent memory

## Code Examples

### Python Implementation
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Basic chain example
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief explanation about {topic}"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Execute the chain
result = chain.run(topic="machine learning")
print(result)
```

### JavaScript/TypeScript Implementation
```typescript
import { OpenAI } from "langchain/llms/openai";
import { LLMChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";

// Basic chain example
const llm = new OpenAI({ temperature: 0.7 });
const prompt = new PromptTemplate({
  inputVariables: ["topic"],
  template: "Write a brief explanation about {topic}"
});
const chain = new LLMChain({ llm, prompt });

// Execute the chain
const result = await chain.call({ topic: "machine learning" });
console.log(result.text);
```

### Agent Example (Python)
```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun, Calculator
from langchain.llms import OpenAI

# Initialize tools
search = DuckDuckGoSearchRun()
calculator = Calculator()
tools = [search, calculator]

# Create agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Execute agent
result = agent.run("What is the population of Tokyo multiplied by 2?")
```

### Agent Example (JavaScript/TypeScript)
```typescript
import { OpenAI } from "langchain/llms/openai";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { DynamicTool } from "langchain/tools";
import { Calculator } from "langchain/tools/calculator";

// Create tools
const calculator = new Calculator();
const searchTool = new DynamicTool({
  name: "web-search",
  description: "Search the web for current information",
  func: async (input: string) => {
    // Implement your search logic here
    return `Search results for: ${input}`;
  },
});

const tools = [calculator, searchTool];

// Create agent
const llm = new OpenAI({ temperature: 0 });
const executor = await initializeAgentExecutorWithOptions(
  tools, 
  llm, 
  { agentType: "zero-shot-react-description", verbose: true }
);

// Execute agent
const result = await executor.call({ 
  input: "What is the population of Tokyo multiplied by 2?" 
});
console.log(result.output);
```

## Resources

### Python LangChain
- [LangChain Python Documentation](https://docs.langchain.com/)
- [LangChain Python GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Python Examples](https://github.com/langchain-ai/langchain/tree/master/docs/docs/modules)

### JavaScript/TypeScript LangChain
- [LangChain.js Documentation](https://js.langchain.com/docs/)
- [LangChain.js GitHub Repository](https://github.com/langchain-ai/langchainjs)
- [LangChain.js Quickstart](https://js.langchain.com/docs/get_started/quickstart)
- [LangChain.js Examples](https://github.com/langchain-ai/langchainjs/tree/main/examples)
- [Vercel AI SDK + LangChain.js](https://sdk.vercel.ai/docs/guides/frameworks/langchain)

### Tutorials & Learning
- [LangChain YouTube Channel](https://www.youtube.com/@LangChain)
- [LangChain.js Tutorial Series](https://js.langchain.com/docs/tutorials/)
- [Building AI Apps with LangChain.js](https://vercel.com/guides/langchain-ai-chatbot)
- Community examples and templates

## Next Week Preview
Week 3 will dive into NLP with Hugging Face transformers and model fine-tuning.
