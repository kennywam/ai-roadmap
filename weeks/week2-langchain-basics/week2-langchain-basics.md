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
