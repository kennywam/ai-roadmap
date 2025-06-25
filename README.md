# 🧠 AI Engineer Learning Roadmap

A comprehensive, hands-on learning path for software engineers who want to master modern AI development. This roadmap covers everything from LLM fundamentals to production deployment, with practical projects that reinforce key concepts.

**🚀 Dual-Language Support**: Learn AI development in both **Python** and **JavaScript/TypeScript** with framework-specific resources, examples, and deployment strategies.

> **Goal**: Transform from AI beginner to proficient AI engineer in 7 weeks through structured learning and hands-on projects, using your preferred programming language and tech stack.

---

## 🎯 What You'll Learn

- **LLM Fundamentals**: Understanding transformer architecture, prompt engineering, and model capabilities
- **Framework Mastery**: Deep dive into LangChain, Hugging Face, and modern AI development tools
- **Production Skills**: RAG systems, vector databases, OCR processing, and AI agents
- **MLOps & Deployment**: Monitoring, scaling, and deploying AI applications in production

---

## 📅 7-Week Learning Schedule

| Week | Topic | Key Learning Outcomes | Main Project |
|------|-------|----------------------|-------------|
| **1** | [LLM Foundations](weeks/week1-llm-foundations/) | Transformer architecture, prompt engineering, API integration | Basic LLM interaction scripts |
| **2** | [LangChain Basics](weeks/week2-langchain-basics/) | Chains, agents, memory systems, tool integration | Conversational AI with memory |
| **3** | [NLP + Hugging Face](weeks/week3-nlp-huggingface/) | Model fine-tuning, tokenization, NER, classification | Custom NER system |
| **4** | [RAG Systems](weeks/week4-rag/) | Vector databases, embeddings, retrieval strategies | Document Q&A system |
| **5** | [OCR & Document Processing](weeks/week5-ocr/) | Text extraction, preprocessing, structured data extraction | Invoice processing pipeline |
| **6** | [AI Agents](weeks/week6-agents/) | Autonomous agents, tool calling, multi-agent systems | Research assistant agent |
| **7** | [Advanced LLMOps](weeks/week7-advanced-llmops/) | Production deployment, monitoring, cost optimization | Full production deployment |

---

## 🚀 Hands-On Projects

Each project builds on previous weeks' knowledge and can be added to your portfolio:

### 📄 [Chat with PDF](projects/chat-with-pdf/)
**Technologies**: RAG, LangChain, Vector DBs  
**Skills**: Document processing, semantic search, conversational AI

### 🧾 [Invoice Extractor](projects/invoice-extractor/)
**Technologies**: OCR, NLP, LLMs  
**Skills**: Document understanding, data extraction, structured output

### 🤖 [Autonomous Agent](projects/autonomous-agent/)
**Technologies**: LangGraph, Tools, Multi-agent systems  
**Skills**: Agent orchestration, tool integration, task automation

### 🌐 [Full-Stack SaaS App](projects/fullstack-saas-app/)
**Technologies**: Complete AI application stack  
**Skills**: End-to-end development, user interface, production deployment

---

## 🛠 Technology Stack

### 🐍 Python AI Ecosystem
- **Frameworks**: LangChain, Hugging Face Transformers, FastAPI
- **ML Libraries**: PyTorch, TensorFlow, scikit-learn
- **Vector DBs**: Chroma, FAISS, Weaviate Python clients
- **OCR**: Tesseract, EasyOCR, PaddleOCR
- **Deployment**: Docker, Kubernetes, FastAPI, Streamlit

### 🟨 JavaScript/TypeScript AI Ecosystem
- **Frameworks**: LangChain.js, Vercel AI SDK, TensorFlow.js
- **Full-Stack**: Next.js, NestJS, Node.js, Express
- **Vector DBs**: Pinecone TS SDK, Weaviate JS client, ChromaDB JS
- **OCR**: Tesseract.js, Cloud OCR APIs (Google Vision, AWS)
- **Deployment**: Vercel, Railway, Render, Cloudflare Workers

### 🌐 Universal Tools & Services
- **LLM APIs**: OpenAI, Anthropic, Google PaLM, Cohere
- **Vector Databases**: Pinecone, Weaviate, Chroma, FAISS
- **Cloud Platforms**: AWS, Google Cloud, Azure, Vercel
- **Monitoring**: Weights & Biases, MLflow, LangSmith, Sentry

---

## 📁 Repository Structure

```
ai-roadmap/
├── README.md                 # This overview
├── weeks/                    # Weekly learning modules
│   ├── week1-llm-foundations/
│   ├── week2-langchain-basics/
│   ├── week3-nlp-huggingface/
│   ├── week4-rag/
│   ├── week5-ocr/
│   ├── week6-agents/
│   └── week7-advanced-llmops/
├── projects/                 # Hands-on project implementations
│   ├── chat-with-pdf/
│   ├── invoice-extractor/
│   ├── autonomous-agent/
│   └── fullstack-saas-app/
└── resources.md             # Additional learning resources
```

---

## 🔄 Dual-Language Learning Approach

This roadmap uniquely supports both **Python** and **JavaScript/TypeScript** developers:

### 🏗️ **Unified Curriculum**
- Same AI concepts and techniques across both languages
- Week-by-week progression regardless of your tech stack choice
- Consistent project outcomes using different implementation approaches

### 📚 **Language-Specific Resources**
Each week includes:
- **Python Resources**: Traditional ML stack (LangChain, Hugging Face, FastAPI)
- **JavaScript/TypeScript Resources**: Modern web stack (LangChain.js, Vercel AI SDK, Next.js)
- **Universal Concepts**: API usage, deployment strategies, best practices

### 🎯 **Flexible Learning Paths**
- **Single Language**: Focus on your preferred language throughout
- **Polyglot Approach**: Learn concepts in both languages for maximum versatility
- **Team Learning**: Different team members can follow different tracks while building the same projects

### 🚀 **Real-World Alignment**
- **Python Track**: Ideal for ML engineers, data scientists, AI researchers
- **JavaScript/TypeScript Track**: Perfect for full-stack developers, frontend engineers, startup builders
- **Both Tracks**: Essential for AI engineering leads and architecture roles

---

## 🎓 Learning Approach

### 📖 Study Method
1. **Read** the weekly material thoroughly
2. **Code** along with examples and exercises
3. **Build** the suggested projects
4. **Experiment** with different approaches
5. **Document** your learnings and challenges

### ⏱ Time Commitment
- **Estimated**: 10-15 hours per week
- **Flexibility**: Self-paced learning
- **Focus**: 70% hands-on coding, 30% theory

### 📊 Progress Tracking
- Weekly learning objectives with checkboxes
- Practical exercises to reinforce concepts
- Project milestones to measure progress
- Resource links for deeper exploration

---

## 🚀 Getting Started

### Prerequisites

**Choose Your Learning Path:**

**🐍 Python Track:**
- Basic Python programming knowledge
- Familiarity with pip and virtual environments
- Understanding of REST APIs

**🟨 JavaScript/TypeScript Track:**
- JavaScript/TypeScript fundamentals
- Node.js and npm experience
- Understanding of async/await and promises

**🌐 Universal:**
- Familiarity with web development concepts
- Basic understanding of APIs and HTTP
- Git version control knowledge
- Understanding of machine learning concepts (helpful but not required)

### Setup Instructions
1. **Clone this repository**
   ```bash
   git clone https://github.com/kennywam/ai-roadmap
   cd ai-roadmap
   ```

2. **Start with Week 1**
   ```bash
   cd weeks/week1-llm-foundations
   # Follow the instructions in the markdown file
   ```

3. **Set up your development environment**
   - Install Python 3.8+
   - Set up virtual environment
   - Install required packages (detailed in each week)

4. **Get API keys**
   - OpenAI API key
   - Other service keys as needed

### Learning Path
- Follow weeks **sequentially** for optimal learning
- Complete **exercises** before moving to the next week
- Build **projects** to solidify understanding
- Refer to **resources.md** for additional materials

---

## 🤝 Contributing

This roadmap is designed to evolve with the rapidly changing AI landscape:

- **Feedback**: Share your learning experience
- **Updates**: Suggest new tools or techniques
- **Improvements**: Enhance existing content
- **Projects**: Add new project ideas

---

## 📞 Support

Having trouble with the material? Here are some resources:

- **Issues**: Open a GitHub issue for specific problems
- **Discussions**: Use GitHub discussions for general questions
- **Resources**: Check `resources.md` for additional learning materials
- **Community**: Join AI/ML communities listed in resources

---

**Ready to start your AI engineering journey?** 🚀

👉 **[Begin with Week 1: LLM Foundations](weeks/week1-llm-foundations/)**

---

*Last updated: June 2025 | This roadmap reflects current best practices in AI development*

