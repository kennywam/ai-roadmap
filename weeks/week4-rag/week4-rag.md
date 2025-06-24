# Week 4: Retrieval Augmented Generation (RAG)

## Learning Objectives
- Understand the RAG architecture and its benefits
- Learn to implement vector databases and embeddings
- Build end-to-end RAG systems
- Optimize retrieval quality and performance

## Topics Covered

### 1. Introduction to RAG
- What is Retrieval Augmented Generation?
- Benefits over pure generative models
- RAG vs Fine-tuning trade-offs
- Use cases and applications

### 2. Vector Embeddings and Similarity Search
- Text embeddings fundamentals
- Embedding models (OpenAI, Sentence-BERT, etc.)
- Vector similarity metrics (cosine, dot product, euclidean)
- Embedding quality evaluation

### 3. Vector Databases
- Popular vector databases (Pinecone, Weaviate, Chroma, FAISS)
- Indexing strategies (HNSW, IVF, LSH)
- Metadata filtering and hybrid search
- Performance optimization

### 4. Document Processing Pipeline
- Text chunking strategies
- Preprocessing and cleaning
- Handling different document formats
- Maintaining context and relationships

### 5. RAG System Architecture
- Retrieval component design
- Generation component integration
- Prompt engineering for RAG
- Evaluation metrics and testing

### 6. Advanced RAG Techniques
- Hierarchical retrieval
- Multi-query retrieval
- Re-ranking and fusion
- Query expansion and reformulation

## Exercises

1. **Basic RAG Implementation**
   - Set up a vector database
   - Implement document ingestion pipeline
   - Build a simple Q&A system
   - Test with different chunk sizes

2. **Advanced RAG System**
   - Implement hybrid search (vector + keyword)
   - Add metadata filtering
   - Implement re-ranking strategies
   - Add conversation memory

3. **RAG Evaluation Project**
   - Create evaluation datasets
   - Implement retrieval metrics (precision, recall)
   - Measure generation quality
   - Compare different embedding models

## Code Examples

### Python Implementation
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

# Load and process documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create RAG chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

# Query the system
result = qa_chain({"query": "What is the main topic of the document?"})
print(result["result"])
```

### JavaScript/TypeScript Implementation
```typescript
import { Chroma } from "langchain/vectorstores/chroma";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { RetrievalQAChain } from "langchain/chains";
import { OpenAI } from "langchain/llms/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";

// Load and process documents
const loader = new TextLoader("document.txt");
const documents = await loader.load();

// Split documents into chunks
const textSplitter = new CharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const texts = await textSplitter.splitDocuments(documents);

// Create vector store
const embeddings = new OpenAIEmbeddings();
const vectorStore = await Chroma.fromDocuments(texts, embeddings);

// Create RAG chain
const llm = new OpenAI({ temperature: 0 });
const qaChain = RetrievalQAChain.fromLLM(
  llm,
  vectorStore.asRetriever({ k: 4 })
);

// Query the system
const result = await qaChain.call({
  query: "What is the main topic of the document?"
});
console.log(result.text);
```

### Advanced RAG with Pinecone (Python)
```python
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-env")
index_name = "rag-example"

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

# Set up memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational RAG chain
llm = OpenAI(temperature=0)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# Query with conversation context
result = qa_chain({"question": "What are the key findings?"})
print(result["answer"])
print("Sources:", [doc.metadata for doc in result["source_documents"]])
```

### Advanced RAG with Pinecone (JavaScript/TypeScript)
```typescript
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { OpenAI } from "langchain/llms/openai";
import { PineconeClient } from "@pinecone-database/pinecone";

// Initialize Pinecone
const pinecone = new PineconeClient();
await pinecone.init({
  apiKey: "your-api-key",
  environment: "your-env",
});
const index = pinecone.Index("rag-example");

// Create embeddings and vector store
const embeddings = new OpenAIEmbeddings();
const vectorStore = await PineconeStore.fromExistingIndex(
  embeddings,
  { pineconeIndex: index }
);

// Set up memory for conversation
const memory = new BufferMemory({
  memoryKey: "chat_history",
  returnMessages: true,
});

// Create conversational RAG chain
const llm = new OpenAI({ temperature: 0 });
const qaChain = ConversationalRetrievalQAChain.fromLLM(
  llm,
  vectorStore.asRetriever(),
  {
    memory,
    returnSourceDocuments: true,
  }
);

// Query with conversation context
const result = await qaChain.call({
  question: "What are the key findings?"
});
console.log(result.text);
console.log("Sources:", result.sourceDocuments.map(doc => doc.metadata));
```

## Resources

### Core Papers
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Original RAG research
- [LangChain RAG Concepts](https://python.langchain.com/docs/concepts/rag/)
- [LangChain RAG Guide Part 1](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain RAG Guide Part 2](https://python.langchain.com/docs/tutorials/qa_chat_history/)

### Python Resources
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Haystack Documentation](https://github.com/deepset-ai/haystack)

### JavaScript/TypeScript Resources
- [LangChain.js RAG Guide](https://js.langchain.com/docs/use-cases/retrieval-augmented-generation)
- [Pinecone TypeScript SDK](https://github.com/pinecone-io/pinecone-ts-client)
- [Weaviate JavaScript Client](https://weaviate.io/developers/weaviate/current/client-libraries/javascript)
- [ChromaDB JavaScript SDK](https://js.chromadb.com/)

### Vector Database Documentation
- [Pinecone Documentation](https://docs.pinecone.io/docs/overview)
- [Weaviate Documentation](https://weaviate.io/docs/)
- [Chroma Documentation](https://docs.trychroma.com/)

### Evaluation Frameworks and Benchmarks
- RAG evaluation frameworks and benchmarks
- [BEIR Benchmark](https://github.com/UKPLab/beir)
- [Dense Passage Retrieval (DPR)](https://github.com/facebookresearch/DPR)

## Next Week Preview
Week 5 will focus on Optical Character Recognition (OCR) and document processing.
