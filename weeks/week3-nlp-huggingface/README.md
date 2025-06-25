# Week 3: NLP with Hugging Face

## Learning Objectives
- Master the Hugging Face ecosystem (Transformers, Datasets, Tokenizers)
- Learn to fine-tune pre-trained models for specific tasks
- Understand different NLP tasks and model architectures
- Deploy models using Hugging Face Hub

## Topics Covered

### 1. Hugging Face Ecosystem Overview
- Transformers library
- Datasets library
- Tokenizers library
- Hugging Face Hub and Model Cards

### 2. Working with Pre-trained Models
- Loading models and tokenizers
- Text classification
- Named Entity Recognition (NER)
- Question Answering
- Text generation
- Summarization

### 3. Tokenization Deep Dive
- Different tokenization strategies
- Subword tokenization (BPE, WordPiece, SentencePiece)
- Handling special tokens
- Custom tokenizer training

### 4. Fine-tuning Models
- Transfer learning concepts
- Fine-tuning for classification tasks
- Fine-tuning for generation tasks
- Training optimization and hyperparameters
- Evaluation metrics

### 5. Model Deployment
- Saving and loading custom models
- Hugging Face Hub integration
- Model versioning and documentation
- Inference optimization

## Exercises

1. **Text Classification Project**
   - Fine-tune BERT for sentiment analysis
   - Evaluate model performance
   - Compare with different base models

2. **Custom NER System**
   - Create a dataset for custom entity recognition
   - Fine-tune a NER model
   - Implement inference pipeline

3. **Text Generation Fine-tuning**
   - Fine-tune GPT-2 on custom dataset
   - Implement controlled generation
   - Evaluate generation quality

## Code Examples

### Python Implementation
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load pre-trained model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
classifier = pipeline("sentiment-analysis", 
                     model=model, 
                     tokenizer=tokenizer)

# Fine-tuning example
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

### JavaScript/TypeScript Implementation
```typescript
import { HfInference } from "@huggingface/inference";

// Initialize Hugging Face Inference
const hf = new HfInference("your-hf-api-token");

// Sentiment analysis example
async function sentimentAnalysis(text: string) {
  const result = await hf.sentiment({
    model: "bert-base-uncased",
    inputs: text,
  });
  console.log(result);
}

sentimentAnalysis("I love using Transformers.js for my projects!");
```

### Fine-tuning in JavaScript/TypeScript
```typescript
// Here we provide a conceptual example as fine-tuning is typically done in Python.
// Leverage Node.js with TensorFlow.js for training or use Python scripts in a Node.js app.
console.log("Fine-tuning is generally recommended to be performed using Python APIs with frameworks such as PyTorch or TensorFlow.");
```
```

## Resources

### Python Hugging Face
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Transformers Course](https://huggingface.co/course)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Datasets Library](https://huggingface.co/docs/datasets)
- [Tokenizers Documentation](https://huggingface.co/docs/tokenizers)

### JavaScript/TypeScript Hugging Face
- [Hugging Face.js Documentation](https://huggingface.co/docs/huggingface.js)
- [Hugging Face.js GitHub](https://github.com/huggingface/huggingface.js)
- [Transformers.js](https://github.com/xenova/transformers.js) - Run Transformers in the browser
- [Hugging Face Inference API](https://huggingface.co/docs/api-inference) - JavaScript client
- [TensorFlow.js Models](https://github.com/tensorflow/tfjs-models) - Pre-trained models for JS

### NLP in JavaScript/TypeScript
- [Natural (Node.js)](https://github.com/NaturalNode/natural) - NLP library for Node.js
- [Compromise](https://github.com/spencermountain/compromise) - Natural language processing
- [ML5.js](https://ml5js.org/) - Machine learning for creative coding
- [TensorFlow.js Text Processing](https://www.tensorflow.org/js/tutorials/training/text_classification)

### General Resources
- [Model Hub](https://huggingface.co/models)
- [Papers With Code NLP section](https://paperswithcode.com/area/natural-language-processing)
- [Hugging Face Spaces](https://huggingface.co/spaces) - Interactive demos

## Next Week Preview
Week 4 will focus on Retrieval Augmented Generation (RAG) systems and vector databases.
