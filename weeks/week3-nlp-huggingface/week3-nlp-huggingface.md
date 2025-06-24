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

## Resources
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers Course](https://huggingface.co/course)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [Model Hub](https://huggingface.co/models)
- Papers With Code NLP section

## Next Week Preview
Week 4 will focus on Retrieval Augmented Generation (RAG) systems and vector databases.
