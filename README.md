# ASKME – AI Support for Knowledge Management & Engagement
### (GenAI-Powered College Advisor)

ASKME is an intelligent GenAI-powered assistant designed to transform university support services. It delivers real-time, context-aware, and multilingual responses to student queries — with a special focus on international student needs — through advanced NLP, Retrieval-Augmented Generation (RAG), and fine-tuned Large Language Models (LLMs).

... 

### Problem Statement

University departments are overwhelmed by thousands of repetitive queries each semester, resulting in:

Long response delays during peak periods

Inconsistent or incomplete information from different departments

High dependency on limited staff hours (8 AM – 5 PM)

Lack of 24/7, personalized assistance for critical student issues (visa, scholarship, course selection, etc.)

ASKME addresses these issues by providing an AI-driven, always-available support system tailored to the academic environment.  

### Objectives

✅ Automate academic and administrative FAQ responses using LLMs

✅ Support personalized and multilingual responses with context retention

✅ Reduce human staff workload and response wait times

✅ Build a scalable and accurate AI assistant for university environments

✅ Enable document-level understanding from PDF, web, and JSON content

✅ Track and log queries for analytics, feedback, and improvement

### Tech Stack: 

Tech Stack

Component

Tools / Frameworks

LLMs

GPT-4o, LLaMA 3.3 (70B)

NLP Frameworks

LangChain, SentenceTransformers

Embeddings

BAAI/bge-m3, Word2Vec, TF-IDF

RAG Framework

ChromaDB + LLM-based Generator

Data Collection

Web scraping, manual JSON construction

Preprocessing

PyMuPDF, Regex, Lowercasing, Cleaning Scripts

Backend/API

Python Scripts, Modular RAG Pipeline

Frontend (UI)

Streamlit

Deployment

Local (tested on RTX 3080), future Azure/Salesforce/AWS

Evaluation Tools

Evidently AI, Human Reviews, BERTScore

### Results Summary
BERTScore F1: 0.82–0.88 → High semantic overlap

Human eval: Majority responses rated “Relevant” and “Fluent”

Reranking improves context relevance over vanilla vector similarity

Time: OpenAI GPT-4o model performs significantly faster than Ollama

### Future Work
Integrate agentic AI for workflow automation

Agentic AI for task automation

Dynamic context length control

Enhance multilingual and regional support

Fine-tuning LLMs on domain-specific Q&A

Expansion to multimodal inputs

Real-time chatbot deployment for eider adoption



### Contributors
Christian Bridge

Chandrasekhar Pedalapu