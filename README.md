# Video Lecture RAG Teaching Assistant

## Overview

Video Lecture RAG Teaching Assistant is an AI-powered educational assistant that enables students to ask questions about recorded tutorial videos.

The system processes video lectures, converts speech into text using Whisper, creates semantic embeddings of lecture content, and retrieves the most relevant lecture segments using vector similarity search. The retrieved context is then provided to a Large Language Model (LLM) to generate accurate answers grounded in the lecture material.

A key feature of the system is its ability to identify where a topic was taught within a video lecture, helping students quickly locate relevant learning content.

---

## Features

* Upload tutorial or educational videos
* Automatic audio extraction from video files
* Speech-to-text transcription using Whisper
* Generate structured JSON lecture chunks
* Create vector embeddings from lecture content
* Semantic search using cosine similarity
* Retrieve Top-K relevant lecture segments
* LLM-powered question answering
* Identify where a topic was discussed in a lecture
* Reduce hallucinations through Retrieval-Augmented Generation (RAG)

---

## System Architecture
Video
   ↓
Audio Extraction
   ↓
Whisper Transcription
   ↓
JSON Chunks
   ↓
BGE-M3 Embeddings (Ollama)
   ↓
Stored in DataFrame (.joblib)
   ↓
Cosine Similarity Search
   ↓
Top 5 Relevant Chunks
   ↓
LLM
   ↓
Answer Generation

### 1. Video Processing

* Input educational video lectures
* Extract audio from video files

### 2. Transcription

* Convert audio to text using OpenAI Whisper
* Generate timestamped transcripts

### 3. Chunking

* Split transcripts into smaller lecture chunks
* Store chunks in JSON format

### 4. Embedding Generation

* Convert text chunks into vector embeddings
* Store embeddings for retrieval

### 5. Retrieval

* Convert user query into an embedding
* Compute cosine similarity
* Retrieve the Top 5 most relevant lecture chunks

### 6. Response Generation

* Send retrieved context and user query to an LLM
* Generate context-aware answers

### 7. Topic Localization

* Identify the lecture segment where a topic was taught
* Help students navigate long video lectures efficiently

---

## Example Query

**User Question:**
Where was Gradient Descent explained in the lecture?

**System Output:**
Gradient Descent was discussed in Lecture 3 during the optimization section. The explanation begins around the retrieved transcript segment discussing cost function minimization and parameter updates.

---

## Tech Stack

* Python
* OpenAI Whisper
* Sentence Transformers
* Vector Embeddings
* Cosine Similarity Search
* Large Language Models (LLMs)
* JSON
* Jupyter Notebook

---

## Project Workflow

Video Input
→ Audio Extraction
→ Whisper Transcription
→ JSON Chunk Creation
→ Embedding Generation
→ Vector Storage
→ User Query
→ Cosine Similarity Search
→ Top 5 Retrieval
→ LLM
→ Final Response

---

## Future Improvements

* Support multiple courses simultaneously
* Timestamp-based video navigation
* Hybrid Search (Keyword + Semantic Search)
* Streamlit/Web Application Interface
* Evaluation Metrics for Retrieval Quality
* Multi-language lecture support

---

## Author

Gobind Singh

Machine Learning & Data Science Enthusiast
