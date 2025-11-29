# 💬 Jam's AI Resume Assistant

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![Ollama](https://img.shields.io/badge/Ollama-AI-blueviolet)](https://ollama.com/)  

---

## 🚀 Project Overview

**Jam's AI Resume Assistant** is an intelligent AI-powered chatbot built with **Streamlit** that provides interactive insights about Jamaica E. Salem's resume.  
It leverages **semantic search**, **intent classification**, and **context-aware responses** to answer questions about skills, experience, education, projects, and more.  

💡 **Key Features**:
- 🧠 **Context Awareness** – Remembers conversation history for follow-up questions  
- 🎯 **Intent Recognition** – Understands user query type (factual, creative, comparison, recommendation, etc.)  
- 🔍 **Smart Semantic Search** – Finds relevant sections from a PDF resume using TF-IDF  
- ✨ **Stylish Dark Mode UI** – Glassmorphism design with animated gradients and smooth chat bubbles  
- 🤖 **AI-Powered Responses** – Uses **Ollama Gemma3** for intelligent, personalized answers  

---

## 🛠️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)  
- **PDF Parsing**: [pdfplumber](https://github.com/jsvine/pdfplumber)  
- **NLP & Semantic Search**: [scikit-learn](https://scikit-learn.org/stable/) (TF-IDF & cosine similarity)  
- **AI Chat Model**: [Ollama Gemma3](https://ollama.com/)  
- **Python Libraries**: `numpy`, `re`, `time`, `datetime`, `typing`  

---

## ⚡ Features in Detail

### 1. Intent Classification
- Classifies queries into types: `factual_specific`, `creative_generation`, `comparison`, `recommendation`, `explanation`, `greeting`.
- Helps tailor responses for more accurate answers.  

### 2. Smart Resume Retrieval
- Splits resume PDF into structured chunks for better matching.  
- Uses TF-IDF semantic search to retrieve relevant sections.  
- Direct section matching for common queries: skills, experience, education, projects.  

### 3. Context-Aware Responses
- Keeps conversation memory for follow-up questions.  
- Enhances queries using recent conversation context.  
- AI generates high-quality responses using full resume information.  

### 4. Interactive UI
- Modern **glassmorphic dark mode** design  
- Animated chat bubbles with typing indicators  
- Sidebar shows configuration and model settings  
- Mobile-friendly and responsive  
://github.com/jamaica-salem/jams-ai-resume-assistant.git
cd jams-ai-resume-assistant
