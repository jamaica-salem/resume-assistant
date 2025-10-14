import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
import numpy as np
import re
import time
from typing import List, Dict, Tuple
from datetime import datetime

# ==============================
# CONFIGURATION & STYLING
# ==============================
st.set_page_config(
    page_title="Jam's Resume Chatbot",
    layout="centered",
    page_icon="💬",
)

# Dark Mode Glassmorphism CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&display=swap');

/* ============ BASE STYLES ============ */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Fira Sans', sans-serif;
    color: #e4e4e7;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

/* Animated Background Gradient */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,191,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: drift 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes drift {
    0% { transform: translate(0, 0); }
    100% { transform: translate(50px, 50px); }
}

/* ============ MAIN CONTAINER ============ */
div.block-container {
    padding: 2rem 1rem !important;
    max-width: 900px;
    position: relative;
    z-index: 1;
}

/* ============ HEADER SECTION ============ */
.header-container {
    background: rgba(30, 30, 46, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 191, 255, 0.2);
    padding: 2.5rem 2rem;
    border-radius: 24px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00bfff, transparent);
    animation: scan 3s linear infinite;
}

@keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'Fira Sans', sans-serif;
    background: linear-gradient(135deg, #00bfff 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
    text-shadow: 0 0 30px rgba(0, 191, 255, 0.3);
}

.subtitle {
    font-family: 'Fira Sans', sans-serif;
    font-size: 1rem;
    color: #a1a1aa;
    font-weight: 300;
    line-height: 1.6;
    max-width: 600px;
    margin: 0 auto;
}

.info-badge {
    display: inline-block;
    background: rgba(0, 191, 255, 0.1);
    border: 1px solid rgba(0, 191, 255, 0.3);
    color: #00bfff;
    padding: 0.5rem 1.2rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 1.2rem;
    box-shadow: 0 0 20px rgba(0, 191, 255, 0.2);
}

/* ============ CHAT CONTAINER ============ */
.chat-wrapper {
    background: rgba(30, 30, 46, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 191, 255, 0.15);
    padding: 2rem;
    border-radius: 24px;
    min-height: 450px;
    max-height: 550px;
    overflow-y: auto;
    margin-bottom: 1.5rem;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* Custom Scrollbar */
.chat-wrapper::-webkit-scrollbar {
    width: 6px;
}

.chat-wrapper::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
}

.chat-wrapper::-webkit-scrollbar-thumb {
    background: rgba(0, 191, 255, 0.3);
    border-radius: 10px;
}

.chat-wrapper::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 191, 255, 0.5);
}

/* ============ MESSAGE BUBBLES ============ */
.message-container {
    display: flex;
    margin-bottom: 1.5rem;
    animation: fadeSlideIn 0.5s ease-out;
    opacity: 0;
    animation-fill-mode: forwards;
}

@keyframes fadeSlideIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-message {
    justify-content: flex-end;
}

.bot-message {
    justify-content: flex-start;
}

.user-bubble {
    background: linear-gradient(135deg, rgba(0, 191, 255, 0.2) 0%, rgba(124, 58, 237, 0.2) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0, 191, 255, 0.4);
    color: #ffffff;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 4px 20px;
    max-width: 75%;
    box-shadow: 
        0 4px 20px rgba(0, 191, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    font-size: 0.95rem;
    line-height: 1.6;
    font-weight: 400;
    font-family: 'Fira Sans', sans-serif;
    position: relative;
}

.user-bubble::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 20px 20px 4px 20px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(0, 191, 255, 0.3), rgba(124, 58, 237, 0.3));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.bot-bubble {
    background: rgba(39, 39, 42, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(82, 82, 91, 0.3);
    color: #e4e4e7;
    padding: 1rem 1.5rem;
    border-radius: 20px 20px 20px 4px;
    max-width: 75%;
    box-shadow: 
        0 4px 20px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    font-size: 0.95rem;
    font-family: 'Fira Sans', sans-serif;
    line-height: 1.6;
    font-weight: 400;
}

/* ============ TYPING INDICATOR ============ */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 1.5rem;
    background: rgba(39, 39, 42, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(82, 82, 91, 0.3);
    border-radius: 20px 20px 20px 4px;
    max-width: 100px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: #00bfff;
    border-radius: 50%;
    animation: typingBounce 1.4s infinite;
    box-shadow: 0 0 10px rgba(0, 191, 255, 0.5);
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typingBounce {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.7;
    }
    30% {
        transform: translateY(-10px);
        opacity: 1;
    }
}

/* ============ INPUT SECTION ============ */
.input-container {
    background: rgba(30, 30, 46, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 191, 255, 0.2);
    padding: 1.5rem 2rem;
    border-radius: 24px;
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

/* Form container positioning */
.stForm {
    position: relative !important;
}

/* Text input wrapper */
.stTextInput {
    position: relative !important;
}

.stTextInput>div>div>input {
    background: rgba(39, 39, 42, 0.4) !important;
    border: 1px solid rgba(82, 82, 91, 0.3) !important;
    border-radius: 16px !important;
    color: #e4e4e7 !important;
    padding: 1rem 4.5rem 1rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-family: 'Fira Sans', sans-serif !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    width: 100% !important;
}

.stTextInput>div>div>input:focus {
    border: 1px solid rgba(82, 82, 91, 0.3) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    background: rgba(39, 39, 42, 0.6) !important;
    outline: none !important;
    box-shadow: none !important;
}

.stTextInput>div>div>input::placeholder {
    color: #71717a !important;
    font-family: 'Fira Sans', sans-serif !important;
}

/* Submit button positioned inside input */
.stForm button[type="submit"] {
    position: absolute !important;
    right: 6px !important;
    top: 6px !important;
    transform: none !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.65rem 0.9rem !important;
    width: auto !important;
    min-width: auto !important;
    height: calc(100% - 12px) !important;
    font-size: 1.3rem !important;
    line-height: 1 !important;
    transition: all 0.3s ease !important;
    box-shadow: 
        0 4px 15px rgba(16, 185, 129, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    cursor: pointer !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    z-index: 10 !important;
}

.stForm button[type="submit"]:hover {
    transform: scale(1.05) !important;
    box-shadow: 
        0 6px 25px rgba(16, 185, 129, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
}

.stForm button[type="submit"]:active {
    transform: scale(0.95) !important;
}

/* Hide button text, only show arrow via content */
.stForm button[type="submit"] > div {
    display: none !important;
}

.stForm button[type="submit"]::before {
    content: '→' !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    display: block !important;
    line-height: 1 !important;
}

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
    background: rgba(20, 20, 30, 0.9) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(0, 191, 255, 0.2) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Fira Sans', sans-serif !important;
}

[data-testid="stSidebar"] .css-1d391kg {
    padding: 2rem 1rem;
}

[data-testid="stSidebar"] h3 {
    color: #00bfff !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stSuccess,
[data-testid="stSidebar"] .stInfo {
    background: rgba(0, 191, 255, 0.1) !important;
    border: 1px solid rgba(0, 191, 255, 0.3) !important;
    border-radius: 12px !important;
    color: #e4e4e7 !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #a1a1aa !important;
}

/* ============ ALERTS ============ */
.stAlert {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 12px !important;
    color: #fca5a5 !important;
    backdrop-filter: blur(10px) !important;
    font-family: 'Fira Sans', sans-serif !important;
}

.stSpinner > div {
    border-color: #00bfff transparent transparent transparent !important;
}

/* ============ FOOTER ============ */
.footer-text {
    text-align: center;
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.9rem;
    margin-top: 2rem;
    font-weight: 300;
    font-family: 'Fira Sans', sans-serif;
}

.footer-text b {
    font-weight: 600;
    color: #00bfff;
}

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .header-container,
    .chat-wrapper,
    .input-container {
        padding: 1.5rem;
    }
    
    .user-bubble,
    .bot-bubble {
        max-width: 90%;
        font-size: 0.9rem;
    }
}

/* ============ HIDE STREAMLIT BRANDING ============ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================
# INTELLIGENT HELPER FUNCTIONS
# ==============================

class IntentClassifier:
    """Advanced intent classification for better query understanding"""
    
    INTENT_PATTERNS = {
        'factual_specific': {
            'keywords': ['what', 'which', 'where', 'when', 'who', 'how many', 'list'],
            'examples': ['what skills', 'where did', 'which languages', 'how many years']
        },
        'creative_generation': {
            'keywords': ['create', 'generate', 'write', 'draft', 'compose', 'make', 'prepare', 'develop'],
            'examples': ['create questions', 'write cover letter', 'generate summary']
        },
        'comparison': {
            'keywords': ['compare', 'difference between', 'versus', 'vs', 'better than', 'similar to'],
            'examples': ['compare skills', 'difference between projects']
        },
        'recommendation': {
            'keywords': ['recommend', 'suggest', 'advice', 'should', 'best', 'tips'],
            'examples': ['recommend projects', 'what should i highlight', 'best skills']
        },
        'explanation': {
            'keywords': ['explain', 'how does', 'why', 'describe', 'tell me about', 'elaborate'],
            'examples': ['explain experience', 'describe project', 'tell me about background']
        },
        'greeting': {
            'keywords': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
            'examples': ['hello', 'hi there']
        }
    }
    
    @staticmethod
    def classify(query: str) -> str:
        """Classify user intent from query"""
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, data in IntentClassifier.INTENT_PATTERNS.items():
            for keyword in data['keywords']:
                if keyword in query_lower:
                    return intent
        
        return 'factual_specific'  # Default intent

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text: str):
    """Smart chunking that preserves resume structure"""
    text = text.strip()
    
    # Split by major sections
    sections = re.split(r'\n{2,}', text)
    
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) > 30:
            chunks.append(section)
    
    # If we have very few chunks, split more aggressively
    if len(chunks) < 5:
        lines = text.split('\n')
        current_chunk = []
        for line in lines:
            line = line.strip()
            if line:
                current_chunk.append(line)
                if len(current_chunk) >= 3 and (line.endswith(':') or len(current_chunk) >= 5):
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 30:
                        chunks.append(chunk_text)
                    current_chunk = []
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 30:
                chunks.append(chunk_text)
    
    return chunks if chunks else [text]

def create_semantic_search_engine(chunks: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Create enhanced semantic search with better matching"""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),      # Capture phrases up to 3 words
        max_features=2000,        # Larger vocabulary
        min_df=1,
        max_df=0.95,             # Filter very common terms
        sublinear_tf=True        # Use log scaling for term frequency
    )
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X

def get_conversation_context(messages: List[Dict], limit: int = 3) -> str:
    """Extract recent conversation context for better continuity"""
    if len(messages) <= 1:
        return ""
    
    recent = messages[-limit:]
    context_parts = []
    
    for msg in recent:
        if msg['role'] == 'user':
            context_parts.append(f"User asked: {msg['content']}")
        elif msg['role'] == 'assistant':
            # Truncate long responses
            content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            context_parts.append(f"Assistant responded: {content}")
    
    return "\n".join(context_parts)

def enhance_query_with_context(query: str, conversation_history: str) -> str:
    """Enhance query understanding using conversation history"""
    if not conversation_history:
        return query
    
    # Add context about follow-up questions
    followup_indicators = ['more', 'also', 'what about', 'how about', 'and', 'else', 'other']
    if any(indicator in query.lower() for indicator in followup_indicators):
        return f"Given this context:\n{conversation_history}\n\nUser is asking: {query}"
    
    return query

@st.cache_resource
def build_index(pdf_path: str):
    resume_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(resume_text)
    vectorizer, X = create_semantic_search_engine(chunks)
    
    # Extract structured data for quick reference
    structured_data = {
        'skills_section': next((chunk for chunk in chunks if 'TECHNICAL SKILLS' in chunk.upper()), ''),
        'experience_section': next((chunk for chunk in chunks if 'EXPERIENCE' in chunk.upper()), ''),
        'education_section': next((chunk for chunk in chunks if 'EDUCATION' in chunk.upper() or 'CERTIFICATIONS' in chunk.upper()), ''),
        'projects_section': next((chunk for chunk in chunks if 'PROJECTS' in chunk.upper()), ''),
    }
    
    return resume_text, chunks, vectorizer, X, structured_data

def smart_retrieve(query: str, vectorizer, X, chunks, structured_data, top_k: int = 5) -> Tuple[str, float]:
    """Intelligent retrieval with fallback strategies"""
    
    # Strategy 1: Direct section matching for common queries
    query_lower = query.lower()
    if 'skill' in query_lower and structured_data['skills_section']:
        return structured_data['skills_section'], 1.0
    elif 'experience' in query_lower or 'work' in query_lower and structured_data['experience_section']:
        return structured_data['experience_section'], 1.0
    elif 'education' in query_lower or 'study' in query_lower or 'school' in query_lower and structured_data['education_section']:
        return structured_data['education_section'], 1.0
    elif 'project' in query_lower and structured_data['projects_section']:
        return structured_data['projects_section'], 1.0
    
    # Strategy 2: Semantic search with TF-IDF
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()
    best_idx = np.argsort(-sims)[:top_k]
    best_scores = sims[best_idx]
    
    if best_scores[0] > 0.05:
        context = "\n\n".join([chunks[idx] for idx in best_idx])
        return context, best_scores[0]
    
    # Strategy 3: Return empty if no good match
    return "", 0.0

def generate_intelligent_response(query: str, context: str, intent: str, conversation_history: str, full_resume: str, model_name: str) -> str:
    """Generate contextually aware responses based on intent"""
    
    # Build conversation-aware system prompt
    base_system = (
        "You are an intelligent AI assistant representing Jamaica E. Salem. "
        "You have access to their complete resume and conversation history. "
    )
    
    if intent == 'greeting':
        system_prompt = base_system + (
            "Respond warmly and professionally. Introduce yourself briefly and offer to help with questions about Jamaica's resume."
        )
        user_prompt = query
        
    elif intent == 'creative_generation':
        system_prompt = base_system + (
            "Use the resume information to create professional, high-quality content. "
            "Be creative but accurate. Tailor responses to showcase Jamaica's strengths."
        )
        user_prompt = f"Full Resume:\n{full_resume}\n\n{conversation_history}\n\nTask: {query}"
        
    elif intent == 'comparison':
        system_prompt = base_system + (
            "Provide analytical comparisons based on the resume data. "
            "Highlight differences, similarities, and unique aspects clearly."
        )
        user_prompt = f"Resume Context:\n{context}\n\n{conversation_history}\n\nComparison request: {query}"
        
    elif intent == 'recommendation':
        system_prompt = base_system + (
            "Provide strategic recommendations and advice based on Jamaica's background. "
            "Be specific and actionable. Reference actual resume details."
        )
        user_prompt = f"Resume Context:\n{context}\n\nFull Resume:\n{full_resume}\n\n{conversation_history}\n\nAdvice needed: {query}"
        
    elif intent == 'explanation':
        system_prompt = base_system + (
            "Provide detailed, clear explanations. Break down complex information. "
            "Use examples from the resume when relevant."
        )
        user_prompt = f"Context:\n{context}\n\n{conversation_history}\n\nExplain: {query}"
        
    else:  # factual_specific
        system_prompt = base_system + (
            "Answer factual questions accurately and concisely based on the resume. "
            "If information isn't in the context, check the full resume before saying you don't know."
        )
        if context:
            user_prompt = f"Relevant Resume Info:\n{context}\n\n{conversation_history}\n\nQuestion: {query}"
        else:
            user_prompt = f"Full Resume:\n{full_resume}\n\n{conversation_history}\n\nQuestion: {query}"
    
    try:
        resp = chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        try:
            answer = resp['message']['content']
        except Exception:
            answer = getattr(resp, 'message', {}).get('content', str(resp))
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease ensure Ollama is running and the model is available."

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("### 🎯 Configuration")
st.sidebar.success("✅ Resume loaded successfully")
st.sidebar.caption("📄 Source: `jam_resume.pdf`")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Model Settings")
st.sidebar.info("🤖 Model: Gemma3 (Ollama)")
st.sidebar.caption("🧠 Enhanced with Intent Classification")
st.sidebar.caption("💡 Context-Aware Responses")
st.sidebar.caption("🔍 Smart Semantic Search")

# ==============================
# LOAD RESUME
# ==============================
PDF_PATH = "jam_resume.pdf"

try:
    full_resume_text, chunks, vectorizer, X, structured_data = build_index(PDF_PATH)
except Exception as e:
    st.error(f"❌ Could not load resume: {e}")
    st.stop()

# ==============================
# INITIALIZE SESSION STATE
# ==============================
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_thinking' not in st.session_state:
    st.session_state.is_thinking = False
if 'conversation_start' not in st.session_state:
    st.session_state.conversation_start = datetime.now()

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class='header-container'>
    <div class='main-title'>💬 Jam's AI Resume Assistant</div>
    <div class='subtitle'>
        Intelligent AI assistant with <b>context awareness</b>, <b>intent understanding</b>, and <b>conversation memory</b>.
        Ask anything about Jamaica E. Salem!
    </div>
    <div class='info-badge'>✨ Powered by Advanced AI</div>
</div>
""", unsafe_allow_html=True)

# ==============================
# CHAT DISPLAY
# ==============================
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class='message-container bot-message'>
        <div class='bot-bubble'>
            👋 Hello! I'm Jam's intelligent AI assistant with advanced capabilities including:<br><br>
            🧠 <b>Context Awareness</b> - I remember our conversation<br>
            🎯 <b>Intent Recognition</b> - I understand what you need<br>
            🔍 <b>Smart Search</b> - I find relevant information quickly<br><br>
            Try asking me:<br>
            • "What skills does Jam have?"<br>
            • "Create interview questions for a software role"<br>
            • "Compare Jam's frontend and backend experience"<br>
            • "What would you recommend highlighting?"
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class='message-container user-message'>
                <div class='user-bubble'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='message-container bot-message'>
                <div class='bot-bubble'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

# Show typing indicator when processing
if st.session_state.is_thinking:
    st.markdown("""
    <div class='message-container bot-message'>
        <div class='typing-indicator'>
            <div class='typing-dot'></div>
            <div class='typing-dot'></div>
            <div class='typing-dot'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# INPUT SECTION
# ==============================
with st.form(key='chat_form', clear_on_submit=True):
    query = st.text_input(
        "Your question",
        placeholder="e.g., What makes Jam qualified for an AI engineer role?",
        label_visibility="collapsed",
        key="user_input"
    )
    
    submit_button = st.form_submit_button("→")

st.markdown("</div>", unsafe_allow_html=True)

model_name = "gemma3"

if submit_button:
    if not query.strip():
        st.error("⚠️ Please enter a question first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.is_thinking = True
        st.rerun()

# Process the query if thinking state is active
if st.session_state.is_thinking:
    query = st.session_state.messages[-1]['content']
    
    # Step 1: Classify intent
    intent = IntentClassifier.classify(query)
    
    # Step 2: Get conversation context
    conversation_history = get_conversation_context(st.session_state.messages[:-1], limit=4)
    
    # Step 3: Enhance query with context
    enhanced_query = enhance_query_with_context(query, conversation_history)
    
    # Step 4: Smart retrieval
    context, confidence = smart_retrieve(
        enhanced_query, 
        vectorizer, 
        X, 
        chunks, 
        structured_data, 
        top_k=5
    )
    
    # Step 5: Generate intelligent response
    answer = generate_intelligent_response(
        query=query,
        context=context,
        intent=intent,
        conversation_history=conversation_history,
        full_resume=full_resume_text,
        model_name=model_name
    )
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.is_thinking = False
    st.rerun()

st.markdown("""
<div class='footer-text'>
    🧠 Built with <b>Advanced AI</b> + <b>Intelligent Search</b> + <b>Context Memory</b><br>
    Created by <b>Jamaica E. Salem</b>
</div>
""", unsafe_allow_html=True)