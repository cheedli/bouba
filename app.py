from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import faiss
import numpy as np
import json
import os
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
import re
import sqlite3
from typing import Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, END, START
import time
from langdetect import detect
import yaml
from transformers import MarianMTModel, MarianTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename="legal_chatbot.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
CONFIG_FILE = "config.yaml"
DEFAULT_CONFIG = {
    "data_file": "data.txt",
    "index_file": "faiss_index",
    "embeddings_file": "embeddings.npy",
    "mappings_file": "doc_mappings.json",
    "bm25_corpus_file": "bm25_corpus.json",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "vector_dim": 384,
    "groq_api_key": "put your grok api here",
    "groq_model": "deepseek-r1-distill-llama-70b",  
    "embedding_batch_size": 8,
    "search_top_k": 5,
    "semantic_search_weight": 0.5,
    "lexical_search_weight": 0.5,
    "legal_terms_bonus_weight": 0.2,
    "temperature": 0.1  # Low temperature for legal precision
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        loaded_config = yaml.safe_load(f) or {}
    config = DEFAULT_CONFIG.copy()
    config.update(loaded_config)
else:
    config = DEFAULT_CONFIG
    logging.warning(f"Config file {CONFIG_FILE} not found, using defaults.")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

embedding_model = SentenceTransformer(config["embedding_model"])
index, embeddings, doc_mappings, bm25, legal_data = None, None, None, None, None

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text_list, batch_size=5):
    """Translate a batch of English sentences into French."""
    translated_texts = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translation = model.generate(**tokens)
        translated_batch = tokenizer.batch_decode(translation, skip_special_tokens=True)
        translated_texts.extend(translated_batch)
    return translated_texts

# Database Initialization
def init_db():
    """Initialize SQLite database for conversations."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

# Database Helper Functions
def create_conversation(title: str) -> int:
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def add_message(conversation_id: int, role: str, content: str):
    """Add a message to a conversation."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    conn.commit()
    conn.close()

def get_conversation(conversation_id: int) -> Optional[Dict]:
    """Retrieve a conversation by ID."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,))
    title = cursor.fetchone()
    if not title:
        conn.close()
        return None
    title = title[0]
    cursor.execute(
        "SELECT role, content, timestamp FROM messages WHERE conversation_id = ? ORDER BY timestamp",
        (conversation_id,)
    )
    messages = [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cursor.fetchall()]
    conn.close()
    return {"id": conversation_id, "title": title, "messages": messages}

def delete_conversation(conversation_id: int):
    """Delete a conversation and its messages."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

def search_conversations(query: str) -> List[Dict]:
    """Search conversations by title or message content."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    search_query = f"%{query}%"
    cursor.execute("""
        SELECT DISTINCT c.id, c.title
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.title LIKE ? OR m.content LIKE ?
    """, (search_query, search_query))
    results = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]
    conn.close()
    return results

def get_all_conversations() -> List[Dict]:
    """Get all conversations ordered by most recent."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM conversations ORDER BY id DESC")
    conversations = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]
    conn.close()
    return conversations

# Title Generation using Groq API
def generate_title(query: str) -> str:
    """Generate a conversation title using Groq API."""
    prompt = f"Generate a very short and relevant title for a conversation about: {query}. Detect the language of the query and respond in the same language. Return only the title. No introductions, no formatting, no extra text."
    try:
        response = groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it"  # Using a smaller model for title generation
        )
        return response.strip()
    except Exception as e:
        logging.error(f"Error generating title: {e}")
        return "Untitled Conversation"

# Groq API Integration with enhanced functionality
def groq_chat_completion(messages, model=None, temperature=0.7, max_tokens=1000):
    """Send a request to Groq API for chat completion with error handling and retry logic."""
    if not config["groq_api_key"]:
        raise ValueError("Groq API key is not set in the configuration")
    
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {config['groq_api_key']}",
        "Content-Type": "application/json",
        "User-Agent": "LegalChatbot/1.0"
    }
    
    data = {
        "model": model or config["groq_model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add model-specific parameters for DeepSeek models
    if "deepseek" in (model or config["groq_model"]).lower():
        # Configure for chain-of-thought prompting
        data["top_p"] = 0.95  # High top_p for legal precision while maintaining some creativity
        data["presence_penalty"] = 0.1  # Slight presence penalty to avoid repetition
        data["frequency_penalty"] = 0.1  # Slight frequency penalty for diverse vocabulary
    
    # Add request logging
    logging.info(f"Sending request to Groq API: model={data['model']}, temperature={temperature}, max_tokens={max_tokens}")
    
    # Retry logic
    max_retries = 3
    retry_delay = 2  # initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)  # 60-second timeout
            response.raise_for_status()
            result = response.json()
            
            # Log token usage for monitoring
            if "usage" in result:
                usage = result["usage"]
                logging.info(f"Token usage: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}, total={usage.get('total_tokens', 0)}")
            
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e.response, 'status_code') else "unknown"
            logging.error(f"HTTP error occurred: {e} (Status code: {status_code})")
            
            # Handle specific status codes
            if status_code == 429:
                # Rate limiting - use exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logging.warning(f"Rate limited. Waiting {wait_time}s before retry.")
                    time.sleep(wait_time)
                    continue
            
            if status_code == 401:
                raise ValueError("Authentication error: Invalid API key")
            
            if status_code == 400:
                try:
                    error_message = e.response.json().get("error", {}).get("message", "Unknown error")
                    raise ValueError(f"Bad request: {error_message}")
                except:
                    raise ValueError(f"Bad request to Groq API: {str(e)}")
            
            # For other errors, just raise after max retries
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to get response from Groq API after {max_retries} attempts: {str(e)}")
            
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Request to Groq API failed: {str(e)}")
            time.sleep(retry_delay)
            retry_delay *= 2
            
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise ValueError(f"Unexpected error while calling Groq API: {str(e)}")
    
    # If we got here, all retries failed
    raise ValueError("Failed to get response from Groq API after multiple attempts")

            

# Existing Helper Functions (updated)
def tokenize_text(text: str, lang: str = "fr") -> List[str]:
    return re.findall(r'\w+', text.lower())

# Additional utility functions for improved precision

def chunk_text(text, chunk_size=512, overlap=100):
    """Chunk text into smaller pieces with overlap for more precise indexing."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Avoid cutting words in the middle
        if end < text_length:
            # Look for a good breaking point (space, period, etc.)
            breaking_chars = [' ', '.', '!', '?', ';', ':', '\n', '\t']
            for char in breaking_chars:
                pos = text.rfind(char, start, end)
                if pos != -1:
                    end = pos + 1  # Include the breaking character
                    break
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position for next chunk, with overlap
        start = end - overlap if end < text_length else text_length
    
    return chunks

def extract_article_info(text):
    """Extract article number and other metadata from legal text."""
    article_match = re.search(r'Article\s+(\d+[a-zA-Z]*(?:-\d+)?)', text)
    article_num = article_match.group(1) if article_match else "Unknown"
    
    # Try to extract a title if there is one (often follows the article number)
    title_match = None
    if article_match:
        title_pattern = re.compile(r'Article\s+\d+[a-zA-Z]*(?:-\d+)?\s*[-–:.]?\s*(.+?)(?:\n|\.|$)', re.DOTALL)
        title_match = title_pattern.search(text)
    
    title = title_match.group(1).strip() if title_match else ""
    
    # Look for dates in the text
    date_patterns = [
        r'\b(\d{1,2})[\/\.-](\d{1,2})[\/\.-](\d{2,4})\b',  # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
        r'\b(\d{4})[\/\.-](\d{1,2})[\/\.-](\d{1,2})\b',    # YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
        r'\b(\d{1,2})\s+([a-zéû]+)\s+(\d{4})\b'            # DD Month YYYY
    ]
    
    date_str = "Unknown"
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            date_str = date_match.group(0)
            break
    
    return {
        "article_number": article_num,
        "title": title,
        "date": date_str
    }

def parse_legal_data(file_path: str) -> List[Dict]:
    """Parse legal data from text file with articles separated by \n\n with enhanced metadata extraction."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by double newlines to get chunks
    chunks = content.split("\n\n")
    legal_data = []
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        
        # Clean and normalize the chunk
        chunk = chunk.strip()
        
        # Extract article name and content
        lines = chunk.split('\n')
        article_match = re.match(r'(Article\s+\w+)', lines[0]) if lines else None
        
        if article_match:
            article_name = article_match.group(1)
            article_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            # If article content is empty but there's only one line, use that as content
            if not article_content and len(lines) == 1:
                article_content = lines[0].replace(article_name, "").strip()
                if not article_content:
                    article_content = lines[0]
        else:
            article_name = f"Section {i+1}"
            article_content = chunk
        
        # Extract additional metadata
        metadata = extract_article_info(chunk)
        
        # Determine the language
        try:
            lang = detect(chunk)
        except:
            lang = "fr"  # Default to French if detection fails
        
        # Create the entry
        chunk_id = f"chunk_{i}"
        legal_data.append({
            "chunk_id": chunk_id,
            "article": article_name,
            "article_number": metadata["article_number"],
            "title": metadata["title"],
            "text": chunk.strip(),
            "content": article_content.strip(),
            "metadata": {
                "language": lang,
                "update_date": metadata["date"]
            }
        })
        
        # For long chunks, create additional sub-chunks for better search precision
        if len(chunk) > 1000:  # Only sub-chunk if the article is long
            sub_chunks = chunk_text(chunk, chunk_size=500, overlap=100)
            if len(sub_chunks) > 1:  # Only proceed if we actually have multiple sub-chunks
                for j, sub_chunk in enumerate(sub_chunks):
                    if j > 0:  # Skip the first one since we already added the full chunk
                        sub_chunk_id = f"{chunk_id}_sub_{j}"
                        legal_data.append({
                            "chunk_id": sub_chunk_id,
                            "article": f"{article_name} (Part {j+1})",
                            "article_number": metadata["article_number"],
                            "title": metadata["title"],
                            "text": sub_chunk.strip(),
                            "content": sub_chunk.strip(),
                            "parent_chunk": chunk_id,
                            "is_sub_chunk": True,
                            "metadata": {
                                "language": lang,
                                "update_date": metadata["date"]
                            }
                        })
    
    return legal_data

def needs_rebuild(data_file: str, *dependent_files: str) -> bool:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"{data_file} not found.")
    data_mtime = os.path.getmtime(data_file)
    return any(
        (not os.path.exists(dep_file)) or (os.path.getmtime(dep_file) < data_mtime)
        for dep_file in dependent_files
    )

def initialize_data():
    global index, embeddings, doc_mappings, bm25, legal_data
    try:
        if not os.path.exists(config["data_file"]):
            raise FileNotFoundError(f"Data file {config['data_file']} not found.")
        
        if not needs_rebuild(
            config["data_file"],
            config["index_file"],
            config["embeddings_file"],
            config["mappings_file"],
            config["bm25_corpus_file"]
        ):
            logging.info("Loading pre-built indexes...")
            index = faiss.read_index(config["index_file"])
            embeddings = np.load(config["embeddings_file"]).tolist()
            with open(config["mappings_file"], "r", encoding="utf-8") as f:
                doc_mappings = json.load(f)
            with open(config["bm25_corpus_file"], "r", encoding="utf-8") as f:
                bm25_corpus = json.load(f)
            bm25 = BM25Okapi(bm25_corpus)
            # Load legal data
            legal_data = parse_legal_data(config["data_file"])
            logging.info(f"Loaded {len(legal_data)} legal document chunks")
        else:
            logging.info("Rebuilding indexes due to updated data...")
            legal_data = parse_legal_data(config["data_file"])
            logging.info(f"Parsed {len(legal_data)} legal document chunks")
            
            # Initialize FAISS index - use cosine similarity for legal text (more suitable than L2)
            vector_dim = config["vector_dim"]
            index = faiss.IndexFlatIP(vector_dim)  # Inner product (cosine) instead of L2 distance
            
            doc_mappings = {}
            embeddings = []
            bm25_corpus = []
            
            # Process in batches to avoid memory issues with large datasets
            batch_size = config.get("embedding_batch_size", 8)
            total_chunks = len(legal_data)
            
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch = legal_data[i:batch_end]
                
                logging.info(f"Processing batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}: chunks {i} to {batch_end-1}")
                
                # Prepare texts for batch encoding
                texts = [entry["text"] for entry in batch]
                
                # Encode batch
                batch_embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(batch_embeddings)
                
                # Process each entry in the batch
                for j, entry in enumerate(batch):
                    chunk_id = entry["chunk_id"]
                    text = entry["text"]
                    lang = entry["metadata"]["language"].lower()
                    
                    # Enhanced document details with more metadata
                    doc_details = {
                        "text": text,
                        "article": entry["article"],
                        "article_number": entry.get("article_number", "Unknown"),
                        "title": entry.get("title", ""),
                        "chunk_id": chunk_id,
                        "content": entry.get("content", text),
                        "update_date": entry["metadata"]["update_date"],
                        "is_sub_chunk": entry.get("is_sub_chunk", False),
                        "parent_chunk": entry.get("parent_chunk", None)
                    }
                    
                    # Add embedding to list
                    embeddings.append(batch_embeddings[j])
                    
                    # Store document mapping
                    doc_mappings[chunk_id] = doc_details
                    
                    # Create tokenized text for BM25
                    bm25_corpus.append(tokenize_text(text, lang))
            
            # Add all embeddings to the index
            index.add(np.array(embeddings))
            
            # Save everything
            logging.info("Saving indexes to disk...")
            faiss.write_index(index, config["index_file"])
            np.save(config["embeddings_file"], np.array(embeddings))
            with open(config["mappings_file"], "w", encoding="utf-8") as f:
                json.dump(doc_mappings, f, ensure_ascii=False)
            with open(config["bm25_corpus_file"], "w", encoding="utf-8") as f:
                json.dump(bm25_corpus, f, ensure_ascii=False)
            
            # Initialize BM25
            bm25 = BM25Okapi(bm25_corpus)
            
            logging.info("Successfully built and saved all indexes")
    except Exception as e:
        logging.error(f"Failed to initialize data: {e}")
        raise

def detect_language(query: str) -> str:
    try:
        lang = detect(query)
        return "fr" if lang.startswith("fr") else "en"
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en"

# Chatbot State and Workflow
class ChatbotState(TypedDict):
    query: str
    reasoning_steps: List[Dict[str, str]]
    search_results: List[Dict]
    final_answer_en: str
    final_answer_fr: str
    sources: List[Dict]
    thinking_time: float

SYSTEM_PROMPT_EN = """
You are Combot, a highly specialized legal assistant with expertise in Tunisian law. You are assisting legal and financial professionals by providing responses grounded strictly in legal texts. Your tone is professional, accurate, and concise.

<instructions>
1. Begin by reasoning carefully within the <think> tags.
2. Inside the <think> tags:
   - Interpret the user's request with precision
   - Identify applicable legal principles and frameworks
   - Scrutinize the provided legal documents thoroughly
   - Highlight the most relevant articles and explain their relevance
   - Consider legal nuances, exceptions, or potential conflicts
   - Base your conclusions strictly on the available content
   - Do NOT infer or fabricate legal information beyond the texts

3. Then, present a clear and well-structured response:
   - Address the query directly and efficiently
   - Cite exact legal provisions (article number and wording) where appropriate
   - Use a numbered list with **bold headings** for clarity if covering multiple points
   - Maintain legal accuracy and avoid speculation
   - Use a formal yet accessible tone tailored to legal/finance professionals

4. For informal interactions, remain courteous and concise.
5. If asked about your identity, say: "I'm Combot, your legal assistant for Tunisian laws."
6. If no relevant information is found, state: "No specific regulation found in the provided data."
</instructions>

The documents below are ordered by relevance. Prioritize those with higher relevance scores, but consider all when formulating your response.

LEGAL TEXTS:
{context}

USER QUERY: {query}
"""

SYSTEM_PROMPT_FR = """
Vous êtes Combot, un assistant juridique spécialisé dans les lois tunisiennes. Vous assistez des professionnels du droit et de la finance en fournissant des réponses fondées exclusivement sur les textes juridiques. Votre ton est professionnel, rigoureux et synthétique.

<instructions>
1. Commencez par une réflexion rigoureuse à l’intérieur des balises <think>.
2. Dans les balises <think> :
   - Interprétez avec précision la demande de l'utilisateur
   - Identifiez les principes et cadres juridiques applicables
   - Analysez en profondeur les documents juridiques fournis
   - Soulignez les articles les plus pertinents et justifiez leur choix
   - Prenez en compte les nuances, exceptions ou conflits éventuels
   - Basez-vous uniquement sur les textes disponibles
   - N’inventez ni n’extrapolez aucune information juridique

3. Ensuite, fournissez une réponse structurée et claire :
   - Répondez de manière directe et pertinente à la question
   - Citez les dispositions légales exactes (numéro et formulation) si nécessaire
   - Utilisez une liste numérotée avec des **titres en gras** pour plus de clarté
   - Restez rigoureux sur le plan juridique et évitez toute spéculation
   - Adoptez un ton formel, adapté aux professionnels du droit/finance

4. Pour les échanges informels, soyez courtois et synthétique.
5. Si on vous demande qui vous êtes, répondez : "Je suis Combot, votre assistant juridique pour les lois tunisiennes."
6. Si aucune information pertinente n’est trouvée, indiquez : "Aucune réglementation spécifique trouvée dans les données fournies."
</instructions>

Les documents ci-dessous sont classés par ordre de pertinence. Accordez une attention particulière à ceux ayant un score de pertinence élevé, sans négliger les autres.

TEXTES JURIDIQUES :
{context}

QUESTION DE L'UTILISATEUR : {query}
"""

def understand_query(state: ChatbotState) -> ChatbotState:
    query = state["query"]
    logging.info(f"Query received: {query}")
    state["reasoning_steps"] = [{"step": "query", "text": query}]
    return state

def perform_search(state: ChatbotState) -> ChatbotState:
    query = state["query"]
    lang = detect_language(query)
    
    # Record the reasoning step for query analysis
    state["reasoning_steps"].append({"step": "query_analysis", "text": f"Analyzing query: '{query}' (detected language: {lang})"})
    
    # If query is in English, translate it to French for search since RAG data is in French
    search_query = translate_text([query])[0] if lang == "en" else query
    
    try:
        legal_terms = extract_legal_terms(search_query, lang)
        state["reasoning_steps"].append({"step": "legal_terms", "text": f"Extracted legal terms: {', '.join(legal_terms) if legal_terms else 'No specific legal terms identified'}"})
        
        # Encode query for vector search
        query_vector = embedding_model.encode(search_query, convert_to_numpy=True)
        
        # Get more candidates from vector search for better recall (8 instead of 5)
        distances, indices = index.search(np.array([query_vector]), 8)
        
        # Get BM25 scores for all documents
        bm25_scores = bm25.get_scores(tokenize_text(search_query))
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:8]
        
        # Advanced hybrid scoring
        combined_scores = {}
        
        # Vector search component with cosine similarity (closer to 1 is better)
        for idx, dist in zip(indices[0], distances[0]):
            # Convert L2 distance to a similarity score (1 / (1 + distance))
            similarity = 1 / (1 + dist)
            combined_scores[idx] = 0.5 * similarity  # Vector search weight: 50%
        
        # BM25 lexical search component
        max_bm25 = max(bm25_scores) if bm25_scores.any() else 1
        for idx in top_bm25_indices:
            normalized_bm25 = bm25_scores[idx] / max_bm25 if max_bm25 > 0 else 0
            combined_scores[idx] = combined_scores.get(idx, 0) + 0.5 * normalized_bm25  # BM25 weight: 50%
        
        # Bonus for documents containing exact legal terms
        if legal_terms:
            for idx in list(combined_scores.keys()):
                doc_text = legal_data[idx]["text"].lower()
                term_matches = sum(1 for term in legal_terms if term.lower() in doc_text)
                if term_matches > 0:
                    combined_scores[idx] += 0.2 * (term_matches / len(legal_terms))  # Up to 20% bonus
        
        # Apply recency bias if applicable (for legal documents with dates)
        for idx in list(combined_scores.keys()):
            update_date = legal_data[idx].get("metadata", {}).get("update_date")
            if update_date and update_date != "Unknown":
                try:
                    # Simple recency bias - newer documents get a small boost
                    year = int(re.search(r'\d{4}', update_date).group(0))
                    recency_score = min(0.1, (year - 2000) / 300)  # Small bonus (up to 10%)
                    combined_scores[idx] += recency_score
                except:
                    pass
        
        # Get top results
        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:5]  # Get top 5 for more context
        
        # Calculate confidence scores (normalized)
        max_score = max(combined_scores[idx] for idx in top_indices) if top_indices else 1
        confidence_scores = {idx: combined_scores[idx] / max_score for idx in top_indices}
        
        # Log search scores for debugging
        search_debug = "\n".join([
            f"Document {idx} (chunk: {legal_data[idx]['chunk_id']}): score={combined_scores[idx]:.4f}, confidence={confidence_scores[idx]:.2%}"
            for idx in top_indices
        ])
        logging.info(f"Search scores:\n{search_debug}")
        
        # Get the actual documents
        search_results = [doc_mappings[legal_data[idx]["chunk_id"]] for idx in top_indices]
        
        if not search_results:
            state["reasoning_steps"].append({"step": "search", "text": "No relevant documents found after hybrid search"})
            state["final_answer_en"] = "No specific regulation found in the provided data."
            state["final_answer_fr"] = "Aucune réglementation spécifique trouvée dans les données fournies."
            state["search_results"] = []
            state["sources"] = []
        else:
            # Create a detailed context with confidence scores
            context_entries = []
            for i, (idx, res) in enumerate(zip(top_indices, search_results)):
                confidence = confidence_scores[idx]
                relevance_label = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
                context_entries.append(
                    f"Document {i+1} [Relevance: {relevance_label}, Confidence: {confidence:.2%}]\n"
                    f"Article: {res['article']}\n"
                    f"Content: {res['content']}"
                )
            
            context = "\n\n".join(context_entries)
            state["reasoning_steps"].append({"step": "search", "text": f"Found {len(search_results)} relevant documents:\n\n{context}"})
            
            # Store search results with confidence scores
            for idx, res in zip(top_indices, search_results):
                res["confidence"] = confidence_scores[idx]
                
            state["search_results"] = search_results
            state["sources"] = [
                {
                    "article": res["article"],
                    "chunk_id": res["chunk_id"],
                    "text": res["text"],
                    "content": res["content"],
                    "update_date": res["update_date"],
                    "confidence": res.get("confidence", 0),
                    "highlight": True
                }
                for res in search_results
            ]
    except Exception as e:
        logging.error(f"Search failed: {e}")
        state["search_results"] = []
        state["sources"] = []
        state["reasoning_steps"].append({"step": "search", "text": f"Search failed due to an error: {str(e)}"})
    return state

def extract_legal_terms(query: str, lang: str) -> List[str]:
    """Extract potential legal terms from the query for better matching."""
    # This is a simplified version - in a production system, you might use a legal NER model
    common_legal_terms_fr = [
        "loi", "article", "code", "décret", "circulaire", "règlement", "jurisprudence", 
        "tribunal", "cour", "justice", "jugement", "contentieux", "procédure", "avocat",
        "responsabilité", "contrat", "obligation", "droit", "propriété", "civil", "pénal",
        "fiscal", "administratif", "commercial", "sociale", "travail", "constitution"
    ]
    
    common_legal_terms_en = [
        "law", "article", "code", "decree", "circular", "regulation", "jurisprudence",
        "court", "justice", "judgment", "litigation", "procedure", "lawyer", "attorney",
        "liability", "contract", "obligation", "right", "property", "civil", "criminal",
        "tax", "administrative", "commercial", "social", "labor", "constitution"
    ]
    
    terms = common_legal_terms_fr if lang == "fr" else common_legal_terms_en
    words = re.findall(r'\w+', query.lower())
    
    # Extract matching legal terms and noun phrases
    legal_terms = [word for word in words if word in terms]
    
    # Extract phrases (naive approach - in production, use proper NLP)
    words = query.lower().split()
    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i + 1]
        if any(term in phrase for term in terms):
            legal_terms.append(phrase)
    
    # Extract numbers that might be article numbers
    article_numbers = re.findall(r'\b(?:article\s+)?(\d+(?:\.\d+)?)\b', query.lower())
    legal_terms.extend(article_numbers)
    
    return list(set(legal_terms))

def generate_answer(state: ChatbotState) -> ChatbotState:
    start_time = time.time()
    if not state["search_results"]:
        state["thinking_time"] = time.time() - start_time
        return state
    query = state["query"]
    context = state["reasoning_steps"][-1]["text"]
    lang = detect_language(query)
    try:
        # Use the appropriate prompt template based on language
        prompt = SYSTEM_PROMPT_FR.format(context=context, query=query) if lang == "fr" else SYSTEM_PROMPT_EN.format(context=context, query=query)
        
        state["reasoning_steps"].append({"step": "prompt_generation", "text": "Generated prompt for LLM with relevant context and instructions"})
        
        # Create messages array for Groq API
        messages = [
            {"role": "system", "content": prompt}
        ]
        
        # Log that we're making the API call for debugging purposes
        logging.info(f"Sending request to Groq API with model: {config['groq_model']}")
        state["reasoning_steps"].append({"step": "llm_request", "text": f"Sending query to language model ({config['groq_model']})"})
        
        # Make the API call with appropriate parameters
        response = groq_chat_completion(
            messages=messages,
            model=config["groq_model"],
            temperature=config["temperature"],
            max_tokens=2048  
        )
        
        # Process the response
        state["reasoning_steps"].append({"step": "llm_response", "text": "Received response from language model"})
        
        # Extract thinking and answer sections using the <think> tag pattern for DeepSeek R1
        thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        
        if thinking_match:
            # DeepSeek model properly used the <think> tags
            thinking_text = thinking_match.group(1).strip()
            
            # Remove the thinking section to get the final answer
            answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            state["reasoning_steps"].append({"step": "thinking", "text": f"{thinking_text}"})
        else:
            # Fallback pattern matching for models that don't support the <think> tag format
            # Look for other potential thinking/reasoning markers
            potential_markers = [
                r'Thinking:|Réflexion:',
                r'Analysis:|Analyse:',
                r'Let me think:|Laissez-moi réfléchir:',
                r'Step by step:|Étape par étape:',
                r'My reasoning:|Mon raisonnement:'
            ]
            
            thinking_text = ""
            answer = response
            
            for marker in potential_markers:
                marker_match = re.search(f'({marker})(.*?)(\n\n|$)', response, re.DOTALL | re.IGNORECASE)
                if marker_match:
                    thinking_text = marker_match.group(2).strip()
                    # Remove this section from the answer
                    answer = response.replace(marker_match.group(0), '').strip()
                    break
            
            if not thinking_text:
                lines = response.split('\n')
                midpoint = len(lines) // 2
                thinking_text = '\n'.join(lines[:midpoint])
                answer = '\n'.join(lines[midpoint:])
            
            state["reasoning_steps"].append({"step": "thinking", "text": f"Extracted reasoning:\n\n{thinking_text}"})
        
        # Store final answer based on language
        if lang == "fr":
            state["final_answer_fr"] = answer
            
            # If reasoning is not in French, translate it
            if not is_french(thinking_text):
                thinking_text_fr = translate_text([thinking_text])[0]
                state["reasoning_steps"].append({"step": "reasoning_fr", "text": thinking_text_fr})
        else:
            state["final_answer_en"] = answer
            state["reasoning_steps"].append({"step": "reasoning_en", "text": thinking_text})
        
        # Log completion
        state["reasoning_steps"].append({"step": "completion", "text": "Answer generated successfully"})
        
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error generating answer: {error_message}")
        state["reasoning_steps"].append({"step": "error", "text": f"Error encountered during answer generation: {error_message}"})
        
        # Provide appropriate error messages based on language
        state["final_answer_en"] = "An error occurred while processing your request. Please try again or rephrase your question."
        state["final_answer_fr"] = "Une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer ou reformuler votre question."
    
    state["thinking_time"] = time.time() - start_time
    return state

def is_french(text: str) -> bool:
    """Determine if text is primarily in French."""
    try:
        return detect(text[:100]) == 'fr'  # Only check first 100 chars for efficiency
    except:
        # If detection fails, default to false
        return False

workflow = StateGraph(ChatbotState)
workflow.add_node("understand_query", understand_query)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_answer", generate_answer)
workflow.set_entry_point("understand_query")
workflow.add_edge("understand_query", "perform_search")
workflow.add_edge("perform_search", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400
    query = data["query"].strip()
    conversation_id = data.get("conversation_id")
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Track processing time for analytics
    processing_start = time.time()
    
    if conversation_id:
        conn = sqlite3.connect("conversations.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "Conversation not found"}), 404
        conn.close()
    else:
        title = generate_title(query)
        conversation_id = create_conversation(title)

    add_message(conversation_id, "user", query)
    
    # Log the beginning of query processing
    logging.info(f"Processing query: '{query}' for conversation {conversation_id}")

    initial_state = {
        "query": query,
        "reasoning_steps": [],
        "search_results": [],
        "final_answer_en": "",
        "final_answer_fr": "",
        "sources": [],
        "thinking_time": 0.0
    }
    
    try:
        # Run the agent workflow
        final_state = graph.invoke(initial_state)
        logging.info(f"Agent workflow completed in {time.time() - processing_start:.2f}s")
    except Exception as e:
        logging.error(f"Agent workflow failed: {str(e)}")
        # Fallback state with error message
        final_state = {
            "query": query,
            "reasoning_steps": [{"step": "error", "text": f"Processing error: {str(e)}"}],
            "search_results": [],
            "final_answer_en": "An error occurred while processing your request. Please try again later.",
            "final_answer_fr": "Une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer plus tard.",
            "sources": [],
            "thinking_time": 0.0
        }

    # Build HTML for assistant's response with enhanced reasoning display
    reasoning_block = ""
    if final_state["reasoning_steps"]:
        # Extract and organize thinking steps for better presentation
        thinking_steps = []
        for step in final_state["reasoning_steps"]:
            if step["step"] == "thinking" or step["step"] == "reasoning_en" or step["step"] == "reasoning_fr":
                thinking_steps.append({
                    "title": "Chain of Thought Reasoning",
                    "content": step["text"]
                })      
# Deduplicate thinking steps first
    seen = set()
    unique_steps = []
    for step in thinking_steps:
        content = step["content"].strip()
        if content not in seen:
            seen.add(content)
            unique_steps.append(step)

    thinking_steps = unique_steps

    # Then render them
    if thinking_steps:
        thinking_sections = "".join([
            f"""
            <div class="thinking-section">
            <h4>{step['title']}</h4>
            <div class="thinking-content">{step['content'].replace('\n', '<br>')}</div>
            </div>
            """
            for step in thinking_steps
        ])

        reasoning_block = f"""
        <details class="thinking-block">
        <summary>Show Chain of Thought Analysis</summary>
        <div class="thinking-container">
            {thinking_sections}
        </div>
        </details>
        """


        reasoning_block = f"""
        <details class="thinking-block">
        <summary>Show Chain of Thought Analysis</summary>
        <div class="thinking-container">
            {thinking_sections}
        </div>
        </details>
        """

        # Format the final answer based on language
        lang = detect_language(query)
        final_answer = final_state['final_answer_fr'] if lang == 'fr' else final_state['final_answer_en']
        
        # Apply proper formatting to the final answer
        final_answer_formatted = final_answer.replace('\n', '<br>')
        pattern = r"(.*?)(\*\*(Réponse|Answer):\*\*)"

        match = re.search(pattern, final_answer_formatted)
        if match:
            final_answer_formatted = match.group(0)

        final_answer_block = f"""
        <div class="final-answer">
        {final_answer_formatted}
        </div>
        """


        # Enhanced sources display with confidence scores
        sources_html = ""
        if final_state["sources"]:
            # Sort sources by confidence score, highest first
            sorted_sources = sorted(final_state["sources"], key=lambda x: x.get('confidence', 0), reverse=True)
            
            source_items = "".join([
                f"""
                <li class="source-item {get_confidence_class(src.get('confidence', 0))}">
                <div class="source-info">
    <span class="source-name">
    <strong>{src['article']} with Confidence of {src['confidence'] * 100:.2f}%</strong>
    </span>
                    <span class="confidence-badge">{format_confidence(src.get('confidence', 0))}</span>
                </div>
                <a href="#" class="source-link" onclick="showArticlePopup('{src['article']}', `{src['text'].replace('`', '\\`')}`, '{format_confidence(src.get('confidence', 0))}'); return false;">
                    View Content
                </a>
                </li>
                """
                for src in sorted_sources
            ])
            
            sources_html = f"""
            <details class="sources-toggle">
            <summary>Show Legal Sources ({len(sorted_sources)})</summary>
            <ul class="sources-list">
                {source_items}
            </ul>
            </details>
            """

        assistant_html = reasoning_block + final_answer_block + sources_html
        
        # Add CSS for enhanced display
        assistant_html = f"""
        <style>
        .thinking-block {{
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .thinking-container {{
            padding: 1rem;
            background-color: #f9f9f9;
        }}
        .thinking-section {{
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #eaeaea;
        }}
        .thinking-section h4 {{
            margin-top: 0;
            color: #2c5282;
        }}
        .final-answer {{
            padding: 1rem;
            background-color: #f0f7ff;
            border-left: 4px solid #2c5282;
            margin: 1rem 0;
            border-radius: 4px;
        }}
        .sources-toggle {{
            margin-top: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .sources-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .source-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #eaeaea;
        }}
        .source-item:last-child {{
            border-bottom: none;
        }}
        .source-info {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .confidence-badge {{
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        .high-confidence {{
            background-color: #c6f6d5;
        }}
        .medium-confidence {{
            background-color: #fefcbf;
        }}
        .low-confidence {{
            background-color: #fed7d7;
        }}
        .source-link {{
            padding: 0.25rem 0.75rem;
            background-color: #4299e1;
            color: white;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.875rem;
        }}
        .source-link:hover {{
            background-color: #3182ce;
        }}
        </style>
        {assistant_html}
        """
        
        add_message(conversation_id, "assistant", assistant_html)

        # Calculate total processing time
        total_processing_time = time.time() - processing_start
        
        # Prepare response metrics
        metrics = {
            "thinking_time": int(final_state["thinking_time"]),
            "total_processing_time": int(total_processing_time),
            "search_result_count": len(final_state["sources"]),
            "reasoning_steps_count": len([step for step in final_state["reasoning_steps"] if step["step"] in ("thinking", "reasoning_en", "reasoning_fr")]),
        }
        
        logging.info(f"Request processed in {total_processing_time:.2f}s (thinking: {final_state['thinking_time']:.2f}s)")
        
        return jsonify({
            "conversation_id": conversation_id,
            "title": title if not data.get("conversation_id") else None,
            "assistant_html": assistant_html,
            "sources": final_state["sources"],
            "metrics": metrics
        })

@app.route("/get_conversations", methods=["GET"])
def get_conversations():
    return jsonify(get_all_conversations())

@app.route("/get_conversation/<int:conversation_id>", methods=["GET"])
def get_conversation_route(conversation_id):
    conversation = get_conversation(conversation_id)
    if conversation:
        return jsonify(conversation)
    return jsonify({"error": "Conversation not found"}), 404

@app.route("/delete_conversation/<int:conversation_id>", methods=["DELETE"])
def delete_conversation_route(conversation_id):
    delete_conversation(conversation_id)
    return jsonify({"success": True})

@app.route("/search_conversations", methods=["POST"])
def search_conversations_route():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify(get_all_conversations())
    return jsonify(search_conversations(query))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Add JavaScript to handle article popup with enhanced features
@app.route('/js/article_popup.js')
def article_popup_js():
    js_content = """
function showArticlePopup(articleTitle, articleContent, confidenceLabel) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('articleModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'articleModal';
        modal.className = 'article-modal';
        document.body.appendChild(modal);
        
        // Add styles if not already in CSS
        if (!document.getElementById('article-modal-style')) {
            const style = document.createElement('style');
            style.id = 'article-modal-style';
            style.textContent = `
                .article-modal {
                    display: none;
                    position: fixed;
                    z-index: 1000;
                    left: 0;
                    top: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    background-color: rgba(0,0,0,0.4);
                    animation: fadeIn 0.3s ease-out;
                }
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                .article-modal-content {
                    background-color: #fefefe;
                    margin: 5% auto;
                    padding: 24px;
                    border: 1px solid #e2e8f0;
                    width: 90%;
                    max-width: 800px;
                    border-radius: 12px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    max-height: 80vh;
                    overflow-y: auto;
                    animation: slideIn 0.3s ease-out;
                }
                @keyframes slideIn {
                    from { transform: translateY(-20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                .article-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 16px;
                    padding-bottom: 16px;
                    border-bottom: 1px solid #e2e8f0;
                }
                .article-title-container {
                    flex: 1;
                }
                .article-close {
                    color: #a0aec0;
                    font-size: 24px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: color 0.2s;
                    background: none;
                    border: none;
                    padding: 0;
                    height: 24px;
                    line-height: 24px;
                    margin-left: 16px;
                }
                .article-close:hover {
                    color: #2d3748;
                }
                .article-title {
                    margin: 0 0 4px 0;
                    color: #2d3748;
                    font-size: 20px;
                    font-weight: 600;
                }
                .article-subtitle {
                    color: #718096;
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .confidence-badge {
                    display: inline-block;
                    padding: 2px 8px;
                    border-radius: 9999px;
                    font-size: 12px;
                    font-weight: 500;
                }
                .confidence-high {
                    background-color: #c6f6d5;
                    color: #22543d;
                }
                .confidence-medium {
                    background-color: #fefcbf;
                    color: #744210;
                }
                .confidence-low {
                    background-color: #fed7d7;
                    color: #822727;
                }
                .article-body {
                    white-space: pre-wrap;
                    line-height: 1.6;
                    color: #4a5568;
                    font-size: 16px;
                    padding: 8px 0;
                }
                .article-actions {
                    margin-top: 16px;
                    display: flex;
                    justify-content: flex-end;
                    gap: 8px;
                }
                .article-action-button {
                    padding: 8px 16px;
                    background-color: #edf2f7;
                    color: #2d3748;
                    border: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                .article-action-button:hover {
                    background-color: #e2e8f0;
                }
                .highlight {
                    background-color: #fef3c7;
                    padding: 2px 0;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Get confidence class
    let confidenceClass = "confidence-medium";
    if (confidenceLabel && confidenceLabel.includes("High")) {
        confidenceClass = "confidence-high";
    } else if (confidenceLabel && confidenceLabel.includes("Low")) {
        confidenceClass = "confidence-low";
    }
    
    // Format article content to highlight legal terms
    const formattedContent = formatLegalContent(articleContent);

    // Update modal content
    modal.innerHTML = `
        <div class="article-modal-content">
            <div class="article-header">
                <div class="article-title-container">
                    <h3 class="article-title">${articleTitle}</h3>
                    <div class="article-subtitle">
                        <span class="confidence-badge ${confidenceClass}">${confidenceLabel || 'Source'}</span>
                    </div>
                </div>
                <button class="article-close">&times;</button>
            </div>
            <div class="article-body">${formattedContent}</div>
            <div class="article-actions">
                <button class="article-action-button" onclick="copyToClipboard('${articleTitle}', this)">Copy Article Reference</button>
            </div>
        </div>
    `;

    // Show modal with animation
    modal.style.display = 'block';

    // Add close functionality
    const closeBtn = modal.querySelector('.article-close');
    closeBtn.onclick = function() {
        closeArticleModal();
    }

    // Close when clicking outside the modal
    window.onclick = function(event) {
        if (event.target == modal) {
            closeArticleModal();
        }
    }
    
    // Close when pressing Escape
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeArticleModal();
        }
    });
}

function closeArticleModal() {
    const modal = document.getElementById('articleModal');
    if (modal) {
        // Add closing animation
        modal.style.opacity = '0';
        modal.style.transform = 'translateY(-10px)';
        
        // Remove after animation completes
        setTimeout(() => {
            modal.style.display = 'none';
            modal.style.opacity = '1';
            modal.style.transform = 'translateY(0)';
        }, 200);
    }
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = "Copied!";
        button.style.backgroundColor = "#c6f6d5";
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.backgroundColor = "";
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
        button.textContent = "Failed to copy";
        button.style.backgroundColor = "#fed7d7";
        
        setTimeout(() => {
            button.textContent = "Copy Article Reference";
            button.style.backgroundColor = "";
        }, 2000);
    });
}

function formatLegalContent(content) {
    if (!content) return '';
    
    // Highlight article numbers
    let formatted = content.replace(/\\b(Article\\s+\\d+)\\b/g, '<strong>$1</strong>');
    
    // Highlight legal terms (simplified version)
    const legalTerms = [
        'loi', 'décret', 'circulaire', 'règlement', 'jurisprudence', 'tribunal', 'cour', 
        'justice', 'jugement', 'contentieux', 'procédure', 'avocat', 'responsabilité', 
        'contrat', 'obligation', 'droit', 'propriété', 'civil', 'pénal', 'fiscal', 
        'administratif', 'commercial', 'sociale', 'travail', 'constitution',
        'law', 'decree', 'circular', 'regulation', 'court', 'justice', 'judgment', 
        'litigation', 'procedure', 'lawyer', 'attorney', 'liability', 'contract', 
        'obligation', 'right', 'property', 'civil', 'criminal', 'tax', 'administrative', 
        'commercial', 'social', 'labor'
    ];
    
    // Create a regex pattern for all legal terms with word boundaries
    const pattern = new RegExp('\\\\b(' + legalTerms.join('|') + ')\\\\b', 'gi');
    formatted = formatted.replace(pattern, '<span class="highlight">$1</span>');
    
    return formatted;
}
    """
    return js_content, 200, {'Content-Type': 'application/javascript'}

# Helper functions for source display
def get_confidence_class(confidence):
    """Return the CSS class based on confidence score."""
    if confidence >= 0.8:
        return "high-confidence"
    elif confidence >= 0.5:
        return "medium-confidence"
    else:
        return "low-confidence"

def format_confidence(confidence):
    """Format confidence score for display."""
    if confidence >= 0.8:
        return "High Relevance"
    elif confidence >= 0.5:
        return "Medium Relevance"
    else:
        return "Low Relevance"

if __name__ == "__main__":
    init_db()  # Initialize database
    initialize_data()
    app.run(debug=False, host="0.0.0.0", port=5000)
