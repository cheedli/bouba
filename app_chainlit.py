# app.py
import os
import re
import json
import time
import yaml
import logging
import sqlite3
import numpy as np
from typing import Dict, List, TypedDict, Optional
from datetime import datetime

# Vector Database
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Embedding & Language Processing
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Model Serving with vLLM
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

# UI Framework
import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.element import Element

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
    "collection_name": "legal_documents",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "vector_dim": 384,
    "vllm_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "embedding_batch_size": 8,
    "search_top_k": 5,
    "semantic_search_weight": 0.5,
    "lexical_search_weight": 0.5,
    "legal_terms_bonus_weight": 0.2,
    "temperature": 0.1,  # Low temperature for legal precision
    "milvus_host": "localhost",
    "milvus_port": "19530"
}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        loaded_config = yaml.safe_load(f) or {}
    config = DEFAULT_CONFIG.copy()
    config.update(loaded_config)
else:
    config = DEFAULT_CONFIG
    logging.warning(f"Config file {CONFIG_FILE} not found, using defaults.")

# Initialize global variables
embedding_model = SentenceTransformer(config["embedding_model"])
bm25 = None
legal_data = None
milvus_collection = None

# Initialize translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name)

from transformers import BitsAndBytesConfig

# Initialize BitsAndBytes config for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,        # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Best tradeoff between speed and accuracy
    bnb_4bit_compute_dtype="float16"  # Use half precision for speed
)

# Initialize vLLM engine with quantization support
engine_args = AsyncEngineArgs(
    model=config["vllm_model"],
    dtype="half",  # or "float16"
    gpu_memory_utilization=0.8,
    max_model_len=8192,
    quantization_config=bnb_config  # Add this line to use 4-bit quantization
)

llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

def translate_text(text_list, batch_size=5):
    """Translate a batch of English sentences into French."""
    translated_texts = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        translation = translator_model.generate(**tokens)
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
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

# Title Generation using vLLM
async def generate_title(query: str) -> str:
    """Generate a conversation title using vLLM."""
    prompt = f"Generate a very short and relevant title for a conversation about: {query}. Detect the language of the query and respond in the same language. Return only the title. No introductions, no formatting, no extra text."
    try:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=20)
        request_id = random_uuid()
        result = await llm_engine.generate(prompt, sampling_params, request_id)
        
        # Extract generated text
        generated_text = result.outputs[0].text.strip()
        
        # Remove any quotation marks that might be included
        title = generated_text.strip('"\'')
        
        # Limit title length
        if len(title) > 50:
            title = title[:47] + "..."
            
        return title
    except Exception as e:
        logging.error(f"Error generating title: {e}")
        return "Untitled Conversation"

def tokenize_text(text: str, lang: str = "fr") -> List[str]:
    """Tokenize text into words."""
    return re.findall(r'\w+', text.lower())

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

# Milvus Collection Setup
def setup_milvus():
    """Set up Milvus connection and create collection if it doesn't exist."""
    global milvus_collection
    
    # Connect to Milvus
    connections.connect(
        alias="default", 
        host=config["milvus_host"], 
        port=config["milvus_port"]
    )
    
    collection_name = config["collection_name"]
    
    # Check if collection exists
    if utility.has_collection(collection_name):
        milvus_collection = Collection(collection_name)
        milvus_collection.load()
        logging.info(f"Loaded existing Milvus collection: {collection_name}")
        return milvus_collection
    
    # Define collection schema
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="article", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="article_number", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="update_date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="is_sub_chunk", dtype=DataType.BOOL),
        FieldSchema(name="parent_chunk", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config["vector_dim"])
    ]
    
    schema = CollectionSchema(fields=fields, description="Legal document collection")
    
    # Create collection
    milvus_collection = Collection(collection_name, schema)
    
    # Create index
    index_params = {
        "metric_type": "IP",  # Inner product (cosine similarity)
        "index_type": "HNSW",  # Hierarchical Navigable Small World for fast retrieval
        "params": {"M": 16, "efConstruction": 200}
    }
    
    milvus_collection.create_index("embedding", index_params)
    milvus_collection.load()
    
    logging.info(f"Created new Milvus collection: {collection_name}")
    return milvus_collection

def initialize_data():
    """Initialize vector database and BM25 index."""
    global milvus_collection, bm25, legal_data
    
    try:
        # Check if data file exists
        if not os.path.exists(config["data_file"]):
            raise FileNotFoundError(f"Data file {config['data_file']} not found.")
        
        # Setup Milvus connection
        milvus_collection = setup_milvus()
        
        # Check if collection already has data
        if milvus_collection.num_entities > 0:
            logging.info(f"Milvus collection already contains {milvus_collection.num_entities} entities.")
            
            # Load legal data for BM25
            legal_data = parse_legal_data(config["data_file"])
            
            # Create BM25 corpus
            bm25_corpus = []
            for entry in legal_data:
                bm25_corpus.append(tokenize_text(entry["text"], entry["metadata"]["language"].lower()))
            
            # Initialize BM25
            bm25 = BM25Okapi(bm25_corpus)
            logging.info(f"Loaded {len(legal_data)} legal document chunks")
            
        else:
            # Parse legal data
            legal_data = parse_legal_data(config["data_file"])
            logging.info(f"Parsed {len(legal_data)} legal document chunks")
            
            # Prepare BM25
            bm25_corpus = []
            
            # Process in batches to avoid memory issues with large datasets
            batch_size = config.get("embedding_batch_size", 8)
            total_chunks = len(legal_data)
            
            # Insert data into Milvus
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch = legal_data[i:batch_end]
                
                logging.info(f"Processing batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}: chunks {i} to {batch_end-1}")
                
                # Prepare texts and data for batch insertion
                texts = [entry["text"] for entry in batch]
                chunk_ids = [entry["chunk_id"] for entry in batch]
                articles = [entry["article"] for entry in batch]
                article_numbers = [entry.get("article_number", "Unknown") for entry in batch]
                titles = [entry.get("title", "") for entry in batch]
                contents = [entry.get("content", entry["text"]) for entry in batch]
                languages = [entry["metadata"]["language"] for entry in batch]
                update_dates = [entry["metadata"]["update_date"] for entry in batch]
                is_sub_chunks = [entry.get("is_sub_chunk", False) for entry in batch]
                parent_chunks = [entry.get("parent_chunk", "") for entry in batch]
                
                # Encode batch
                batch_embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                
                # Add to BM25 corpus
                for j, entry in enumerate(batch):
                    bm25_corpus.append(tokenize_text(entry["text"], entry["metadata"]["language"].lower()))
                
                # Insert into Milvus
                milvus_collection.insert([
                    chunk_ids, 
                    articles, 
                    article_numbers, 
                    titles, 
                    texts, 
                    contents, 
                    languages, 
                    update_dates, 
                    is_sub_chunks, 
                    parent_chunks, 
                    batch_embeddings.tolist()
                ])
            
            # Create index if it doesn't exist
            if not milvus_collection.has_index():
                index_params = {
                    "metric_type": "IP",  # Inner product (cosine)
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200}
                }
                milvus_collection.create_index("embedding", index_params)
            
            # Initialize BM25
            bm25 = BM25Okapi(bm25_corpus)
            
            logging.info("Successfully indexed all documents in Milvus")
    except Exception as e:
        logging.error(f"Failed to initialize data: {e}")
        raise

def detect_language(query: str) -> str:
    """Detect language of query text."""
    try:
        lang = detect(query)
        return "fr" if lang.startswith("fr") else "en"
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return "en"

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

def is_french(text: str) -> bool:
    """Determine if text is primarily in French."""
    try:
        return detect(text[:100]) == 'fr'  # Only check first 100 chars for efficiency
    except:
        # If detection fails, default to false
        return False

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
1. Commencez par une réflexion rigoureuse à l'intérieur des balises <think>.
2. Dans les balises <think> :
   - Interprétez avec précision la demande de l'utilisateur
   - Identifiez les principes et cadres juridiques applicables
   - Analysez en profondeur les documents juridiques fournis
   - Soulignez les articles les plus pertinents et justifiez leur choix
   - Prenez en compte les nuances, exceptions ou conflits éventuels
   - Basez-vous uniquement sur les textes disponibles
   - N'inventez ni n'extrapolez aucune information juridique

3. Ensuite, fournissez une réponse structurée et claire :
   - Répondez de manière directe et pertinente à la question
   - Citez les dispositions légales exactes (numéro et formulation) si nécessaire
   - Utilisez une liste numérotée avec des **titres en gras** pour plus de clarté
   - Restez rigoureux sur le plan juridique et évitez toute spéculation
   - Adoptez un ton formel, adapté aux professionnels du droit/finance

4. Pour les échanges informels, soyez courtois et synthétique.
5. Si on vous demande qui vous êtes, répondez : "Je suis Combot, votre assistant juridique pour les lois tunisiennes."
6. Si aucune information pertinente n'est trouvée, indiquez : "Aucune réglementation spécifique trouvée dans les données fournies."
</instructions>

Les documents ci-dessous sont classés par ordre de pertinence. Accordez une attention particulière à ceux ayant un score de pertinence élevé, sans négliger les autres.

TEXTES JURIDIQUES :
{context}

QUESTION DE L'UTILISATEUR : {query}
"""

# Chainlit Integration
@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    init_db()
    initialize_data()
    
    # Set conversation metadata
    conversation_id = create_conversation("New Conversation")
    cl.user_session.set("conversation_id", conversation_id)
    
    # Display welcome message
    await cl.Message(
        content="👋 Welcome to the Legal Assistant! I specialize in Tunisian law. How can I help you today?",
        author="System"
    ).send()

async def process_query(query: str):
    """Process user query and return response with sources."""
    # Start timing
    processing_start = time.time()
    
    # Get conversation ID
    conversation_id = cl.user_session.get("conversation_id")
    
    # Add user message to database
    add_message(conversation_id, "user", query)
    
    # Detect language
    lang = detect_language(query)
    
    # If query is in English, translate it to French for search since RAG data is in French
    search_query = translate_text([query])[0] if lang == "en" else query
    
    # Extract legal terms for better search
    legal_terms = extract_legal_terms(search_query, lang)
    
    # Prepare thinking steps message
    thinking_msg = cl.Message(content="", author="Assistant")
    await thinking_msg.send()
    
    # Update with thinking indicator
    await thinking_msg.stream_token("🧠 Analyzing your query...")
    
    try:
        # Vector search with Milvus
        query_vector = embedding_model.encode(search_query, convert_to_numpy=True)
        
        # Search in Milvus with inner product (cosine similarity)
        search_params = {
            "metric_type": "IP",
            "params": {"ef": 64}  # Higher ef means more accurate but slower search
        }
        
        # Get more candidates for better recall
        milvus_results = milvus_collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=8,
            output_fields=["chunk_id", "article", "article_number", "title", "text", "content", 
                          "language", "update_date", "is_sub_chunk", "parent_chunk"]
        )
        
        # BM25 search for lexical matching
        bm25_scores = bm25.get_scores(tokenize_text(search_query))
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:8]
        
        # Combine results with hybrid scoring
        combined_scores = {}
        
        # Vector search results
        for hit in milvus_results[0]:
            chunk_id = hit.entity.get("chunk_id")
            similarity = hit.distance  # cosine similarity score
            combined_scores[chunk_id] = config["semantic_search_weight"] * similarity
        
        # Lexical search results
        max_bm25 = max(bm25_scores) if bm25_scores.any() else 1
        for idx in top_bm25_indices:
            chunk_id = legal_data[idx]["chunk_id"]
            normalized_bm25 = bm25_scores[idx] / max_bm25 if max_bm25 > 0 else 0
            combined_scores[chunk_id] = combined_scores.get(chunk_id, 0) + config["lexical_search_weight"] * normalized_bm25
        
        # Bonus for documents containing exact legal terms
        if legal_terms:
            # Get all chunk IDs
            all_chunk_ids = list(combined_scores.keys())
            
            # Query to get full text content
            query_expr = f"chunk_id in [" + ",".join([f"'{chunk_id}'" for chunk_id in all_chunk_ids]) + "]"
            results = milvus_collection.query(
                expr=query_expr,
                output_fields=["chunk_id", "text"]
            )
            
            # Apply term bonus
            for result in results:
                chunk_id = result["chunk_id"]
                doc_text = result["text"].lower()
                term_matches = sum(1 for term in legal_terms if term.lower() in doc_text)
                if term_matches > 0:
                    combined_scores[chunk_id] += config["legal_terms_bonus_weight"] * (term_matches / len(legal_terms))
        
        # Get top results
        top_chunk_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:config["search_top_k"]]
        
        # Calculate confidence scores (normalized)
        max_score = max(combined_scores[chunk_id] for chunk_id in top_chunk_ids) if top_chunk_ids else 1
        confidence_scores = {chunk_id: combined_scores[chunk_id] / max_score for chunk_id in top_chunk_ids}
        
        # Query Milvus to get full documents for top results
        if top_chunk_ids:
            query_expr = f"chunk_id in [" + ",".join([f"'{chunk_id}'" for chunk_id in top_chunk_ids]) + "]"
            search_results = milvus_collection.query(
                expr=query_expr,
                output_fields=["chunk_id", "article", "article_number", "title", "content", "update_date", "is_sub_chunk", "parent_chunk"]
            )
            
            # Add confidence scores to results
            for result in search_results:
                chunk_id = result["chunk_id"]
                result["confidence"] = confidence_scores[chunk_id]
            
            # Sort by confidence score
            search_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)
        else:
            search_results = []
        
        # Update the thinking message
        await thinking_msg.update(content="🔍 Found relevant legal documents. Analyzing content...")
        
        if not search_results:
            # No relevant documents found
            if lang == "fr":
                response = "Aucune réglementation spécifique trouvée dans les données fournies."
            else:
                response = "No specific regulation found in the provided data."
            
            # Add response to database
            add_message(conversation_id, "assistant", response)
            
            # Update the thinking message with final response
            await thinking_msg.update(content=response)
            
            return None
        
        # Create a detailed context with confidence scores for the LLM
        context_entries = []
        for i, res in enumerate(search_results):
            confidence = res["confidence"]
            relevance_label = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
            context_entries.append(
                f"Document {i+1} [Relevance: {relevance_label}, Confidence: {confidence:.2%}]\n"
                f"Article: {res['article']}\n"
                f"Content: {res['content']}"
            )
        
        context = "\n\n".join(context_entries)
        
        # Use the appropriate prompt template based on language
        prompt = SYSTEM_PROMPT_FR.format(context=context, query=query) if lang == "fr" else SYSTEM_PROMPT_EN.format(context=context, query=query)
        
        # Generate response with vLLM
        await thinking_msg.update(content="💭 Formulating response based on legal texts...")
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=0.95,
            max_tokens=2048,
        )
        
        # Generate response
        request_id = random_uuid()
        result = await llm_engine.generate(prompt, sampling_params, request_id)
        response = result.outputs[0].text.strip()
        
        # Extract thinking and answer sections
        thinking_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        
        if thinking_match:
            # Model properly used the <think> tags
            thinking_text = thinking_match.group(1).strip()
            
            # Remove the thinking section to get the final answer
            answer = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
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
        
        # Add response to database
        add_message(conversation_id, "assistant", answer)
        
        # Create source elements for display
        sources = []
        for res in search_results:
            confidence = res["confidence"]
            relevance_label = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
            
            # Create source element
            source = cl.Text(
                name=f"{res['article']} ({relevance_label} Relevance)",
                content=res["content"],
                display="inline"
            )
            sources.append(source)
        
        # Update thinking message with final answer
        await thinking_msg.update(content=answer, elements=sources)
        
        # Display thinking process in a separate message if available
        if thinking_text:
            thinking_element = cl.Text(
                name="Chain of Thought Analysis",
                content=thinking_text,
                display="side"
            )
            await cl.Message(
                content="Here's my reasoning process:",
                elements=[thinking_element],
                author="Thinking Process"
            ).send()
        
        # Calculate total processing time
        total_processing_time = time.time() - processing_start
        logging.info(f"Query processed in {total_processing_time:.2f}s")
        
        # Return the answer
        return answer
        
    except Exception as e:
        error_message = str(e)
        logging.error(f"Error processing query: {error_message}")
        
        # Provide appropriate error messages based on language
        if lang == "fr":
            error_response = "Une erreur s'est produite lors du traitement de votre demande. Veuillez réessayer ou reformuler votre question."
        else:
            error_response = "An error occurred while processing your request. Please try again or rephrase your question."
        
        # Add error response to database
        add_message(conversation_id, "assistant", error_response)
        
        # Update the thinking message with error
        await thinking_msg.update(content=error_response)
        
        return error_response

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""
    query = message.content
    if not query.strip():
        await cl.Message(content="Please provide a valid query.").send()
        return
    
    # Process the query
    await process_query(query)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> bool:
    """Simple authentication callback."""
    # In a real application, this would check against a database
    # For demo purposes, we'll use a simple check
    valid_credentials = {
        "admin": "password123",
        "legal": "legal123"
    }
    return username in valid_credentials and password == valid_credentials[username]

# File upload handler
@cl.on_file
async def on_file(file: cl.File):
    """Handle file uploads."""
    conversation_id = cl.user_session.get("conversation_id")
    
    await cl.Message(content=f"Received file: {file.name}. Processing...").send()
    
    # Get the file content
    content = await file.get_bytes()
    
    # Create a temporary file
    temp_filename = f"temp_{file.name}"
    with open(temp_filename, "wb") as f:
        f.write(content)
    
    try:
        if file.name.endswith(".txt"):
            # Process text file
            with open(temp_filename, "r", encoding="utf-8") as f:
                file_text = f.read()
            
            # Parse legal data
            if len(file_text.strip()) > 0:
                # For simplicity, we'll just add this as a user message
                add_message(conversation_id, "user", f"File content: {file.name}\n\n{file_text[:500]}...")
                
                await cl.Message(
                    content=f"Processed text file. File contains {len(file_text)} characters. Would you like me to analyze it?",
                    author="System"
                ).send()
            else:
                await cl.Message(content="The uploaded file appears to be empty.", author="System").send()
                
        elif file.name.endswith((".pdf", ".docx")):
            # In a real application, add PDF/DOCX processing logic here
            await cl.Message(
                content="File received. PDF/DOCX processing would be implemented in a production version.",
                author="System"
            ).send()
        else:
            await cl.Message(
                content="Unsupported file type. Please upload .txt, .pdf, or .docx files.",
                author="System"
            ).send()
            
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        await cl.Message(content=f"Error processing file: {str(e)}", author="System").send()
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates."""
    # You can implement custom settings like language preference, search depth, etc.
    cl.user_session.set("settings", settings)
    
    # Acknowledge the settings change
    await cl.Message(content=f"Settings updated: {settings}", author="System").send()

@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end."""
    # Save conversation history or perform cleanup
    pass

# Utility function to format conversation title
async def update_conversation_title(conversation_id: int, query: str):
    """Update the conversation title based on the first query."""
    title = await generate_title(query)
    
    # Update title in database
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
    conn.commit()
    conn.close()
    
    # Return the new title
    return title

# Main entry point
if __name__ == "__main__":
    # This won't be called when using chainlit, but useful for debugging
    # initialize_data will be called by on_chat_start instead
    init_db()
    initialize_data()