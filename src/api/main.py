from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import logging
import re
from typing import List, Dict
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import itertools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CPU usage and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely disable CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
torch.backends.cudnn.enabled = False
torch.backends.cuda.enable_mem_eager_sdp = False
torch.cuda.is_available = lambda: False  # Force CPU-only mode
torch.cuda.device_count = lambda: 0  # Force CPU-only mode

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model = None
vector_store = None
articles = []
policies = ""

def clean_text(text: str) -> str:
    """Clean text by removing HTML entities and extra whitespace."""
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs and clean them."""
    # Split on newlines and filter out empty paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    # Clean each paragraph
    return [clean_text(p) for p in paragraphs]

def extract_specific_details(text: str, question: str) -> str:
    """Extract specific details like numbers and percentages from text based on the question."""
    # Look for dollar amounts
    if "amount" in question.lower() or "raised" in question.lower():
        dollar_matches = re.findall(r'\$[\d,]+', text)
        if dollar_matches:
            return f"The amount raised was {dollar_matches[0]}"
    
    # Look for percentages
    if "percentage" in question.lower() or "increase" in question.lower():
        percent_matches = re.findall(r'(\d+)\s*per\s*cent', text, re.IGNORECASE)
        if percent_matches:
            return f"The expected increase is {percent_matches[0]}%"
    
    return text

def extract_full_policy_section(policies: str, heading: str) -> str:
    """Extract the full policy section given a heading from the policies text."""
    lines = policies.split('\n')
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == heading.strip().lower():
            start_idx = i
            break
    if start_idx is None:
        return heading  # fallback: just return the heading
    # Find the next heading or end of file
    section_lines = [lines[start_idx]]
    for line in itertools.islice(lines, start_idx + 1, None):
        if line.strip().startswith('CBC Editorial Guidelines:') and line.strip() != heading.strip():
            break
        section_lines.append(line)
    return '\n'.join(section_lines).strip()

def split_policy_sections(policies: str):
    """Split the policies text into sections, each starting with a heading."""
    sections = []
    current_section = []
    current_title = None
    
    for line in policies.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('CBC Editorial Guidelines:'):
            # Save previous section if exists
            if current_title and current_section:
                sections.append({
                    'title': current_title,
                    'text': '\n'.join(current_section)
                })
            # Start new section
            current_title = line
            current_section = [line]
        elif current_title:
            current_section.append(line)
    
    # Add the last section
    if current_title and current_section:
        sections.append({
            'title': current_title,
            'text': '\n'.join(current_section)
        })
    
    return sections

def initialize_components():
    """Initialize all required components for the API."""
    global model, tokenizer, flan_model, vector_store, articles, policies, policy_sections
    
    try:
        # Load SentenceTransformer model
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        logger.info("SentenceTransformer model loaded successfully")
        
        # Load Flan-T5 model and tokenizer
        logger.info("Loading Flan-T5 model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
        flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            local_files_only=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        logger.info("Flan-T5 model and tokenizer loaded successfully")
        
        # Load data
        logger.info("Loading data...")
        data_dir = Path("data")
        logger.info(f"Data directory: {data_dir.absolute()}")
        
        # Load articles
        logger.info("Loading articles...")
        with open(data_dir / "articles.json", "r") as f:
            articles = json.load(f)
        logger.info(f"Successfully loaded {len(articles)} articles")
        
        # Load policies
        logger.info("Loading policies...")
        with open(data_dir / "policies.txt", "r") as f:
            policies = f.read()
        logger.info("Policies loaded")
        
        # Split policies into sections
        policy_sections = split_policy_sections(policies)
        logger.info(f"Split policies into {len(policy_sections)} sections")
        
        # Create vector store
        logger.info("Creating vector store...")
        texts = []
        sources = []
        
        # Add articles (paragraphs)
        for article in articles:
            paragraphs = split_into_paragraphs(article["body"])
            texts.extend(paragraphs)
            sources.extend([{
                "type": "article",
                "title": article["content_headline"]
            } for _ in paragraphs])
        
        # Add policy sections
        for section in policy_sections:
            texts.append(section["text"])
            sources.append({
                "type": "policy",
                "title": section["title"]
            })
        
        embeddings = model.encode(texts)
        logger.info(f"Created embeddings of shape: {embeddings.shape}")
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        vector_store = faiss.IndexFlatL2(dimension)
        vector_store.add(embeddings.astype("float32"))
        logger.info("Vector store created")
        
        # Store sources and texts for later use
        app.state.sources = sources
        app.state.texts = texts
        app.state.policy_sections = policy_sections  # Store policy sections for reference
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "ok", "message": "CBC Editorial Assistant API is running"}

@app.post("/api/qa")
async def qa_endpoint(request: Question):
    try:
        question = request.question
        logger.info(f"Received question: {question}")
        
        # Encode question
        question_embedding = model.encode([question])
        
        # Search for similar content
        logger.info("Searching for similar content...")
        D, I = vector_store.search(question_embedding.astype("float32"), k=5)
        
        # Get relevant text
        logger.info("Getting relevant text...")
        citations = []
        seen_sources = set()
        policy_section = None
        
        # First try to find a relevant policy section
        for idx in I[0]:
            source = app.state.sources[idx]
            if source["type"] == "policy":
                # Find the full policy section
                for section in app.state.policy_sections:
                    if section["title"] == source["title"]:
                        policy_section = {
                            "source": section["title"],
                            "text": section["text"]
                        }
                        break
                if policy_section:
                    break
        
        # If no policy section found, look for relevant articles
        if not policy_section:
            for idx in I[0]:
                source = app.state.sources[idx]
                if source["type"] == "article":
                    source_key = f"{source['type']}:{source['title']}"
                    if source_key in seen_sources:
                        continue
                    seen_sources.add(source_key)
                    
                    text = app.state.texts[idx]
                    relevance_score = model.encode([text])[0] @ question_embedding[0]
                    if relevance_score > 0.3:
                        citations.append({
                            "source": f"CBC Article: {source['title']}",
                            "text": text
                        })
                    if len(citations) >= 3:
                        break
        
        if not citations and not policy_section:
            logger.warning("No relevant information found")
            raise HTTPException(status_code=404, detail="No relevant information found")
        
        # Create prompt for Flan-T5
        if policy_section:
            prompt = f"""You are a CBC editorial assistant. Answer the following question about CBC's editorial guidelines.
            Your answer MUST follow this exact format:
            
            SUMMARY:
            [Write a 1-2 sentence summary of the key guidelines]
            
            KEY REQUIREMENTS:
            • [First requirement]
            • [Second requirement]
            • [Third requirement]
            
            ADDITIONAL NOTES:
            [Any important exceptions or special cases]
            
            Policy Guidelines:
            {policy_section['text']}

            Question: {question}

            Answer:"""
        else:
            context = "\n".join([f"Source: {c['source']}\nText: {c['text']}" for c in citations])
            prompt = f"""You are a CBC editorial assistant. Answer the following question about CBC's content.
            Your answer MUST follow this exact format:
            
            SUMMARY:
            [Write a 1-2 sentence summary of the key information]
            
            KEY DETAILS:
            • [First important detail]
            • [Second important detail]
            • [Third important detail]
            
            CONTEXT:
            [Any relevant background information]
            
            Context: {context}

            Question: {question}

            Answer:"""
        
        # Log the prompt for debugging
        logger.info(f"Prompt sent to model:\n{prompt}")
        
        # Generate answer using Flan-T5
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = flan_model.generate(
            inputs["input_ids"],
            max_length=500,
            num_beams=4,
            temperature=0.7,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer
        answer = answer.strip()
        if not answer or len(answer) < 10:
            answer = "I apologize, but I couldn't generate a proper answer. Please try rephrasing your question."
        
        return {
            "answer": answer,
            "citations": [policy_section] if policy_section else citations
        }
    except Exception as e:
        logger.error(f"Error in QA endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    initialize_components()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 