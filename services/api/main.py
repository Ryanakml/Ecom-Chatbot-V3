import logging

import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

import tempfile
import json

from typing import Dict, Any
import uuid
from fastapi import FastAPI, HTTPException

import boto3
import faiss
import mlflow
import numpy as np
import pandas as pd
import spacy
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from elasticsearch import Elasticsearch

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Service Configurations ---
MLFLOW_TRACKING_URI = "http://localhost:5001"
MINIO_ENDPOINT = 'localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
RETRIEVER_BUCKET = 'retriever-artifacts'
ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'faq-keyword-index'


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Ecommerce Chatbot API",
    description="Serves Intent Classification, NER, and Retrieval models.",
    version="1.0.0"
)

# --- Global State for Models & Artifacts ---
# This dictionary will be populated at startup
models = {}

# --- Pydantic Models for API I/O ---
class TextInput(BaseModel):
    text: str

class IntentResponse(BaseModel):
    intent: str
    confidence: float

class Slot(BaseModel):
    entity: str
    value: str
    start: int
    end: int

class NERResponse(BaseModel):
    slots: list[Slot]

class RetrievalResult(BaseModel):
    doc_id: int
    text: str
    score: float

class RetrievalResponse(BaseModel):
    dense_results: list[RetrievalResult]
    sparse_results: list[RetrievalResult]

class GenerateResponse(BaseModel):
    response: str

class ChatInput(BaseModel):
    text: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response_text: str
    session_id: str
    status: str # e.g., 'awaiting_user_input', 'escalated'

# --- DIALOG MANAGER ---
class DialogManager:
    """Orchestrates the conversation flow."""
    def __init__(self, models_dict: dict):
        self.models = models_dict
        self.conversations: Dict[str, Any] = {} # In-memory state storage

    def _get_or_create_session(self, session_id: str | None) -> str:
        if not session_id or session_id not in self.conversations:
            session_id = str(uuid.uuid4())
            self.conversations[session_id] = {
                "history": [],
                "state": "awaiting_user_input",
                "state_info": {}
            }
        return session_id

    async def _predict_intent(self, text: str) -> IntentResponse:
        vectorizer = self.models['intent_vectorizer']
        model = self.models['intent_model']
        labels = self.models['intent_labels']
        
        transformed_text = vectorizer.transform([text]).toarray()
        tensor_text = torch.tensor(transformed_text, dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            outputs = model(tensor_text)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        intent = labels[predicted_idx.item()]
        return IntentResponse(intent=intent, confidence=confidence.item())

    async def _extract_slots(self, text: str) -> NERResponse:
        ner_model = self.models['ner_model']
        doc = ner_model(text)
        slots = [Slot(entity=ent.label_, value=ent.text) for ent in doc.ents]
        return NERResponse(slots=slots)

    async def _retrieve_docs(self, text: str) -> str:
        embedding_model = self.models['embedding_model']
        faiss_index = self.models['faiss_index']
        docs_df = self.models['retriever_docs']
        
        query_embedding = embedding_model.encode([text]).astype('float32')
        _, indices = faiss_index.search(query_embedding, k=1)
        
        best_match_text = docs_df.iloc[indices[0][0]]['text']
        return f"I found this information that might help: '{best_match_text}'"

    async def handle_message(self, request: ChatInput) -> ChatResponse:
        session_id = self._get_or_create_session(request.session_id)
        session = self.conversations[session_id]
        
        # --- Multi-turn logic based on state ---
        if session["state"] == "awaiting_order_id":
            # Simple check for a potential order ID
            if any(char.isdigit() for char in request.text):
                order_id = request.text
                response_text = f"Thank you! Looking up the status for order ID: {order_id}..."
                session["state"] = "awaiting_user_input"
            else:
                response_text = "That doesn't look like an order ID. Please provide a valid order ID."
            
            return ChatResponse(response_text=response_text, session_id=session_id, status=session["state"])

        # --- Standard NLU pipeline ---
        intent_response = await self._predict_intent(request.text)
        intent = intent_response.intent
        confidence = intent_response.confidence

        response_text = "I'm not sure how to handle that."
        session["state"] = "awaiting_user_input"

        # --- Intent-based routing ---
        if confidence < 0.6:
            intent = "out_of_scope" # Force fallback if confidence is too low

        if intent == 'greet':
            response_text = "Hello! How can I help you with your order today?"
        
        elif intent == 'track_order':
            slots_response = await self._extract_slots(request.text)
            order_id_slot = next((s for s in slots_response.slots if s.entity == 'CARDINAL'), None) # Simplified entity check

            if order_id_slot:
                response_text = f"Sure, I am looking up the status for order {order_id_slot.value}."
            else:
                response_text = "I can help with that. What is your order ID?"
                session["state"] = "awaiting_order_id"
        
        elif intent in ['change_order', 'cancel_order', 'get_refund']:
            response_text = await self._retrieve_docs(request.text)

        elif intent == 'out_of_scope':
            response_text = "I'm sorry, I can't handle that request. I am escalating you to a human agent."
            session["state"] = "escalated"

        session["history"].append({"user": request.text, "bot": response_text})
        return ChatResponse(response_text=response_text, session_id=session_id, status=session["state"])

# --- Model Loading on Startup ---
@app.on_event("startup")
async def load_models_and_artifacts():
    """
    Loads all required models and artifacts from MLflow and MinIO
    into the global 'models' dictionary when the application starts.
    """
    logging.info("--- Starting application and loading models ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # 1. Load Intent Classification Model from MLflow
    try:
        logging.info("Loading Intent model from MLflow...")
        intent_model_uri = "models:/intent-classification/Production"
        models['intent_model'] = mlflow.pytorch.load_model(intent_model_uri)

        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("intent-classification", stages=["Production"])[0]
        run_id = latest_version.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = client.download_artifacts(run_id, "vocabulary.json", tmpdir)
            labels_path = client.download_artifacts(run_id, "label_encoder_classes.json", tmpdir)

            with open(vocab_path) as f:
                raw_vocab = json.load(f)
            with open(labels_path) as f:
                labels = json.load(f)

        # Debug log: lihat isi vocab
        logging.info(f"Sample vocab: {list(raw_vocab.items())[:5]}")

        # Pastikan format vocab sesuai (word -> index)
        if all(k.isdigit() for k in raw_vocab.keys()):
            vocab = {v: int(k) for k, v in raw_vocab.items()}
            logging.info("Detected index→word format, converted to word→index")
        else:
            vocab = raw_vocab

        # Fix: Ensure vocabulary has consecutive indices starting from 0
        vocab_items = list(vocab.items())
        vocab = {word: idx for idx, (word, _) in enumerate(vocab_items)}
        logging.info(f"Fixed vocab size: {len(vocab)}, indices: 0-{len(vocab)-1}")

        # Add this debug code after loading the vocabulary
        logging.info(f"Vocab keys (first 10): {list(vocab.keys())[:10]}")
        logging.info(f"Vocab values (first 10): {list(vocab.values())[:10]}")
        logging.info(f"Min index: {min(vocab.values()) if vocab.values() else 'No values'}")
        logging.info(f"Max index: {max(vocab.values()) if vocab.values() else 'No values'}")
        logging.info(f"Vocab size: {len(vocab)}")

        vectorizer = CountVectorizer(vocabulary=vocab)
        models['intent_vectorizer'] = vectorizer
        models['intent_labels'] = {int(k): v for k, v in labels.items()}
        logging.info("Intent model and artifacts loaded successfully.")

    except Exception as e:
        logging.exception("Failed to load intent model and artifacts")

    # 2. Load NER Model from MLflow
    try:
        logging.info("Loading NER model from MLflow...")
        # ner_model_uri = "models:/ner-training/Production" # Assumes model is in Production stage
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions("ner-training", stages=["Production"])[0]
        ner_model_uri = f"runs:/{latest.run_id}/spacy-ner-model"
        models['ner_model'] = mlflow.spacy.load_model(ner_model_uri)
        logging.info("NER model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load NER model: {e}")

    # 3. Load Retriever Artifacts from MinIO
    try:
        logging.info("Loading Retriever artifacts from MinIO...")
        s3 = boto3.client(
            's3',
            endpoint_url=f'http://{MINIO_ENDPOINT}',
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            faiss_path = os.path.join(tmpdir, "faq_index.faiss")
            docs_path = os.path.join(tmpdir, "documents.csv")
            s3.download_file(RETRIEVER_BUCKET, "faiss_index.faiss", faiss_path)
            s3.download_file(RETRIEVER_BUCKET, "documents.csv", docs_path)
            
            models['faiss_index'] = faiss.read_index(faiss_path)
            models['retriever_docs'] = pd.read_csv(docs_path)
        
        models['embedding_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        models['es_client'] = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
        logging.info("Retriever artifacts loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load retriever artifacts: {e}")

    logging.info("--- Model and artifact loading complete ---")


# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatInput):
    """Main endpoint for user-bot interaction."""
    if 'dialog_manager' not in models:
        raise HTTPException(status_code=503, detail="Dialog Manager is not available.")
    return await models['dialog_manager'].handle_message(request)

@app.post("/predict_intent", response_model=IntentResponse)
async def predict_intent(request: TextInput):
    """Predicts the intent of a given text."""
    vectorizer = models['intent_vectorizer']
    model = models['intent_model']
    labels = models['intent_labels']

    transformed_text = vectorizer.transform([request.text]).toarray()
    tensor_text = torch.tensor(transformed_text, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_text)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

    intent = labels[predicted_idx]
    return IntentResponse(intent=intent, confidence=confidence) 

@app.post("/extract_slots", response_model=NERResponse)
async def extract_slots(request: TextInput):
    """Extracts named entities (slots) from a given text."""
    ner_model = models['ner_model']
    doc = ner_model(request.text)
    slots = [
        Slot(entity=ent.label_, value=ent.text, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]
    return NERResponse(slots=slots)

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: TextInput):
    """Retrieves relevant documents using hybrid (dense + sparse) search."""
    embedding_model = models['embedding_model']
    faiss_index = models['faiss_index']
    docs_df = models['retriever_docs']
    es_client = models['es_client']
    
    # Dense search (FAISS)
    query_embedding = embedding_model.encode([request.text]).astype('float32')
    distances, indices = faiss_index.search(query_embedding, k=5)
    dense_results = [
        RetrievalResult(
            doc_id=docs_df.iloc[idx]['doc_id'],
            text=docs_df.iloc[idx]['text'],
            score=1 - dist  # Convert L2 distance to a similarity score
        ) for dist, idx in zip(distances[0], indices[0])
    ]

    # Sparse search (Elasticsearch)
    es_query = {"query": {"match": {"text": request.text}}}
    es_response = es_client.search(index=ES_INDEX_NAME, body=es_query, size=5)
    sparse_results = [
        RetrievalResult(
            doc_id=int(hit['_id']),
            text=hit['_source']['text'],
            score=hit['_score']
        ) for hit in es_response['hits']['hits']
    ]
    
    return RetrievalResponse(dense_results=dense_results, sparse_results=sparse_results)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: TextInput):
    """Placeholder for the generative response model."""
    # In a future step, this will take the retrieved context and user query
    # and feed them to a RAG model.
    return GenerateResponse(response="Generative model (RAG) is not yet implemented.")