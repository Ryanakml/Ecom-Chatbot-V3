import logging
import os
import tempfile

import boto3
import faiss
import mlflow
import numpy as np
import pandas as pd
import spacy
import torch
from fastapi import FastAPI
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
        intent_model_uri = "models:/intent-classification/Production" # Assumes model is in Production stage
        models['intent_model'] = mlflow.pytorch.load_model(intent_model_uri)
        
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions("intent-classification", stages=["Production"])[0]
        run_id = latest_version.run_id
        
        # Download artifacts associated with the model run
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = client.download_artifacts(run_id, "vocabulary.json", tmpdir)
            labels_path = client.download_artifacts(run_id, "label_encoder_classes.json", tmpdir)
            
            with open(vocab_path) as f:
                vocab = json.load(f)
            with open(labels_path) as f:
                labels = json.load(f)

        vectorizer = CountVectorizer(vocabulary=vocab)
        models['intent_vectorizer'] = vectorizer
        models['intent_labels'] = {int(k): v for k, v in labels.items()}
        logging.info("Intent model and artifacts loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load intent model: {e}")

    # 2. Load NER Model from MLflow
    try:
        logging.info("Loading NER model from MLflow...")
        ner_model_uri = "models:/ner-training/Production" # Assumes model is in Production stage
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
@app.post("/predict_intent", response_model=IntentResponse)
async def predict_intent(request: TextInput):
    """Predicts the intent of a given text."""
    vectorizer = models['intent_vectorizer']
    model = models['intent_model']
    labels = models['intent_labels']

    transformed_text = vectorizer.transform([request.text]).toarray()
    tensor_text = torch.tensor(transformed_text, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        outputs = model(tensor_text)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    intent = labels[predicted_idx.item()]
    return IntentResponse(intent=intent, confidence=confidence.item())

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