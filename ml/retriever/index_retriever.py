import argparse
import logging
import os
import time

import boto3
import faiss
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from datasets import load_dataset
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Service Configurations ---
MINIO_ENDPOINT = 'localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET_NAME = 'retriever-artifacts'

ES_HOST = 'localhost'
ES_PORT = 9200
ES_INDEX_NAME = 'faq-keyword-index'

def wait_for_elasticsearch(es_client, timeout=60):
    """Waits for Elasticsearch to be available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if es_client.ping():
                logging.info("Elasticsearch is up and running!")
                return True
        except Exception:
            logging.info("Waiting for Elasticsearch to start...")
            time.sleep(3)
    logging.error("Elasticsearch did not start within the timeout period.")
    return False

def get_s3_client():
    """Initializes and returns a boto3 S3 client for MinIO."""
    return boto3.client(
        's3',
        endpoint_url=f'http://{MINIO_ENDPOINT}',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

def create_minio_bucket(s3_client, bucket_name):
    """Creates a MinIO bucket if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        logging.info(f"Creating MinIO bucket '{bucket_name}'")
        s3_client.create_bucket(Bucket=bucket_name)

def load_and_prepare_data(dataset_name: str, sample_size: int = None) -> pd.DataFrame:
    """Loads a dataset from Hugging Face and prepares it for indexing."""
    logging.info(f"Loading dataset '{dataset_name}' from Hugging Face...")

    if sample_size:
        dataset = load_dataset(dataset_name, split=f"train[:{sample_size}]")
    else:
        dataset = load_dataset(dataset_name, split="train")

    df = dataset.to_pandas()

    if "question" in df.columns and "answer" in df.columns:
        logging.info("Using 'question' + 'answer' pairs")
        df = df[["question", "answer"]].dropna().reset_index(drop=True)
        df["text"] = df["question"] + " || " + df["answer"]
        doc_df = df[["text"]].reset_index(drop=True)
        doc_df["doc_id"] = doc_df.index
    else:
        candidate_cols = ["questions", "question1", "text", "content", "sentence"]
        chosen_col = next((col for col in candidate_cols if col in df.columns), None)
        if not chosen_col:
            raise ValueError(
                f"Dataset {dataset_name} tidak punya kolom teks yang dikenali! "
                f"Kolom yang tersedia: {list(df.columns)}"
            )
        if chosen_col == "questions":
            canonical_texts = df["questions"].explode().dropna().unique()
        else:
            canonical_texts = df[chosen_col].dropna().unique()

        doc_df = pd.DataFrame(canonical_texts, columns=["text"]).dropna().drop_duplicates()
        doc_df["doc_id"] = doc_df.index

    logging.info(f"Prepared {len(doc_df)} records for indexing.")
    return doc_df

def generate_embeddings(documents: pd.DataFrame, model_name: str) -> np.ndarray:
    """Generates dense vector embeddings for documents."""
    logging.info(f"Generating embeddings using '{model_name}'...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents['text'].tolist(), show_progress_bar=True)
    return embeddings.astype('float32') # FAISS requires float32

def create_faiss_index(embeddings: np.ndarray, index_path: str):
    """Creates and saves a FAISS index."""
    logging.info("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    logging.info(f"FAISS index with {index.ntotal} vectors saved to '{index_path}'.")

def index_in_elasticsearch(es_client, documents: pd.DataFrame):
    """Indexes documents into Elasticsearch for keyword search."""
    if es_client.indices.exists(index=ES_INDEX_NAME):
        logging.info(f"Deleting existing Elasticsearch index: '{ES_INDEX_NAME}'")
        es_client.indices.delete(index=ES_INDEX_NAME)

    logging.info(f"Creating new Elasticsearch index: '{ES_INDEX_NAME}'")
    es_client.indices.create(index=ES_INDEX_NAME)

    actions = [
        {
            "_index": ES_INDEX_NAME,
            "_id": row['doc_id'],
            "_source": {"text": row['text']}
        }
        for _, row in documents.iterrows()
    ]

    logging.info(f"Bulk indexing {len(actions)} documents into Elasticsearch...")
    bulk(es_client, actions)
    logging.info("Elasticsearch indexing complete.")

def upload_artifacts_to_minio(s3_client, artifacts: dict):
    """Uploads index artifacts to MinIO."""
    create_minio_bucket(s3_client, MINIO_BUCKET_NAME)
    for key, local_path in artifacts.items():
        logging.info(f"Uploading '{local_path}' to MinIO as '{key}'...")
        s3_client.upload_file(local_path, MINIO_BUCKET_NAME, key)
    logging.info("All artifacts uploaded to MinIO.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and index a retrieval system.")
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="all-MiniLM-L6-v2", 
        help="Sentence Transformer model name."
    )
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="toughdata/quora-question-answer-dataset", 
        help="Hugging Face dataset name."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of rows to sample from dataset for testing."
    )
    args = parser.parse_args()

    # Create a local directory for artifacts
    os.makedirs("artifacts", exist_ok=True)
    FAISS_INDEX_FILE = "artifacts/faq_index.faiss"
    DOCUMENTS_FILE = "artifacts/faq_documents.csv"

    # 1. Initialize Clients
    es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
    s3 = get_s3_client()
    if not wait_for_elasticsearch(es):
        exit(1)

    # 2. Load and Prepare Data
    docs_df = load_and_prepare_data(args.dataset_name)
    docs_df.to_csv(DOCUMENTS_FILE, index=False)

    # 3. Generate Embeddings
    doc_embeddings = generate_embeddings(docs_df, args.model_name)

    # 4. Create FAISS Index (Dense)
    create_faiss_index(doc_embeddings, FAISS_INDEX_FILE)

    # 5. Index in Elasticsearch (Sparse)
    index_in_elasticsearch(es, docs_df)

    # 6. Upload Artifacts to MinIO
    artifacts_to_upload = {
        "faiss_index.faiss": FAISS_INDEX_FILE,
        "documents.csv": DOCUMENTS_FILE
    }
    upload_artifacts_to_minio(s3, artifacts_to_upload)

    logging.info("\nRetriever indexing process complete.")
    logging.info(f"FAISS index and document map are stored in the '{MINIO_BUCKET_NAME}' bucket in MinIO.")
    logging.info(f"Keyword index is ready in Elasticsearch under the name '{ES_INDEX_NAME}'.")
