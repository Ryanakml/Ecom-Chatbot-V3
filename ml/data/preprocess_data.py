import os
import argparse
import boto3
from botocore.exceptions import ClientError
import json
import yaml
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Setup ---
# The user needs to run this manually once, or we can do it in the script.
try:
    stopwords.words('english')
except LookupError:
    logging.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logging.info("Downloading NLTK WordNet lemmatizer...")
    nltk.download('wordnet', quiet=True)


# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MINIO_ENDPOINT = 'localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET_NAME = 'chatbot-data'

def get_s3_client():
    """Initializes and returns a boto3 S3 client."""
    return boto3.client(
        's3',
        endpoint_url=f'http://{MINIO_ENDPOINT}',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

def download_from_minio(s3_client, bucket: str, remote_dir: str, local_dir: str):
    """Downloads a directory from MinIO."""
    logging.info(f"Downloading data from s3://{bucket}/{remote_dir} to {local_dir}...")
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=remote_dir)
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            local_file_path = os.path.join(local_dir, os.path.basename(key))
            logging.info(f"Downloading {key} to {local_file_path}")
            s3_client.download_file(bucket, key, local_file_path)

def upload_to_minio(s3_client, local_dir: str, bucket: str, remote_dir: str):
    """Uploads a directory to MinIO."""
    logging.info(f"Uploading data from {local_dir} to s3://{bucket}/{remote_dir}...")
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        remote_path = f"{remote_dir}/{filename}"
        logging.info(f"Uploading {local_path} to {remote_path}")
        s3_client.upload_file(local_path, bucket, remote_path)

def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes a single piece of text.
    - Lowercases
    - Removes punctuation
    - Removes stop words
    - Lemmatizes
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stop words and lemmatize
    processed_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(processed_tokens)

def process_data(input_dir: str, output_dir: str, dataset_type: str):
    """
    Reads raw data, processes it, and saves to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Processing data from {input_dir} to {output_dir}")

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                record = json.loads(line)
                if dataset_type == 'intent':
                    record['text'] = preprocess_text(record['text'])
                elif dataset_type == 'ner':
                    # For NER, we just lowercase to preserve token alignment with tags
                    record['tokens'] = [token.lower() for token in record['tokens']]
                outfile.write(json.dumps(record) + '\n')
    logging.info("Data processing complete.")

def update_dvc_file(dataset_type: str, deps_path: str, outs_path: str):
    """Adds the preprocess stage to the dvc.yaml file."""
    logging.info("Updating dvc.yaml with preprocess stage...")
    with open('dvc.yaml', 'r') as f:
        dvc_config = yaml.safe_load(f)

    stage_name = f'preprocess_{dataset_type}_data'
    dvc_config['stages'][stage_name] = {
        'cmd': f'python ml/data/preprocess_data.py --dataset {dataset_type}',
        'deps': [
            deps_path,
            'ml/data/preprocess_data.py'
        ],
        'outs': [
            outs_path
        ]
    }
    with open('dvc.yaml', 'w') as f:
        yaml.dump(dvc_config, f, sort_keys=False)
    logging.info("dvc.yaml updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw chatbot data.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['intent', 'ner'],
        help="The type of dataset to preprocess."
    )
    args = parser.parse_args()

    dataset_name = "intent_bitext" if args.dataset == 'intent' else "conll2003hf"
    raw_local_path = os.path.join('ml', 'data', 'raw', dataset_name)
    processed_local_path = os.path.join('ml', 'data', 'processed', dataset_name)

    s3_client = get_s3_client()

    # 1. Download raw data from MinIO (as defined by the previous DVC stage)
    download_from_minio(s3_client, MINIO_BUCKET_NAME, dataset_name, raw_local_path)

    # 2. Process the data
    process_data(raw_local_path, processed_local_path, args.dataset)

    # 3. Upload processed data back to MinIO
    upload_to_minio(s3_client, processed_local_path, MINIO_BUCKET_NAME, f"processed/{dataset_name}")

    # 4. Update dvc.yaml to track this new stage
    update_dvc_file(args.dataset, raw_local_path, processed_local_path)

    logging.info("\nPreprocessing pipeline step complete.")
    logging.info("Run 'dvc repro' to execute the full pipeline.")
    logging.info("Then, run 'dvc push' to save processed data to remote storage.")