import os
import argparse
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import yaml
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MinIO Configuration ---
# These credentials match the ones in our docker-compose.yml file.
# In a real production system, use environment variables or a secrets manager.
MINIO_ENDPOINT = 'localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_BUCKET_NAME = 'chatbot-data' # The bucket we will use
USE_SECURE = False # Set to False for local http MinIO

def setup_minio_bucket(s3_client, bucket_name: str):
    """Checks if a bucket exists and creates it if it doesn't."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logging.info(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.info(f"Bucket '{bucket_name}' not found. Creating it now.")
            s3_client.create_bucket(Bucket=bucket_name)
            logging.info(f"Successfully created bucket '{bucket_name}'.")
        else:
            logging.error(f"Error checking for bucket: {e}")
            raise

def load_raw_data(dataset_type: str):
    """
    Runs the load_data.py script as a subprocess.
    This ensures our raw data is present locally before ingestion.
    """
    script_path = 'ml/data/load_data.py'
    logging.info(f"Running data loading script for '{dataset_type}' dataset...")
    try:
        subprocess.run(
            ['python', script_path, '--dataset', dataset_type],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info("Data loading script completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to run data loading script.")
        logging.error(f"Stderr: {e.stderr}")
        raise e

def upload_to_minio(s3_client, local_directory: str, bucket_name: str):
    """
    Uploads a directory and its contents to a MinIO bucket.
    """
    logging.info(f"Uploading data from '{local_directory}' to MinIO bucket '{bucket_name}'...")
    if not os.path.isdir(local_directory):
        logging.error(f"Directory not found: {local_directory}")
        return

    for root, _, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Create a relative path for the object key in S3
            relative_path = os.path.relpath(local_path, local_directory)
            s3_object_key = f"{os.path.basename(local_directory)}/{relative_path}"

            try:
                s3_client.upload_file(local_path, bucket_name, s3_object_key)
                logging.info(f"Successfully uploaded '{local_path}' to '{bucket_name}/{s3_object_key}'")
            except NoCredentialsError:
                logging.error("Credentials not available for MinIO.")
                raise
            except Exception as e:
                logging.error(f"Failed to upload {local_path}: {e}")
                raise

def create_dvc_file(dataset_type: str, local_data_path: str):
    """
    Creates or updates the dvc.yaml file to track the ingested data.
    """
    logging.info("Creating/updating dvc.yaml...")
    dvc_config = {
        'stages': {
            f'ingest_{dataset_type}_data': {
                'cmd': f'python ml/data/ingest_data.py --dataset {dataset_type}',
                'deps': [
                    'ml/data/ingest_data.py',
                    'ml/data/load_data.py'
                ],
                'outs': [
                    local_data_path
                ]
            }
        }
    }
    with open('dvc.yaml', 'w') as f:
        yaml.dump(dvc_config, f, sort_keys=False)
    logging.info("dvc.yaml has been updated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest data: load it, upload to MinIO, and track with DVC."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['intent', 'ner'],
        help="The type of dataset to ingest ('intent' or 'ner')."
    )
    args = parser.parse_args()

    # Define the local path based on dataset type
    dataset_name = "intent_clinc150" if args.dataset == 'intent' else "conll2003hf"
    local_raw_path = os.path.join('ml', 'data', 'raw', dataset_name)

    # 1. Run the script to download data locally
    load_raw_data(args.dataset)

    # 2. Connect to MinIO and upload the data
    s3 = boto3.client(
        's3',
        endpoint_url=f'http://{MINIO_ENDPOINT}',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=boto3.session.Config(signature_version='s3v4')
    )
    setup_minio_bucket(s3, MINIO_BUCKET_NAME)
    upload_to_minio(s3, local_raw_path, MINIO_BUCKET_NAME)

    # 3. Create the dvc.yaml file to track this step
    create_dvc_file(args.dataset, local_raw_path)

    logging.info("\nData ingestion process complete.")
    logging.info("Next steps:")
    logging.info("1. Initialize DVC: 'dvc init'")
    logging.info("2. Configure DVC remote storage (see instructions).")
    logging.info("3. Run 'dvc repro' to execute the pipeline.")