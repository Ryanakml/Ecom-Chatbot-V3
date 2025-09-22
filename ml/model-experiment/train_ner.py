import os
import argparse
import json
import logging
import random
from pathlib import Path
import subprocess

import dvc.api
import mlflow
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import fsspec

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ner-training")

def read_jsonl(path: str):
    """Helper untuk baca JSONL dari lokal atau remote S3"""
    if path.startswith("s3://"):
        with fsspec.open(path, "r") as f:
            return [json.loads(line) for line in f]
    else:
        with open(path, "r") as f:
            return [json.loads(line) for line in f]

def load_data_from_dvc(dataset_name: str):
    """Load train & validation dataset dari DVC (lokal atau remote)."""
    
    try:
        # Langsung baca data dari DVC (tidak perlu get_url)
        train_content = dvc.api.read(
            path=f"ml/data/processed/{dataset_name}/train.jsonl",
            repo=".",
            rev="HEAD",
            mode="r"
        )
        val_content = dvc.api.read(
            path=f"ml/data/processed/{dataset_name}/validation.jsonl",
            repo=".",
            rev="HEAD", 
            mode="r"
        )
        
        # Parse JSONL content
        train_data = [json.loads(line) for line in train_content.strip().split('\n') if line.strip()]
        val_data = [json.loads(line) for line in val_content.strip().split('\n') if line.strip()]

    except Exception as e:
        logging.warning(f"DVC failed, fallback to local: {e}")
        
        # Fallback: load dari local files
        train_data = read_jsonl(f"ml/data/processed/{dataset_name}/train.jsonl")
        val_data = read_jsonl(f"ml/data/processed/{dataset_name}/validation.jsonl")

    return train_data, val_data

def convert_to_spacy_format(data: list) -> list:
    """
    Converts data from token/tag list format to spaCy's entity offset format.
    Example: (text, {"entities": [(start, end, label), ...]})
    """
    spacy_data = []
    for record in data:
        tokens = record['tokens']
        tags = record['tags']
        
        text = " ".join(tokens)
        entities = []
        current_pos = 0
        
        for token, tag in zip(tokens, tags):
            start = text.find(token, current_pos)
            end = start + len(token)
            current_pos = end
            
            if tag != 'O': # 'O' means no entity
                # The tag format is B-TYPE, I-TYPE (e.g., B-PER, I-PER)
                entity_label = tag.split('-')[1]
                entities.append((start, end, entity_label))
        
        spacy_data.append((text, {"entities": entities}))
        
    return spacy_data

def train_spacy_ner(train_data: list, val_data: list, base_model: str, iterations: int):
    """Trains a spaCy NER model."""
    # Load a blank model or a pre-trained one
    if base_model:
        nlp = spacy.load(base_model)
        logging.info(f"Loaded model '{base_model}'")
    else:
        nlp = spacy.blank("en")
        logging.info("Created blank 'en' model")

    # Add the NER pipe if it doesn't exist
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from the training data
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipes for training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
            
            # Evaluate on validation data
            scorer = spacy.scorer.Scorer()
            val_examples = []
            for text, annotations in val_data:
                 doc = nlp.make_doc(text)
                 val_examples.append(Example.from_dict(doc, annotations))
            scores = scorer.score(val_examples)
            
            logging.info(f"Iteration {itn+1}/{iterations} | Loss: {losses['ner']:.3f} | Val F-Score: {scores['ents_f']:.3f}")
            mlflow.log_metric("train_loss", losses['ner'], step=itn)
            mlflow.log_metric("val_f_score", scores['ents_f'], step=itn)
            
    return nlp, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--base-model", type=str, default="en_core_web_sm", help="Base spaCy model.")
    args = parser.parse_args()

    with mlflow.start_run() as run:
        logging.info("Starting NER model training run...")
        mlflow.log_params(vars(args))

        # 1. Load and convert data
        logging.info("Loading and converting data for spaCy...")
        # Note: We train on CoNLL-2003 which has PER, ORG, LOC entities.
        # To train on custom entities like 'product' or 'order_id', you would
        # need to create a custom labeled dataset (e.g., with Label Studio).
        train_records, val_records = load_data_from_dvc("conll2003hf")
        train_data_spacy = convert_to_spacy_format(train_records)
        val_data_spacy = convert_to_spacy_format(val_records)
        logging.info(f"Loaded {len(train_data_spacy)} training examples and {len(val_data_spacy)} validation examples.")

        # 2. Train the model
        trained_nlp, final_scores = train_spacy_ner(
            train_data=train_data_spacy,
            val_data=val_data_spacy,
            base_model=args.base_model,
            iterations=args.iterations
        )
        
        # 3. Log final metrics
        logging.info("Logging final metrics to MLflow...")
        final_metrics = {
            "ents_f": final_scores["ents_f"],
            "ents_p": final_scores["ents_p"],
            "ents_r": final_scores["ents_r"],
        }
        mlflow.log_metrics(final_metrics)
        logging.info(f"Final evaluation scores: {final_metrics}")

        # 4. Log the trained model to MLflow
        logging.info("Logging trained spaCy model to MLflow...")
        mlflow.spacy.log_model(
            spacy_model=trained_nlp,
            artifact_path="spacy-ner-model"
        )
        
        logging.info("NER model training run finished successfully!")