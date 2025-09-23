import os
import argparse
import json
import logging
import warnings

# --- Suppress Warnings ---
# Suppress MLflow pkg_resources deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from typing import Dict, List

import dvc.api
import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import MLFlowLogger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

# Set better logging format
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Suppress some PyTorch Lightning verbose logs
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Suppress MLflow autolog warnings
mlflow.pytorch.autolog(disable=True)

try:
    mlflow.set_experiment("intent-classification-bitext")
except Exception as e:
    logging.warning(f"Could not set MLflow experiment: {e}")


class IntentDataset(Dataset):
    """Custom PyTorch Dataset for intent classification."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class IntentDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule to handle data loading and preparation."""
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.vectorizer = None
        self.label_encoder = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def load_data(self, split: str) -> pd.DataFrame:
        """Loads data from a .jsonl file using DVC API."""
        try:
            file_path = os.path.join(self.data_path, f'{split}.jsonl')
            with dvc.api.open(file_path, repo='.', rev='HEAD', mode='r') as f:
                data = []
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line.strip()))
                return pd.DataFrame(data)
        except Exception as e:
            logging.warning(f"DVC loading failed for {split}: {e}")
            # Fallback to local file
            local_file_path = os.path.join(self.data_path, f"{split}.jsonl")
            if os.path.exists(local_file_path):
                logging.info(f"Using local file: {local_file_path}")
                return pd.read_json(local_file_path, lines=True)
            else:
                raise FileNotFoundError(f"Could not load data for split: {split}")

    def setup(self, stage: str = None):
        """Prepares data for training, validation, and testing."""
        logging.info("Loading and preprocessing data...")
        
        train_df = self.load_data("train")
        val_df = self.load_data("val")
        test_df = self.load_data("test")

        # Fit vectorizer and label encoder on training data
        self.vectorizer = CountVectorizer(
            max_features=5000, 
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.label_encoder = LabelEncoder()

        X_train = self.vectorizer.fit_transform(train_df['text']).toarray()
        y_train = self.label_encoder.fit_transform(train_df['label'])

        X_val = self.vectorizer.transform(val_df['text']).toarray()
        y_val = self.label_encoder.transform(val_df['label'])

        X_test = self.vectorizer.transform(test_df['text']).toarray()
        y_test = self.label_encoder.transform(test_df['label'])

        self.train_dataset = IntentDataset(X_train, y_train)
        self.val_dataset = IntentDataset(X_val, y_val)
        self.test_dataset = IntentDataset(X_test, y_test)
        
        logging.info(f"Data loaded - Train: {len(self.train_dataset)}, "
                    f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,  # Reduced from 4 to avoid too many workers
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=2,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=2,
            persistent_workers=True
        )


class IntentClassifier(pl.LightningModule):
    """A simple intent classifier using a feed-forward neural network."""
    def __init__(self, vocab_size: int, num_classes: int, hidden_dim: int, learning_rate: float, dropout_rate: float = 0.3):
        super().__init__()
        self.save_hyperparameters()
        
        self.network = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        return {"preds": logits.argmax(dim=1), "labels": y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', acc, sync_dist=True)
        return {"preds": logits.argmax(dim=1), "labels": y}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def main():
    parser = argparse.ArgumentParser(description="Intent Classification Training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=5, help="Maximum epochs")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--data-path", type=str, default="ml/data/processed/intent_bitext", 
                       help="Path to the data directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Initialize DataModule
    dm = IntentDataModule(data_path=args.data_path, batch_size=args.batch_size)
    dm.setup()
    
    # Initialize Model
    model = IntentClassifier(
        vocab_size=len(dm.vectorizer.vocabulary_),
        num_classes=len(dm.label_encoder.classes_),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate
    )

    # MLflow Logger
    try:
        mlf_logger = MLFlowLogger(
            experiment_name="intent-classification-bitext",
            tracking_uri=MLFLOW_TRACKING_URI
        )
    except Exception as e:
        logging.error(f"MLflow logger failed: {e}")
        mlf_logger = None

    # Trainer with cleaner output
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=mlf_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=50,  # Less frequent logging
        check_val_every_n_epoch=1
    )

    if mlf_logger:
        with mlflow.start_run(run_id=mlf_logger.run_id):
            # Log hyperparameters
            mlflow.log_params({
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "hidden_dim": args.hidden_dim,
                "dropout_rate": args.dropout_rate,
                "vocab_size": len(dm.vectorizer.vocabulary_),
                "num_classes": len(dm.label_encoder.classes_)
            })
            
            logging.info("Starting model training...")
            trainer.fit(model, dm)
            
            logging.info("Starting model testing...")
            trainer.test(model, dm)

            # Log model and artifacts
            logging.info("Logging model and artifacts to MLflow...")
            mlflow.pytorch.log_model(model, "model")
            
            # Save preprocessing artifacts
            mlflow.log_dict(dm.vectorizer.vocabulary_, "vocabulary.json")
            mlflow.log_dict(
                {i: label for i, label in enumerate(dm.label_encoder.classes_)}, 
                "label_encoder_classes.json"
            )
            
            logging.info("Training completed successfully!")
            logging.info(f"Check MLflow UI at: {MLFLOW_TRACKING_URI}")
    else:
        logging.warning("Running without MLflow logging")
        trainer.fit(model, dm)
        trainer.test(model, dm)


if __name__ == "__main__":
    main()