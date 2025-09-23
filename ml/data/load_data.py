import os
import json
import argparse
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_intent_data(output_dir: str, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Downloads and processes the Bitext E-commerce dataset for intent classification.

    This dataset contains 150 intent classes and one 'out_of_scope' class,
    making it ideal for real-world chatbot applications.

    Args:
        output_dir: The directory t2o save the processed data.
    """
    dataset_name = "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset"
    dataset = load_dataset(dataset_name)

    # Create the directory if it doesn't exist
    save_path = os.path.join(output_dir, "intent_bitext")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving data to '{save_path}'...")

    # Train data
    full_train = dataset['train']

    # Split train vs temp (val+test)
    train_test = full_train.train_test_split(test_size=(1-train_ratio), seed=seed)
    train_data = train_test['train']
    temp_data = train_test['test']

    # Split temp (val+test) into val and test
    val_test = temp_data.train_test_split(
        test_size=test_ratio / (val_ratio + test_ratio),
        seed=seed
    )
    val_data = val_test['train']
    test_data = val_test['test']

    # Export to jsonl file
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }

    for split, dset in splits.items():
        file_path = os.path.join(save_path, f"{split}.jsonl")
        with open(file_path, 'w') as f:
            for example in dset:
                # Get the string label from the integer class
                record = {
                    'text': example['instruction'],
                    'label': example['intent']
                }
                f.write(json.dumps(record) + '\n')
        print(f"Exported {split} -> {file_path} ({len(dset)} samples)")
    print(f"Successfully exported intent dataset with train/val/test splits to '{save_path}'.")


def load_ner_data(output_dir: str):
    """
    Downloads and processes the CoNLL-2003 dataset for Named Entity Recognition.

    This dataset is a standard benchmark for identifying entities like persons,
    organizations, locations, etc.

    Args:
        output_dir: The directory to save the processed data.
    """
    dataset_name = "conll2003"
    print(f"Downloading NER dataset: '{dataset_name}'...")
    dataset = load_dataset(dataset_name)

    # Create the directory if it doesn't exist
    save_path = os.path.join(output_dir, "conll2003hf")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving data to '{save_path}'...")

    # Get the mapping from integer ID to tag name
    ner_tags_map = dataset['train'].features['ner_tags'].feature.names

    for split in dataset.keys():
        file_path = os.path.join(save_path, f"{split}.jsonl")
        with open(file_path, 'w') as f:
            for example in dataset[split]:
                # Map integer tags to their string representations
                tags = [ner_tags_map[tag_id] for tag_id in example['ner_tags']]
                record = {'tokens': example['tokens'], 'tags': tags}
                f.write(json.dumps(record) + '\n')
    print(f"Successfully exported NER data to '{save_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process production-ready datasets for chatbot training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['intent', 'ner'],
        help="The type of dataset to download ('intent' for Bitext E-commerce, 'ner' for CoNLL-2003)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/data/raw",
        help="The directory where the processed data will be saved."
    )
    args = parser.parse_args()

    if args.dataset == 'intent':
        load_intent_data(args.output_dir)
    elif args.dataset == 'ner':
        load_ner_data(args.output_dir)
    else:
        print("Invalid dataset choice. Please choose 'intent' or 'ner'.")