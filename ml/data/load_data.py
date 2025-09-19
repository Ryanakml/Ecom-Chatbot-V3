import os
import json
import argparse
from datasets import load_dataset

def load_intent_data(output_dir: str):
    """
    Downloads and processes the CLINC150 dataset for intent classification.

    This dataset contains 150 intent classes and one 'out_of_scope' class,
    making it ideal for real-world chatbot applications.

    Args:
        output_dir: The directory to save the processed data.
    """
    dataset_name = "clinc_oos"
    print(f"Downloading intent dataset: '{dataset_name}'...")
    dataset = load_dataset(dataset_name, 'plus')

    # Create the directory if it doesn't exist
    save_path = os.path.join(output_dir, "intent_clinc150")
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving data to '{save_path}'...")

    for split in dataset.keys():
        file_path = os.path.join(save_path, f"{split}.jsonl")
        with open(file_path, 'w') as f:
            for example in dataset[split]:
                # Get the string label from the integer class
                intent_label = dataset[split].features['intent'].int2str(example['intent'])
                record = {'text': example['text'], 'label': intent_label}
                f.write(json.dumps(record) + '\n')
    print(f"Successfully exported intent data to '{save_path}'.")


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
        help="The type of dataset to download ('intent' for CLINC150, 'ner' for CoNLL-2003)."
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