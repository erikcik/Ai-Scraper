import json
import torch
from torch.utils.data import Dataset
from transformers import (
    MarkupLMProcessor,
    MarkupLMForQuestionAnswering,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

class AmazonProductTokenDataset(Dataset):
    """
    A dataset for token‑level extractive QA.
    Each example consists of an HTML string, a question string, and an answer substring.
    We compute character-level offsets for the answer in the HTML and then use the processor’s
    offset mapping to map them to token positions.
    """
    def __init__(self, json_file, processor, max_length=512):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.qa_items = []  # List of tuples: (html, question, answer)

        self._extract_qa_items(self.data)

    def _extract_qa_items(self, data_obj):
        """
        Recursively walk through the output part of each JSON item.
        If an object has a non-null "value", use it as the answer.
        """
        if isinstance(data_obj, dict):
            if "value" in data_obj and data_obj["value"] is not None:
                answer = data_obj["value"].strip()
                if answer:
                    # For simplicity, use the current HTML and question from the parent item.
                    # In our dataset each JSON object already contains 'input' at the top level.
                    self.qa_items.append(answer)
            else:
                for v in data_obj.values():
                    self._extract_qa_items(v)
        elif isinstance(data_obj, list):
            for v in data_obj:
                self._extract_qa_items(v)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Each item in self.data is a dictionary with "input" and "output" keys.
        item = self.data[idx]
        html_str = item["input"]["html"]
        question_str = item["input"]["text"].strip()
        # For token-level training, we assume one target answer per example.
        # Here we assume that in "output", there is one field that contains a non-null "value".
        # (If there are multiple, you might create multiple examples per item.)
        answer_str = None
        # Traverse the output fields
        def traverse_output(obj):
            nonlocal answer_str
            if isinstance(obj, dict):
                if "value" in obj and obj["value"] is not None:
                    candidate = obj["value"].strip()
                    if candidate:
                        answer_str = candidate
                        return
                else:
                    for v in obj.values():
                        traverse_output(v)
                        if answer_str is not None:
                            return
            elif isinstance(obj, list):
                for v in obj:
                    traverse_output(v)
                    if answer_str is not None:
                        return

        traverse_output(item["output"])
        # If no answer found, fallback to an empty string.
        if answer_str is None:
            answer_str = ""

        # 1) Find the character offsets in the raw HTML (case-insensitive)
        lower_html = html_str.lower()
        lower_answer = answer_str.lower()
        answer_start_char = lower_html.find(lower_answer)
        if answer_start_char == -1:
            # If answer is not found, fallback to first character.
            answer_start_char = 0
            answer_end_char = 0
        else:
            answer_end_char = answer_start_char + len(answer_str)

        # 2) Encode with offset mapping
        encoding = self.processor(
            html_strings=html_str,
            questions=question_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze(0)

        offsets = encoding["offset_mapping"]  # shape: [seq_length, 2]
        start_positions, end_positions = None, None

        # 3) Map character offsets to token indices.
        for i, (start, end) in enumerate(offsets.tolist()):
            # Choose first token whose span covers answer_start_char
            if start_positions is None and start <= answer_start_char < end:
                start_positions = i
            if start < answer_end_char <= end:
                end_positions = i
                break

        if start_positions is None or end_positions is None:
            start_positions = 0
            end_positions = 0

        encoding["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
        encoding["end_positions"] = torch.tensor(end_positions, dtype=torch.long)

        # Remove offset mapping before returning
        del encoding["offset_mapping"]

        return encoding

def main():
    # Load the processor and the base model.
    processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
    processor.parse_html = True
    base_model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base")

    # Set up LoRA configuration.
    lora_config = LoraConfig(
        r=8,            # Rank of LoRA updates.
        lora_alpha=32,  # Scaling factor.
        lora_dropout=0.1,
        target_modules=["query", "value"]  # for example; adjust based on the model architecture.
    )
    # Wrap the model with LoRA.
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Optional: see how many params are trainable.

    # Create our token-level QA dataset
    dataset = AmazonProductTokenDataset("dataset.json", processor, max_length=512)

    # Train/validation split
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    training_args = TrainingArguments(
        output_dir="./markuplm_amazon_qa_token_lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs_token_lora",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        seed=42,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        fp16=True,
        optim="adamw_torch",
        learning_rate=2e-5,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model("./markuplm_amazon_qa_token_lora_final")
    print("Training complete. Model saved to ./markuplm_amazon_qa_token_lora_final")

if __name__ == "__main__":
    main()
