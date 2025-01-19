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
    A dataset for token‐level extractive QA.
    Each example consists of an HTML string, a question string, and an answer substring.
    The answer substring is located in the HTML text (via a case‐insensitive search)
    and its character offsets are mapped to token offsets.
    """
    def __init__(self, json_file, processor, max_length=512):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.qa_items = []  # List of tuples: (html, question, answer)

        for item in self.data:
            html_str = item["input"]["html"]
            question_str = item["input"]["text"].strip()
            # Traverse the output field recursively to get the first non-null value.
            answer_str = self._traverse_output(item["output"])
            if answer_str is None:
                answer_str = ""
            self.qa_items.append((html_str, question_str, answer_str))

    def _traverse_output(self, obj):
        if isinstance(obj, dict):
            if "value" in obj and obj["value"] is not None:
                val = obj["value"].strip()
                if val:
                    return val
            for v in obj.values():
                ans = self._traverse_output(v)
                if ans is not None:
                    return ans
        elif isinstance(obj, list):
            for v in obj:
                ans = self._traverse_output(v)
                if ans is not None:
                    return ans
        return None

    def __len__(self):
        return len(self.qa_items)

    def __getitem__(self, idx):
        html_str, question_str, answer_str = self.qa_items[idx]

        # 1) Find answer substring character offsets (case insensitive)
        lower_html = html_str.lower()
        lower_answer = answer_str.lower()
        answer_start_char = lower_html.find(lower_answer)
        if answer_start_char == -1:
            answer_start_char, answer_end_char = 0, 0
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

        offsets = encoding["offset_mapping"]  # shape [seq_length, 2]
        start_positions, end_positions = None, None

        # 3) Map char offsets to token indices using the offset mapping.
        for i, (start, end) in enumerate(offsets.tolist()):
            if start_positions is None and start <= answer_start_char < end:
                start_positions = i
            if end_positions is None and end > answer_end_char >= start:
                end_positions = i
                break

        if start_positions is None or end_positions is None:
            start_positions, end_positions = 0, 0

        encoding["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
        encoding["end_positions"]   = torch.tensor(end_positions, dtype=torch.long)
        # Remove offset_mapping key to avoid extra inputs.
        del encoding["offset_mapping"]

        return encoding

def main():
    # Load the processor and base model.
    processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
    processor.parse_html = True
    base_model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base")

    # Configure LoRA.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]  # Adjust target modules as needed.
    )
    # Wrap the base model with LoRA.
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset.
    dataset = AmazonProductTokenDataset("dataset.json", processor, max_length=512)
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
        load_best_model_at_end=False,  # Remove best model saving to avoid missing metric key
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
