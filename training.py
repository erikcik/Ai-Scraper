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
import os

class AmazonProductTokenDataset(Dataset):
    """
    Token-level extractive QA dataset.
    For each JSON item, we extract the HTML, the query (question), and the first non-null answer.
    We then compute token-level start/end positions from character offsets.
    """
    def __init__(self, json_file, processor, max_length=512):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.qa_items = []  # List of tuples: (html, question, answer)
        for item in self.data:
            html = item["input"]["html"]
            question = item["input"]["text"].strip()
            answer = self._find_answer(item["output"])
            if answer is None:
                answer = ""
            self.qa_items.append((html, question, answer))

    def _find_answer(self, output_obj):
        # Traverse the output dictionary/list and return the first non-null "value"
        if isinstance(output_obj, dict):
            if "value" in output_obj and output_obj["value"]:
                return output_obj["value"].strip()
            for v in output_obj.values():
                ans = self._find_answer(v)
                if ans is not None:
                    return ans
        elif isinstance(output_obj, list):
            for v in output_obj:
                ans = self._find_answer(v)
                if ans is not None:
                    return ans
        return None

    def __len__(self):
        return len(self.qa_items)

    def __getitem__(self, idx):
        html, question, answer = self.qa_items[idx]
        # Lowercase for matching.
        lower_html = html.lower()
        lower_answer = answer.lower()
        answer_start_char = lower_html.find(lower_answer)
        if answer_start_char == -1:
            answer_start_char, answer_end_char = 0, 0
        else:
            answer_end_char = answer_start_char + len(answer)

        # Encoding with offset mapping (so we can map char positions to tokens)
        encoding = self.processor(
            html_strings=html,
            questions=question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        # Remove batch dimension
        for key, value in encoding.items():
            encoding[key] = value.squeeze(0)
        offsets = encoding["offset_mapping"]  # shape: [seq_length, 2]

        start_index, end_index = None, None
        for i, (start, end) in enumerate(offsets.tolist()):
            if start_index is None and start <= answer_start_char < end:
                start_index = i
            if end_index is None and end > answer_end_char >= start:
                end_index = i
                break
        if start_index is None or end_index is None:
            start_index, end_index = 0, 0
        encoding["start_positions"] = torch.tensor(start_index, dtype=torch.long)
        encoding["end_positions"] = torch.tensor(end_index, dtype=torch.long)
        # Remove offsets from inputs
        del encoding["offset_mapping"]
        return encoding

def main():
    # Create output directory if it doesn't exist
    output_dir = "./markuplm_amazon_qa_token_lora_final"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading processor and base model...")
    processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
    processor.parse_html = True
    base_model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base")

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
        task_type="QUESTION_ANSWERING"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    print("Preparing dataset...")
    dataset = AmazonProductTokenDataset("dataset.json", processor, max_length=512)
    total_items = len(dataset)
    train_size = int(0.8 * total_items)
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_items))

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
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

    print("Starting training...")
    trainer.train()

    print("Training complete. Saving model...")
    
    # Merge LoRA weights and save the complete model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    
    # Save processor configuration
    processor.save_pretrained(output_dir)
    
    # Save additional model files
    config = merged_model.config
    config.save_pretrained(output_dir)
    
    print(f"Model and processor saved to {output_dir}")

if __name__ == "__main__":
    main()
