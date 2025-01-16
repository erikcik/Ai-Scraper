### sample_colab_training.py

# Mount Google Drive (avoid losing your model in case Colab disconnects)
from google.colab import drive
drive.mount('/content/gdrive')

from transformers import MarkupLMProcessor, MarkupLMForQuestionAnswering
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import json

class AmazonProductDataset(Dataset):
    def __init__(self, json_file, processor):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.qa_pairs = []
        for item in self.data:
            html_content = item['input']['html']
            input_text = item['input']['text']
            for field_name, field_data in item['output'].items():
                if isinstance(field_data, list):
                    for entry in field_data:
                        if all(v.get('value') is None for v in entry.values()):
                            continue
                        self.qa_pairs.append({
                            'html': html_content,
                            'question': input_text,
                            'answer': str(entry),
                            'xpath': next((v['element'] for v in entry.values() if v.get('element')), None)
                        })
                else:
                    if field_data.get('value') is not None:
                        self.qa_pairs.append({
                            'html': html_content,
                            'question': input_text,
                            'answer': field_data['value'],
                            'xpath': field_data['element']
                        })

    def __getitem__(self, idx):
        qa_pair = self.qa_pairs[idx]
        encoding = self.processor(
            html_strings=qa_pair['html'],
            questions=qa_pair['question'],
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        if 'answer' in qa_pair:
            features = self.processor.feature_extractor(qa_pair['html'])
            answer_text = qa_pair['answer'].strip().lower()
            found_index = None
            for idx_node, node_text in enumerate(features['nodes']):
                if isinstance(node_text, list):
                    node_text = " ".join(str(x) for x in node_text)
                node_text_str = str(node_text).strip()
                if answer_text in node_text_str.lower():
                    found_index = idx_node
                    break
            if found_index is None:
                found_index = 0
            encoding['start_positions'] = torch.tensor(found_index)
            encoding['end_positions'] = torch.tensor(found_index)
        return encoding

    def __len__(self):
        return len(self.qa_pairs)

def main():
    # Processor + model init
    processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
    processor.parse_html = True
    model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base")

    # Load dataset
    dataset = AmazonProductDataset("/content/gdrive/MyDrive/your_folder/dataset.json", processor)

    # Train/validation split
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, dataset_size))

    # Training args: saving to Drive so we don’t lose progress on disconnect
    training_args = TrainingArguments(
        output_dir="/content/gdrive/MyDrive/markuplm_amazon_qa_ckpts",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="/content/gdrive/MyDrive/markuplm_logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,  # frequently save to avoid losing progress
        save_total_limit=3,  # keep only the last 3 checkpoints
        load_best_model_at_end=True,
        seed=42,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=16,
        fp16=True,
        optim='adamw_torch',
        learning_rate=2e-5,
        max_grad_norm=1.0,
        # You can also use push_to_hub=True if you have a HF account
        # push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Training
    trainer.train()

    # Final save
    trainer.save_model("/content/gdrive/MyDrive/markuplm_amazon_qa_final")

if __name__ == "__main__":
    main()
