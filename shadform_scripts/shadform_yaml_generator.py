import yaml
import argparse

def generate_yaml(dataset_path, output_path='axolotl_scrape_long.yaml'):
    yaml_string = f"""
base_model: Qwen/Qwen2.5-7B-Instruct-1M

load_in_4bit: true
strict: false

datasets:
  - path: {dataset_path}
    type: alpaca
val_set_size: 0.05
output_dir: ./outputs/lora-out
sequence_len: 524288
sample_packing: true

adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
  - w1
  - w2
  - w3

gradient_accumulation_steps: 2
micro_batch_size: 1
num_epochs: 4
optimizer: adafactor
lr_scheduler: cosine
learning_rate: 2e-5

bf16: true

gradient_checkpointing: true
logging_steps: 10
flash_attention: true
"""

    # Convert the YAML string to a Python dictionary
    yaml_dict = yaml.safe_load(yaml_string)

    # Write the YAML file
    with open(output_path, 'w') as file:
        yaml.dump(yaml_dict, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate YAML configuration file for training')
    parser.add_argument('--dataset', type=str, default="/home/user/Ai-Scraper/web_scraping_dataset.jsonl",
                        help='Path to the dataset file')
    parser.add_argument('--output', type=str, default="axolotl_scrape_long.yaml",
                        help='Path to output YAML file')
    
    args = parser.parse_args()
    
    generate_yaml(args.dataset, args.output)
    print(f"YAML configuration file generated at: {args.output}")