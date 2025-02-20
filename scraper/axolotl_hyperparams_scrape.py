%cd /content/drive/MyDrive/Ai-Scraper
import yaml

yaml_string = yaml_string = """
base_model: Qwen/Qwen2.5-7B-Instruct-1M

load_in_4bit: true
strict: false

datasets:
  - path: /content/drive/MyDrive/Ai-Scraper/preprocessed_dataset_trial_tokenized.jsonl
    type:
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

# Specify your file path
file_path = 'axolotl_scrape_long.yaml'

# Write the YAML file
with open(file_path, 'w') as file:
    yaml.dump(yaml_dict, file)