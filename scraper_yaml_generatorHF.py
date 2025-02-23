import yaml

yaml_string = """
base_model: mistralai/Mistral-7B-v0.1

# Model loading settings
load_in_4bit: true
strict: false
trust_remote_code: true

datasets:
  - path: /content/drive/MyDrive/Ai-Scraper/preprocessed_dataset_axolotl_2.jsonl
    type: completion
val_set_size: 0.05
output_dir: ./outputs/lora-out

# Chat template settings
chat_template: chatml  # Changed from tokenizer_default to chatml
train_on_inputs: true

# Special tokens config
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
  pad_token: "</s>"

# Precision settings
bf16: true
fp16: false

# Memory management
gpu_memory_limit: 38GiB
gradient_checkpointing: true
gradient_accumulation_steps: 16
micro_batch_size: 1
eval_batch_size: 1
sample_packing: true
pad_to_sequence_len: true

# Sequence handling
sequence_len: 4096
group_by_length: true

# LoRA settings
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training optimizations
learning_rate: 2e-4
warmup_ratio: 0.03
weight_decay: 0.01
optimizer: adamw_bnb_8bit
max_grad_norm: 1.0

# Performance optimizations
flash_attention: true
sdp_attention: true

# Early stopping
early_stopping_patience: 3

# Save and evaluation
save_steps: 50
eval_steps: 50
save_total_limit: 3
"""

# Convert the YAML string to a Python dictionary
yaml_dict = yaml.safe_load(yaml_string)

# Specify your file path
file_path = 'axolotl_scrape_v2.yaml'

# Write the YAML file
with open(file_path, 'w') as file:
    yaml.dump(yaml_dict, file)