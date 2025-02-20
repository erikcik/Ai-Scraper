import yaml

yaml_string = """
base_model: mistralai/Mistral-7B-v0.1

# Model loading settings
load_in_4bit: true  # Keep 4-bit quantization
strict: false

datasets:
  - path: /content/drive/MyDrive/Ai-Scraper/preprocessed_dataset_axolotl.jsonl  # Adjust path if needed
    type: completion
val_set_size: 0.05
output_dir: ./outputs/lora-out-h100

# Precision settings - H100 can handle bf16 well
bf16: true
fp16: false

# Memory management - Optimized for H100 80GB
gpu_memory_limit: 78GiB  # Leave some headroom
gradient_checkpointing: true
gradient_accumulation_steps: 8  # Reduced due to larger micro_batch_size
micro_batch_size: 4 # Increased micro_batch_size
eval_batch_size: 4
sample_packing: true
pad_to_sequence_len: true

# Sequence handling - keep consistent with preprocessing
sequence_len: 4096
group_by_length: true

# LoRA settings - keep the same, good starting point
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

# Training optimizations -  adjustments for H100
learning_rate: 2e-4  # Keep the same, good starting point
warmup_ratio: 0.03
weight_decay: 0.01
optimizer: adamw_bnb_8bit  # Keep 8-bit optimizer
max_grad_norm: 1.0

# Performance optimizations - enable all for H100
flash_attention: true
sdp_attention: true  # Use SDP, it's generally preferred

# Early stopping
early_stopping_patience: 3

# Save and evaluation - adjust as needed
save_steps: 50
eval_steps: 50
save_total_limit: 3
"""

# Convert the YAML string to a Python dictionary
yaml_dict = yaml.safe_load(yaml_string)

# Specify your file path -  H100 specific
file_path = 'axolotl_scrape_h100_v1.yaml'

# Write the YAML file
with open(file_path, 'w') as file:
    yaml.dump(yaml_dict, file)

print(f"YAML configuration for H100 saved to {file_path}") 