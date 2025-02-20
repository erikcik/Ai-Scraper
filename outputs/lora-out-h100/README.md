---
library_name: peft
license: apache-2.0
base_model: mistralai/Mistral-7B-v0.1
tags:
- generated_from_trainer
datasets:
- ./preproc_ds_chunked.jsonl
model-index:
- name: outputs/lora-out-h100
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.7.0`
```yaml
adapter: qlora
base_model: mistralai/Mistral-7B-v0.1
bf16: true
datasets:
- path: ./preproc_ds_chunked.jsonl
  type: completion
early_stopping_patience: 3
eval_batch_size: 4
eval_steps: 50
flash_attention: true
fp16: false
gpu_memory_limit: 78GiB
gradient_accumulation_steps: 8
gradient_checkpointing: true
group_by_length: true
learning_rate: 2e-4
load_in_4bit: true
lora_alpha: 64
lora_dropout: 0.05
lora_r: 32
lora_target_modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
max_grad_norm: 1.0
micro_batch_size: 4
optimizer: adamw_bnb_8bit
output_dir: ./outputs/lora-out-h100
pad_to_sequence_len: true
sample_packing: true
save_steps: 50
save_total_limit: 3
sdp_attention: true
sequence_len: 4096
strict: false
val_set_size: 0.05
warmup_ratio: 0.03
weight_decay: 0.01

```

</details><br>

# outputs/lora-out-h100

This model is a fine-tuned version of [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the ./preproc_ds_chunked.jsonl dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2033

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_BNB with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 15
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| No log        | 0.0019 | 1    | 1.5435          |
| 0.3747        | 0.0935 | 50   | 0.6355          |
| 0.1048        | 0.1870 | 100  | 0.3937          |
| 0.0222        | 0.2805 | 150  | 0.2892          |
| 0.0088        | 0.3740 | 200  | 0.2054          |
| 0.0062        | 0.4675 | 250  | 0.2024          |
| 0.0045        | 0.5610 | 300  | 0.2095          |
| 0.0041        | 0.6545 | 350  | 0.2050          |
| 0.0037        | 0.7480 | 400  | 0.2033          |


### Framework versions

- PEFT 0.14.0
- Transformers 4.48.3
- Pytorch 2.5.1+cu121
- Datasets 3.2.0
- Tokenizers 0.21.0