import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from scraper.cleanhtml_trial import clean_html
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(base_model: str, model_path: str) -> Tuple[Any, Any]:
    """Load the fine-tuned model and tokenizer"""
    logger.info(f"Loading base model: {base_model}")
    
    # Set environment variables for better CUDA handling
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Initialize tokenizer with proper padding token
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model with 4-bit quantization
    logger.info(f"Loading adapter from: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, tokenizer

def prepare_chat_prompt(tokenizer: Any, html: str, query: str) -> str:
    """Prepare the prompt using ChatML format"""
    instruction = f"Extract structured information from the HTML content according to this query pattern: {query}"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts structured information from HTML content."},
        {"role": "user", "content": f"{instruction}\n\nInput:\n{html}"}
    ]
    
    # Set ChatML template if not already set
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def generate_response(
    model: Any, 
    tokenizer: Any, 
    html: str, 
    query: str, 
    max_new_tokens: int = 1024
) -> Tuple[str, str]:
    """Generate response from the model using ChatML format"""
    prompt = prepare_chat_prompt(tokenizer, html, query)
    
    # Tokenize with proper padding
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096-max_new_tokens
    )
    
    # Move inputs to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    logger.info("Generating with the following prompt:")
    logger.info("-" * 50)
    logger.info(prompt)
    logger.info("-" * 50)
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        except RuntimeError as e:
            logger.error(f"Error during generation: {str(e)}")
            # Try running on CPU with smaller batch
            logger.info("Retrying with CPU...")
            model = model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Reduce sequence length for CPU
            if inputs["input_ids"].shape[1] > 2048:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
            
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=min(max_new_tokens, 512),  # Reduce tokens for CPU
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            model = model.to(device)
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Full model output:")
    logger.info("-" * 50)
    logger.info(full_response)
    logger.info("-" * 50)
    
    # Extract the assistant's response from the chat
    try:
        response = full_response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
    except IndexError:
        response = full_response.split("<|im_start|>user\n")[-1].split("<|im_end|>")[-1].strip()
    
    return response, full_response

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned scraping model")
    parser.add_argument("--html_file", type=str, required=True, help="Path to the HTML file")
    parser.add_argument("--query", type=str, required=True, help="Query pattern for extraction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model name")
    parser.add_argument("--output_file", type=str, help="Optional path to save the output JSON")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--show_full_output", action="store_true", help="Show full model output including prompt")
    
    args = parser.parse_args()
    
    # Load HTML and clean it
    logger.info(f"Loading and cleaning HTML from {args.html_file}")
    with open(args.html_file, 'r', encoding='utf-8') as f:
        raw_html = f.read()
    cleaned_html = clean_html(raw_html)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.model_path)
    
    # Generate response
    logger.info("Generating response...")
    response, full_response = generate_response(
        model, 
        tokenizer, 
        cleaned_html, 
        args.query, 
        args.max_new_tokens
    )
    
    # Try to parse response as JSON
    try:
        output = json.loads(response)
        if args.output_file:
            # Save both JSON and full output
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            # Save full output to a separate file
            full_output_file = args.output_file.rsplit('.', 1)[0] + '_full.txt'
            with open(full_output_file, 'w', encoding='utf-8') as f:
                f.write(full_response)
                
            logger.info(f"JSON output saved to {args.output_file}")
            logger.info(f"Full output saved to {full_output_file}")
        else:
            if args.show_full_output:
                print("\nFull Model Output:")
                print("-" * 50)
                print(full_response)
                print("-" * 50)
            print("\nParsed JSON Output:")
            print(json.dumps(output, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        logger.error("Failed to parse response as JSON. Raw response:")
        print(full_response)

if __name__ == "__main__":
    main()