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

def load_model_and_tokenizer(model_path: str, base_model: str = "mistralai/Mistral-7B-v0.1") -> tuple:
    """Load the fine-tuned model and tokenizer"""
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    logger.info(f"Loading adapter from: {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer

def prepare_chat_prompt(tokenizer: Any, html: str, query: str) -> str:
    """Prepare the prompt using the chat template"""
    instruction = f"Extract structured information from the HTML content according to this query pattern: {query}"
    
    messages = [
        {"role": "user", "content": f"{instruction}\n\nInput:\n{html}"},
    ]
    
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
    """Generate response from the model using chat template"""
    prompt = prepare_chat_prompt(tokenizer, html, query)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096-max_new_tokens)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    logger.info("Generating with the following prompt:")
    logger.info("-" * 50)
    logger.info(prompt)
    logger.info("-" * 50)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Full model output:")
    logger.info("-" * 50)
    logger.info(full_response)
    logger.info("-" * 50)
    
    # Extract the assistant's response from the chat
    response = full_response.split("</s>")[-1].strip()
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
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
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