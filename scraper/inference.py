import torch # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
import json
from playwright.async_api import async_playwright # type: ignore
import time
import asyncio
import nest_asyncio # type: ignore
from scraper.cleanhtml_trial import clean_html # type: ignore
import random
from typing import List
import tiktoken
import argparse  # Add at the top with other imports

# Enable nested event loops
nest_asyncio.apply()

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def get_random_user_agent() -> str:
    """Get a random user agent from a predefined list"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    ]
    return random.choice(user_agents)

class WebScrapingAssistant:
    def __init__(self, model_path="./readerlm-finetuned-ReaderLM"):
        print(f"[INIT] Initializing WebScrapingAssistant with model path: {model_path}")
        # Initialize tokenizer and model
        print("[INIT] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("[INIT] Loading model with Flash Attention 2...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",  # Enable automatic device mapping
            attn_implementation="flash_attention_2",  # Enable Flash Attention 2
        )
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INIT] Using device: {self.device}")
        
        # Set model to evaluation mode
        print("[INIT] Setting model to evaluation mode")
        self.model.eval()
        
        # Define system prompt
        self.system_prompt = """You are an expert web scraping assistant with deep knowledge of HTML parsing and data extraction. 
        Your task is to analyze HTML content and extract structured information based on the given query. 
        You excel at identifying relevant data patterns and returning clean, well-formatted JSON outputs."""

        # Initialize tiktoken encoding
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        print("[INIT] Initialized tiktoken encoding for token counting")
        print("[INIT] WebScrapingAssistant initialization complete")

    async def get_page_content(self, url):
        """Fetch page content using Playwright with enhanced anti-detection"""
        print(f"\n[FETCH] Starting page content fetch for URL: {url}")
        
        print("[FETCH] Initializing Playwright...")
        async with async_playwright() as p:
            browser_args = [
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--disable-infobars',
                '--no-first-run',
                '--enable-webgl',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox'
            ]
            
            print("[FETCH] Launching browser with anti-detection configuration...")
            browser = await p.chromium.launch(
                headless=True,
                args=browser_args
            )
            
            # Enhanced context creation with additional headers
            context = await browser.new_context(
                viewport={'width': random.randint(1280, 1920), 'height': random.randint(800, 1080)},
                user_agent=get_random_user_agent(),
                java_script_enabled=True,
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'DNT': '1',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            print("[FETCH] Creating new page...")
            page = await context.new_page()
            
            try:
                print(f"[FETCH] Navigating to URL: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # Quick random mouse movement (adds minimal delay)
                if random.random() < 0.5:
                    await page.mouse.move(
                        random.randint(100, 500), 
                        random.randint(100, 500)
                    )
                
                # Smart scrolling with minimal delay
                scroll_amount = random.randint(1, 3)
                for _ in range(scroll_amount):
                    await page.evaluate(
                        "window.scrollTo(0, window.scrollY + window.innerHeight * Math.random());"
                    )
                    await asyncio.sleep(0.1)  # Minimal delay between scrolls
                
                print("[FETCH] Getting page content...")
                content = await page.content()
                
                print("[FETCH] Cleaning HTML content...")
                cleaned_content = clean_html(content)
                
                # Count tokens in cleaned content
                token_count = count_tokens(cleaned_content)
                content_length = len(cleaned_content)
                print(f"[FETCH] Content retrieved successfully. Length: {content_length} characters")
                print(f"[FETCH] Cleaned HTML token count: {token_count} tokens")
                return cleaned_content
                
            except Exception as e:
                print(f"[ERROR] Error during page fetch: {str(e)}")
                return None
            finally:
                print("[FETCH] Closing browser...")
                await browser.close()

    def generate_response(self, query, html_content, temperature=0.7):
        print("\n[GENERATE] Starting response generation")
        print(f"[GENERATE] Query length: {len(query)} characters")
        print(f"[GENERATE] HTML content length: {len(html_content)} characters")
        
        # Count tokens in HTML content
        html_tokens = count_tokens(html_content)
        print(f"[GENERATE] HTML content tokens: {html_tokens}")
        
        # Format input like training data
        input_text = f"Query: {query}\n\nHTML Content: {html_content}"
        total_tokens = count_tokens(input_text)
        print(f"[GENERATE] Total input tokens (including query): {total_tokens}")
        
        # Prepare conversation format
        print("[GENERATE] Preparing conversation format...")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        # Apply chat template
        print("[GENERATE] Applying chat template...")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input with optimized settings
        print("[GENERATE] Tokenizing input...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Enable automatic mixed precision
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=False,  # Don't truncate input
                padding=False,
            ).to(self.device)
            
            print(f"[GENERATE] Input shape: {inputs['input_ids'].shape}")
            
            # Generate response with Flash Attention optimizations
            print("[GENERATE] Generating model response...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    temperature=temperature,
                    do_sample=True,
                    max_new_tokens=512,  # Control output length
                    use_cache=True,      # Enable KV-cache
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    top_k=50,            # Limit vocabulary choices
                    top_p=0.95,          # Nucleus sampling
                )
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Decode response
        print("[GENERATE] Decoding response...")
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's generated response
        try:
            print("[GENERATE] Extracting assistant response...")
            response_parts = full_response.split("assistant")
            if len(response_parts) > 1:
                assistant_response = response_parts[-1].strip()
                print("\n[GENERATE] Assistant Generated Output:")
                print("=" * 80)
                print(assistant_response)
                print("=" * 80)
                return {"response": assistant_response}
            else:
                print("[ERROR] No assistant response found in output")
                return {"error": "No assistant response found in output"}
        except Exception as e:
            print(f"[ERROR] Failed to extract assistant response: {str(e)}")
            return {"error": "Failed to extract assistant response"}

    async def extract_data_from_url(self, url, query):
        """Extract data from a URL using the specified query"""
        print(f"\n[EXTRACT] Starting data extraction for URL: {url}")
        try:
            print("[EXTRACT] Fetching HTML content...")
            html_content = await self.get_page_content(url)
            if html_content is None:
                print("[ERROR] Failed to fetch page content")
                return {"error": "Failed to fetch page content"}
            
            print("[EXTRACT] Generating response from HTML content...")
            result = self.generate_response(query, html_content)
            print("[EXTRACT] Data extraction completed")
            return result
        except Exception as e:
            print(f"[ERROR] Extraction failed: {str(e)}")
            return {"error": f"Extraction failed: {str(e)}"}

async def main(model_path: str = "./readerlm-finetuned-ReaderLM"):
    print("\n[MAIN] Starting main execution")
    print(f"[MAIN] Using model path: {model_path}")
    print("[MAIN] Initializing WebScrapingAssistant...")
    assistant = WebScrapingAssistant(model_path)
    
    url = "https://www.amazon.co.uk/s?k=laptops"
    print(f"[MAIN] Target URL: {url}")
    
    query = """
    {
        search_bar,
        product_list[] {
            product_name,
            product_price(include currency symbol),
            product_description,
            product_rating,
            product_reviews_count,
            product_link
        }
    }
    """
    print("[MAIN] Query prepared")
    
    print("[MAIN] Extracting data...")
    result = await assistant.extract_data_from_url(url, query)
    
    print("[MAIN] Saving results to file...")
    with open('laptop_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, indent=2, ensure_ascii=False, fp=f)
    
    print("[MAIN] Printing results...")
    print(json.dumps(result, indent=2))
    print("[MAIN] Execution completed")

if __name__ == "__main__":
    print("[START] Script execution started")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run web scraping inference with specified model')
    parser.add_argument('--model_path', 
                       type=str, 
                       default="./readerlm-finetuned-ReaderLM",
                       help='Path to the fine-tuned model')
    
    args = parser.parse_args()
    
    loop = asyncio.get_event_loop()
    try:
        result = loop.run_until_complete(main(args.model_path))
    except Exception as e:
        print(f"[ERROR] Error during execution: {str(e)}")
    print("[END] Script execution ended") 