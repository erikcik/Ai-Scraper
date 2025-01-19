import json
import random
import torch
from time import sleep
from transformers import MarkupLMProcessor, MarkupLMForQuestionAnswering
from playwright.sync_api import sync_playwright
import os

class MarkupLMExtractor:
    def __init__(self, model_path):
        print("🔄 Loading model and processor...")
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist!")
            
        # Load processor and model
        try:
            self.processor = MarkupLMProcessor.from_pretrained(model_path)
            self.processor.parse_html = True
            self.model = MarkupLMForQuestionAnswering.from_pretrained(model_path)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 Using device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def fetch_html(self, url):
        """Fetch HTML content using Playwright with basic anti-bot handling."""
        max_retries = 3
        html = None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/121.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                permissions=['geolocation']
            )
            context.add_cookies([
                {'name': 'session-id', 'value': '123-1234567-1234567', 'domain': '.amazon.com', 'path': '/'},
                {'name': 'i18n-prefs', 'value': 'USD', 'domain': '.amazon.com', 'path': '/'},
                {'name': 'sp-cdn', 'value': 'L5Z9:US', 'domain': '.amazon.com', 'path': '/'}
            ])
            page = context.new_page()
            page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1'
            })

            for attempt in range(1, max_retries+1):
                try:
                    print(f"\n🌐 Fetching URL: {url} (Attempt {attempt})")
                    page.goto(url, timeout=90000, wait_until='domcontentloaded')
                    for _ in range(3):
                        page.evaluate("window.scrollBy(0, window.innerHeight)")
                        page.wait_for_timeout(random.randint(800, 1200))
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(random.randint(800, 1200))
                    html = page.content()
                    if "Robot Check" in html or "Enter the characters you see below" in html:
                        print("⚠️ CAPTCHA detected. Retrying...")
                        sleep(random.uniform(2,4))
                        continue
                    return html
                except Exception as e:
                    print(f"❌ Error on attempt {attempt}: {e}")
                    sleep(random.uniform(2,4))
                    continue
            browser.close()
            print("Failed to fetch HTML after several attempts.")
            return None

    def process_url(self, url, query_text):
        html_content = self.fetch_html(url)
        if html_content is None:
            return None

        print("\n🔍 Processing query with MarkupLM model...")
        encoding = self.processor(
            html_strings=html_content,
            questions=query_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        input_ids = encoding["input_ids"][0]
        answer_ids = input_ids[start_idx : end_idx + 1]
        answer_text = self.processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
        return {
            "input": {
                "text": query_text,
                "html": html_content
            },
            "output": answer_text
        }

def main():
    model_path = "./markuplm_amazon_qa_token_lora_final"
    
    # Verify model files exist
    required_files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        print(f"Missing required model files: {missing_files}")
        return
        
    try:
        extractor = MarkupLMExtractor(model_path)
        
        url = ("https://www.amazon.com/s?k=faber+castell+colored+pencils&"
               "crid=ADRA090J7SD4&sprefix=%2Caps%2C211&ref=nb_sb_ss_recent_2_0_recent")
        query_text = """{
          products[] {
            product_price
          }
        }"""
        
        results = extractor.process_url(url, query_text)
        if results:
            print("\n✅ Results:")
            print(json.dumps(results, indent=2))
            with open('results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("\n💾 Results saved to 'results.json'")
        else:
            print("No results obtained.")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
