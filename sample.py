from transformers import MarkupLMProcessor, MarkupLMForQuestionAnswering
import torch
from playwright.sync_api import sync_playwright
import json

class MarkupLMExtractor:
    def __init__(self, model_path):
        print("🔄 Loading model and processor...")
        self.processor = MarkupLMProcessor.from_pretrained("microsoft/markuplm-base")
        self.processor.parse_html = True
        
        self.model = MarkupLMForQuestionAnswering.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def fetch_html(self, url):
        """
        Fetch HTML content using Playwright with improved anti-bot handling
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                geolocation={'latitude': 40.7128, 'longitude': -74.0060},
                permissions=['geolocation']
            )
            
            # Add multiple cookies for better authenticity
            context.add_cookies([
                {
                    'name': 'session-id',
                    'value': '123-1234567-1234567',
                    'domain': '.amazon.com',
                    'path': '/'
                },
                {
                    'name': 'i18n-prefs',
                    'value': 'USD',
                    'domain': '.amazon.com',
                    'path': '/'
                },
                {
                    'name': 'sp-cdn',
                    'value': 'L5Z9:US',
                    'domain': '.amazon.com',
                    'path': '/'
                }
            ])
            
            page = context.new_page()
            
            # Set more realistic headers
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

            try:
                print(f"\n🌐 Fetching URL: {url}")
                page.goto(url, timeout=90000, wait_until='domcontentloaded')

                for _ in range(3):
                    page.evaluate('window.scrollBy(0, window.innerHeight)')
                    page.wait_for_timeout(1000)

                page.evaluate('window.scrollTo(0, 0)')
                page.wait_for_timeout(1000)
                
                html_content = page.content()
                
                if 'Robot Check' in html_content or 'Enter the characters you see below' in html_content:
                    print("Warning: Possible CAPTCHA detected")
                    return None
                    
                return html_content
                
            except Exception as e:
                print(f"Error fetching URL {url}: {e}")
                return None
            finally:
                browser.close()

    def process_url(self, url, query_text):
        """Process URL with the trained model using a question."""
        html_content = self.fetch_html(url)
        if not html_content:
            return None

        print("\n🔍 Processing query with MarkupLM model...")

        # Encode using the processor
        encoding = self.processor(
            html_strings=html_content,
            questions=query_text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        # Move tensors to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()

            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            input_ids = encoding['input_ids'][0]
            answer_tokens = input_ids[start_idx : end_idx + 1]

            # Decode using the processor’s tokenizer
            answer = self.processor.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return {
            "input": {
                "text": query_text,
                "html": html_content
            },
            "output": answer
        }

def main():
    extractor = MarkupLMExtractor("./markuplm_amazon_qa_final")
    
    url = "https://www.amazon.com/s?k=laptop"
    
    query_text = """{
      products[] {
        product_price
      }
      search_bar
    }"""
    
    results = extractor.process_url(url, query_text)
    
    if results:
        print("\n✅ Results:")
        print(json.dumps(results, indent=2))

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n💾 Results saved to 'results.json'")

if __name__ == "__main__":
    main()