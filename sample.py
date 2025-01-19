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
            
        # Check for required files
        required_files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        if missing_files:
            raise ValueError(f"Missing required files in {model_path}: {missing_files}")

        try:
            # Load processor and model
            self.processor = MarkupLMProcessor.from_pretrained(model_path)
            self.processor.parse_html = True
            self.model = MarkupLMForQuestionAnswering.from_pretrained(model_path)
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 Using device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    # ... rest of the class methods remain the same ...

def main():
    try:
        model_path = "./markuplm_amazon_qa_token_lora_final"
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
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
