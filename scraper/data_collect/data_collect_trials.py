from playwright.async_api import async_playwright
import json
from itertools import combinations
from typing import List, Dict, Any, Optional, Union
import asyncio
from dataclasses import dataclass, field
import re
from copy import deepcopy
import os
import tiktoken
from random import choice, uniform
from scraper.cleanhtml_trial import clean_html  # Import the clean_html function
import logging
from datetime import datetime

@dataclass
class Field:
    name: str
    selector: str
    preprocessing: str = None
    value_type: str = "text"  # can be "text", "attribute", "property"
    attribute_name: str = None  # for value_type="attribute"
    property_name: str = None  # for value_type="property"
    preprocessing_description: str = None  # New field for describing preprocessing

@dataclass
class BaseQuery:
    name: str
    selector: str
    fields: List[Union[Field, 'BaseQuery']]  # Recursive type for nested queries
    is_list: bool = True

@dataclass
class ScrapingConfig:
    url: str
    fields: List[Union[Field, BaseQuery]]
    include_html: bool = True

class TrainingDataGenerator:
    def __init__(self, config: ScrapingConfig):
        print("[DEBUG] Initializing TrainingDataGenerator")
        self.config = config
        self.html_content = ""
        self.data = []
        self.page_elements = {}  # Cache for elements and their values
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Create a unique log file name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/scraping_{timestamp}.log'
        
        # Configure logging with custom formatter to include URL
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(url)s] %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # Create URL adapter for logging
        self.logger = logging.LoggerAdapter(self.logger, {'url': 'Not Set'})

    def set_url(self, url: str):
        """Set the current URL for logging"""
        self.logger.extra['url'] = url

    async def get_full_selector(self, element) -> str:
        print("[DEBUG] Getting full selector for element")
        selectors = []
        current = element
        
        try:
            while current:
                # Get element's tag name
                tag = await current.evaluate("el => el.tagName")
                tag = tag.lower()
                
                # Get element's classes
                classes = await current.evaluate("el => Array.from(el.classList).join('.')")
                if classes:
                    classes = f".{classes}"
                    
                # Get element's id
                element_id = await current.evaluate("el => el.id")
                if element_id:
                    element_id = f"#{element_id}"
                    
                # Get nth-child if needed
                nth = ""
                if not element_id:
                    siblings = await current.evaluate("""el => {
                        let siblings = Array.from(el.parentNode.children);
                        return siblings.indexOf(el) + 1;
                    }""")
                    if siblings > 1:
                        nth = f":nth-child({siblings})"
                        
                selector = f"{tag}{element_id}{classes}{nth}"
                print(f"[DEBUG] Built selector part: {selector}")
                selectors.insert(0, selector)
                
                current = await current.evaluate("el => el.parentElement")
                if current and await current.evaluate("el => el.tagName") == "BODY":
                    break
                
            full_selector = " > ".join(selectors)
            print(f"[DEBUG] Complete selector: {full_selector}")
            return full_selector
            
        except Exception as e:
            print(f"[ERROR] Error in get_full_selector: {str(e)}")
            return None

    async def get_element_html(self, element) -> str:
        """Get the outer HTML of an element"""
        try:
            html = await element.evaluate("el => el.outerHTML")
            return html
        except Exception as e:
            print(f"[ERROR] Error getting element HTML: {str(e)}")
            return None

    async def get_element_value(self, element, field: Field) -> Any:
        self.logger.debug(f"Extracting value for field: {field.name}")
        if not element:
            self.logger.debug("No element found")
            return None

        try:
            # Extract raw value based on value_type
            if field.value_type == "text":
                value = await element.text_content()
                value = value.strip() if value else None
                self.logger.debug(f"Extracted text value: {value}")
            elif field.value_type == "attribute":
                value = await element.get_attribute(field.attribute_name)
                self.logger.debug(f"Extracted attribute value: {value}")
            elif field.value_type == "property":
                value = await element.evaluate(f"el => el.{field.property_name}")
                self.logger.debug(f"Extracted property value: {value}")
            else:
                value = await element.text_content()
                value = value.strip() if value else None
                self.logger.debug(f"Extracted default text value: {value}")

            # Apply preprocessing if specified and value exists
            if field.preprocessing and value:
                try:
                    original_value = value
                    preprocessing_func = eval(field.preprocessing)
                    value = preprocessing_func(value)
                    self.logger.debug(f"Successfully preprocessed value from '{original_value}' to '{value}'")
                except Exception as e:
                    self.logger.warning(f"Preprocessing failed for field {field.name}: {str(e)}")
                    return value
            
            return value
            
        except Exception as e:
            self.logger.error(f"Error in get_element_value for field {field.name}: {str(e)}")
            return None

    def generate_field_combinations(self, fields: List[Union[Field, BaseQuery]]) -> List[List[Union[Field, BaseQuery]]]:
        """Generate all possible combinations of fields, preserving structure"""
        print("[DEBUG] Starting field combination generation")
        
        # Separate base queries and single fields
        base_queries = [q for q in fields if isinstance(q, BaseQuery)]
        single_fields = [f for f in fields if isinstance(f, Field)]
        
        all_combinations = []
        
        # Generate combinations of single fields
        if single_fields:
            for r in range(1, len(single_fields) + 1):
                for combo in combinations(single_fields, r):
                    all_combinations.append(list(combo))
        
        # Add base queries with their field combinations
        for base_query in base_queries:
            # Create combinations of fields within the base query
            inner_fields = base_query.fields
            inner_field_combinations = []
            
            for r in range(1, len(inner_fields) + 1):
                for combo in combinations(inner_fields, r):
                    new_query = deepcopy(base_query)
                    new_query.fields = list(combo)
                    # Add single fields to each base query combination
                    if single_fields:
                        for single_combo in combinations(single_fields, 1):
                            all_combinations.append(list(single_combo) + [new_query])
                    else:
                        all_combinations.append([new_query])
        
        print(f"[DEBUG] Generated {len(all_combinations)} field combinations")
        return all_combinations

    def format_query(self, fields: List[Union[Field, BaseQuery]]) -> str:
        """Format the GraphQL-style query string with support for nested queries"""
        parts = []
        
        for field in fields:
            if isinstance(field, Field):
                # Add preprocessing description in parentheses if available
                field_str = field.name
                if field.preprocessing_description:
                    field_str += f"({field.preprocessing_description})"
                parts.append(field_str)
            else:  # BaseQuery
                if field.is_list:
                    inner_fields = self.format_query(field.fields)
                    parts.append(f"{field.name}[] {{ {inner_fields} }}")
                else:
                    inner_fields = self.format_query(field.fields)
                    parts.append(f"{field.name} {{ {inner_fields} }}")
        
        return ", ".join(parts)

    async def cache_page_elements(self, page):
        """Cache all relevant elements and their values from the page"""
        self.logger.debug("Caching page elements")
        try:
            # Cache single fields
            for field in [f for f in self.config.fields if isinstance(f, Field)]:
                element = await page.query_selector(field.selector)
                if element:
                    value = await self.get_element_value(element, field)
                    element_html = await self.get_element_html(element)
                    self.page_elements[field.name] = {
                        'value': value,
                        'selector': element_html  # Now storing full HTML instead of CSS selector
                    }

            # Cache list elements (like products)
            for query in [q for q in self.config.fields if isinstance(q, BaseQuery)]:
                if query.is_list:
                    elements = await page.query_selector_all(query.selector)
                    max_elements = 24
                    elements = elements[:max_elements]
                    
                    self.page_elements[query.name] = []
                    for element in elements:
                        item_data = {}
                        for field in query.fields:
                            if isinstance(field, Field):
                                field_element = await element.query_selector(field.selector)
                                if field_element:
                                    value = await self.get_element_value(field_element, field)
                                    element_html = await self.get_element_html(field_element)
                                    item_data[field.name] = {
                                        'value': value,
                                        'selector': element_html  # Now storing full HTML instead of CSS selector
                                    }
                        if item_data:
                            self.page_elements[query.name].append(item_data)

            self.logger.debug(f"Cached {len(self.page_elements)} top-level elements")
            
        except Exception as e:
            self.logger.error(f"Error in cache_page_elements: {str(e)}")

    def process_combinations(self):
        """Process field combinations using cached elements"""
        print("[DEBUG] Processing combinations from cached elements")
        field_combinations = self.generate_field_combinations(self.config.fields)
        
        for idx, field_combo in enumerate(field_combinations, 1):
            print(f"[DEBUG] Processing combination {idx}/{len(field_combinations)}")
            output_data = {
                "query": "{ " + self.format_query(field_combo) + " }",
                "output": {},
                "html": self.html_content
            }

            for field in field_combo:
                if isinstance(field, Field):
                    if field.name in self.page_elements:
                        cached = self.page_elements[field.name]
                        output_data["output"][field.name] = cached['value']
                        output_data["output"][f"{field.name}_element"] = cached['selector']
                
                elif isinstance(field, BaseQuery) and field.name in self.page_elements:
                    if field.is_list:
                        output_data["output"][field.name] = []
                        output_data["output"][f"{field.name}_element"] = field.selector
                        
                        for item in self.page_elements[field.name]:
                            product_data = {}
                            for subfield in field.fields:
                                if subfield.name in item:
                                    cached = item[subfield.name]
                                    product_data[subfield.name] = cached['value']
                                    product_data[f"{subfield.name}_element"] = cached['selector']
                            if product_data:
                                output_data["output"][field.name].append(product_data)

            if output_data["output"]:
                self.data.append(output_data)

    async def scrape_page(self, page):
        print(f"[INFO] Starting page scraping")  # Keep this print
        try:
            raw_html = await page.content()
            self.html_content = clean_html(raw_html)
            self.logger.debug("HTML content cleaned and filtered")
            
            await self.cache_page_elements(page)
            self.process_combinations()
            
            print(f"[INFO] Page scraping completed. Total data points: {len(self.data)}")  # Keep this print
            
        except Exception as e:
            self.logger.error(f"Error in scrape_page: {str(e)}")
            self.logger.error(traceback.format_exc())

    def save_to_file(self, filename: str, append: bool = True):
        """Save the scraped data to a JSON file with proper handling of lambda functions"""
        try:
            processed_data = []
            
            # Read existing data if appending and file exists
            if append and os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        processed_data = json.load(f)
                    print(f"[DEBUG] Successfully loaded {len(processed_data)} existing entries from {filename}")
                except json.JSONDecodeError:
                    print(f"[WARNING] Could not read existing file {filename}, starting fresh")
                    processed_data = []
            
            # Process new data
            for item in self.data:
                processed_item = {
                    "query": item["query"],
                    "output": {},
                    "html": self.html_content
                }
                
                # Find all BaseQuery names from config
                base_query_names = {
                    query.name for query in self.config.fields 
                    if isinstance(query, BaseQuery)
                }
                
                # Process the output data
                for base_query_name in base_query_names:
                    if base_query_name in item["output"]:
                        # Add the base query selector
                        base_query = next(
                            (query for query in self.config.fields 
                             if isinstance(query, BaseQuery) and query.name == base_query_name),
                            None
                        )
                        if base_query:
                            processed_item["output"][f"{base_query_name}_element"] = base_query.selector
                            processed_item["output"][base_query_name] = []
                            
                            for product in item["output"][base_query_name]:
                                processed_product = {}
                                field_names = set()
                                for key in product.keys():
                                    base_name = key.replace("_element", "")
                                    field_names.add(base_name)
                                
                                for base_name in field_names:
                                    value = product.get(base_name)
                                    element = product.get(f"{base_name}_element")
                                    processed_product[base_name] = value
                                    processed_product[f"{base_name}_element"] = element
                                
                                processed_item["output"][base_query_name].append(processed_product)
                
                # Handle single fields
                single_fields = {
                    key for key in item["output"].keys() 
                    if not key.endswith("_element") and key not in base_query_names
                }
                for base_name in single_fields:
                    value = item["output"].get(base_name)
                    element = item["output"].get(f"{base_name}_element")
                    processed_item["output"][base_name] = value
                    processed_item["output"][f"{base_name}_element"] = element
                
                processed_data.append(processed_item)
            
            # Save all data back to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Successfully saved {len(processed_data)} entries to {filename}")
            
        except Exception as e:
            print(f"[ERROR] Error saving to file: {str(e)}")
            import traceback
            print("[ERROR] Full traceback:")
            print(traceback.format_exc())

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"[ERROR] Error counting tokens: {str(e)}")
            return 0

    def get_total_tokens(self) -> dict:
        """Calculate token counts for HTML and fields"""
        try:
            html_tokens = self.count_tokens(self.html_content)
            
            # Count tokens for each unique field combination
            field_combinations_tokens = {}
            combined_tokens = {}  # New dict for HTML + field tokens
            for item in self.data:
                # Get the query as key for the combination
                query = item["query"]
                if query not in field_combinations_tokens:
                    field_combinations_tokens[query] = 0
                
                # Convert output data to string to count tokens
                output_str = json.dumps(item["output"], ensure_ascii=False)
                combo_tokens = self.count_tokens(output_str)
                field_combinations_tokens[query] = combo_tokens
                combined_tokens[query] = combo_tokens + html_tokens  # Combined tokens for this combination
            
            # Calculate total field tokens
            total_field_tokens = sum(field_combinations_tokens.values())
            
            return {
                "html_tokens": html_tokens,
                "field_tokens": total_field_tokens,
                "total_tokens": html_tokens + total_field_tokens,
                "field_combinations": field_combinations_tokens,
                "combined_tokens": combined_tokens  # Add combined tokens to return value
            }
        except Exception as e:
            print(f"[ERROR] Error calculating total tokens: {str(e)}")
            return {
                "html_tokens": 0, 
                "field_tokens": 0, 
                "total_tokens": 0,
                "field_combinations": {},
                "combined_tokens": {}
            }

# List of common user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/121.0.0.0',
]

async def process_url(url: str, generator: TrainingDataGenerator, output_file: str):
    """Process a single URL and append results to the output file"""
    print(f"\n[INFO] Processing URL: {url}")  # Keep this print
    
    # Set the URL for logging
    generator.set_url(url)
    
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--no-default-browser-check",
                    "--disable-infobars",
                    "--no-first-run",
                    "--enable-webgl",
                    "--disable-dev-shm-usage",
                ]
            )
            
            # Enhanced context configuration
            context = await browser.new_context(
                user_agent=choice(USER_AGENTS),
                viewport={'width': 1920, 'height': 1080},
                device_scale_factor=1,
                java_script_enabled=True,
                has_touch=False,
                is_mobile=False,
                locale='en-GB',
                timezone_id='Europe/London',
                permissions=['geolocation'],
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Sec-Ch-Ua': '"Chromium";v="121", "Not A Brand";v="99"',
                    'Sec-Ch-Ua-Platform': '"Windows"',
                }
            )
            
            page = await context.new_page()
            
            # Stealth configurations
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => false});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """)
            
            try:
                # Set reasonable timeout
                page.set_default_timeout(30000)
                
                # Navigate with enhanced error handling
                response = await page.goto(url, wait_until='load', timeout=30000)
                
                if response.status != 200:
                    print(f"[WARNING] Received non-200 status code: {response.status}")
                    return
                
                # Efficient scroll sequence with random delays
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight/2)")
                await page.wait_for_timeout(uniform(500, 1000))
                
                # Random scroll positions with minimal delays
                scroll_positions = [int(uniform(0, 1000)) for _ in range(3)]
                for pos in scroll_positions:
                    await page.evaluate(f"window.scrollTo(0, {pos})")
                    await page.wait_for_timeout(uniform(200, 400))
                
                # Quick random mouse movement
                await page.mouse.move(
                    x=int(uniform(100, 800)),
                    y=int(uniform(100, 600))
                )
                
                # Scrape the page
                await asyncio.wait_for(generator.scrape_page(page), timeout=300.0)
                generator.save_to_file(output_file, append=True)
                
                # Token counting and stats - keep these prints
                token_counts = generator.get_total_tokens()
                print(f"\n[TOKEN STATS for {url}]")
                print(f"HTML Tokens: {token_counts['html_tokens']:,}")
                print(f"Total Field Tokens: {token_counts['field_tokens']:,}")
                print(f"Total Tokens: {token_counts['total_tokens']:,}")
                print("\nToken counts for each field combination:")
                print("-" * 80)
                for query, tokens in token_counts['field_combinations'].items():
                    print(f"Fields: {query}")
                    print(f"Field Tokens: {tokens:,}")
                    print(f"Combined with HTML: {token_counts['combined_tokens'][query]:,}")
                    print("-" * 80)
                print()
                
            except Exception as e:
                generator.logger.error(f"Error processing URL {url}: {str(e)}")
                generator.logger.error(traceback.format_exc())
            finally:
                await context.close()
                await browser.close()
                
        except Exception as e:
            generator.logger.error(f"Failed to initialize browser for URL {url}: {str(e)}")
            generator.logger.error(traceback.format_exc())

async def main():
    print("[DEBUG] Starting multi-URL scraping process...")
    
    output_file = "datasets/trial_dataset.json"
    
    # Delete the output file if it exists
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            print(f"[DEBUG] Deleted existing output file: {output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to delete output file: {str(e)}")
            return
    # List of URLs to scrape
    urls = [
         "https://www.amazon.co.uk/s?k=laptops",
         "https://www.amazon.co.uk/s?k=monitors", 
         "https://www.amazon.co.uk/s?k=keyboards",
         "https://www.amazon.co.uk/s?k=mice",
         "https://www.amazon.co.uk/s?k=headphones",
         "https://www.amazon.co.uk/s?k=speakers",
         "https://www.amazon.co.uk/s?k=webcams",
         "https://www.amazon.co.uk/s?k=microphones",
         "https://www.amazon.co.uk/s?k=gaming+chairs",
         "https://www.amazon.co.uk/s?k=gaming+desks",
         "https://www.amazon.co.uk/s?k=mousepads",
         "https://www.amazon.co.uk/s?k=computer+cases",
         "https://www.amazon.co.uk/s?k=power+supplies",
         "https://www.amazon.co.uk/s?k=graphics+cards",
         "https://www.amazon.co.uk/s?k=processors",
         "https://www.amazon.co.uk/s?k=motherboards",
        # Add more URLs as needed
    ]
        
    for url in urls:
        config = ScrapingConfig(
            url=url,
            fields=[
                Field(
                    name="search_bar",
                    selector='input[id="twotabsearchtextbox"]',
                    value_type="attribute",
                    attribute_name="value"
                ),
                BaseQuery(
                    name="products",
                    selector="div[data-component-type='s-search-result']:not([data-component-id=''])",
                    fields=[
                        Field(
                            name="product_name",
                            selector="h2 > span",
                        ),
                        Field(
                            name="product_price",
                            selector='a span[class="a-offscreen"]',
                        ),
                        Field(
                            name="product_image_url",
                            selector='a img',
                            value_type="attribute",
                            attribute_name="src",
                            preprocessing_description="full image URL"
                        ),
                        Field(
                            name="product_link",
                            selector="a",
                            value_type="attribute",
                            attribute_name="href",
                            preprocessing="lambda x: 'https://www.amazon.co.uk' + x if x.startswith('/') else x",
                            preprocessing_description="complete URL"
                        ),
                        Field(
                            name="product_rating",
                            selector="span.a-icon-alt",
                            preprocessing="lambda x: float(x.split(' ')[0]) if x else None",
                            preprocessing_description="only the rating"
                        ),
                        Field(
                            name="product_reviews_count",
                            selector="span.a-size-base.s-underline-text",
                            preprocessing="lambda x: int(x.replace(',', '')) if x and x.replace(',', '').isdigit() else None",
                            preprocessing_description="numeric value only"
                        )
                    ]
                )
            ]
        )
        
        generator = TrainingDataGenerator(config)
        await process_url(url, generator, output_file)
        print(f"[DEBUG] Completed processing URL: {url}")
    
    print("[DEBUG] All URLs processed successfully")

if __name__ == "__main__":
    import traceback
    print("[DEBUG] Script started")
    asyncio.run(main())
    print("[DEBUG] Script completed")