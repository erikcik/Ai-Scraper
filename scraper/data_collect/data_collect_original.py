from playwright.async_api import async_playwright
import json
from itertools import combinations
from typing import List, Dict, Any, Optional, Union
import asyncio
from dataclasses import dataclass, field
import re
from copy import deepcopy
import os

@dataclass
class Field:
    name: str
    selector: str
    preprocessing: str = None
    value_type: str = "text"  # can be "text", "attribute", "property"
    attribute_name: str = None  # for value_type="attribute"
    property_name: str = None  # for value_type="property"

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

    async def get_element_value(self, element, field: Field) -> Any:
        print(f"[DEBUG] Extracting value for field: {field.name}")
        if not element:
            print("[DEBUG] No element found")
            return None

        try:
            # Extract raw value based on value_type
            if field.value_type == "text":
                value = await element.text_content()
                value = value.strip() if value else None
                print(f"[DEBUG] Extracted text value: {value}")
            elif field.value_type == "attribute":
                value = await element.get_attribute(field.attribute_name)
                print(f"[DEBUG] Extracted attribute value: {value}")
            elif field.value_type == "property":
                value = await element.evaluate(f"el => el.{field.property_name}")
                print(f"[DEBUG] Extracted property value: {value}")
            else:
                value = await element.text_content()
                value = value.strip() if value else None
                print(f"[DEBUG] Extracted default text value: {value}")

            # Apply preprocessing if specified and value exists
            if field.preprocessing and value:
                try:
                    original_value = value
                    # Create the preprocessing function
                    preprocessing_func = eval(field.preprocessing)
                    # Execute the preprocessing function
                    value = preprocessing_func(value)
                    print(f"[DEBUG] Successfully preprocessed value from '{original_value}' to '{value}'")
                except Exception as e:
                    print(f"[WARNING] Preprocessing failed for field {field.name}: {str(e)}")
                    # Return the original value if preprocessing fails
                    return value
            
            return value
            
        except Exception as e:
            print(f"[ERROR] Error in get_element_value for field {field.name}: {str(e)}")
            return None

    def generate_field_combinations(self, fields: List[Union[Field, BaseQuery]]) -> List[List[Union[Field, BaseQuery]]]:
        """This method is deprecated and will be removed. Keeping for backwards compatibility."""
        return [fields]  # Return only the complete set of fields

    def format_query(self, fields: List[Union[Field, BaseQuery]]) -> str:
        """Format the GraphQL-style query string with support for nested queries"""
        parts = []
        
        for field in fields:
            if isinstance(field, Field):
                parts.append(field.name)
            else:  # BaseQuery
                if field.is_list:
                    inner_fields = self.format_query(field.fields)
                    parts.append(f"{field.name}[] {{ {inner_fields} }}")
                else:
                    inner_fields = self.format_query(field.fields)
                    parts.append(f"{field.name} {{ {inner_fields} }}")
        
        return ", ".join(parts)

    async def scrape_field(self, page, field: Field, container=None) -> tuple:
        print(f"[DEBUG] Scraping field: {field.name}")
        try:
            element = await (container or page).query_selector(field.selector)
            if not element:
                print(f"[DEBUG] No element found for selector: {field.selector}")
                return None, field.selector  # Return original selector even if element not found
                
            print(f"[DEBUG] Element found for field: {field.name}")
            value = await self.get_element_value(element, field)
            element_selector = await self.get_full_selector(element)
            
            # If we couldn't get the full selector, use the original one
            if not element_selector:
                element_selector = field.selector
                
            print(f"[DEBUG] Field {field.name} scraped - Value: {value}, Selector: {element_selector}")
            return value, element_selector
            
        except Exception as e:
            print(f"[ERROR] Error scraping field {field.name}: {str(e)}")
            return None, field.selector  # Return original selector on error

    async def scrape_base_query(self, page, query: BaseQuery, container=None) -> Dict:
        print(f"[DEBUG] Processing base query: {query.name}")
        result = {}
        
        try:
            if query.is_list:
                result[query.name] = []
                result[f"{query.name}_element"] = query.selector  # Store the base query selector
                elements = await (container or page).query_selector_all(query.selector)
                print(f"[DEBUG] Found {len(elements)} elements for query: {query.name}")
                
                # Limit the number of elements to process to avoid timeouts
                max_elements = 24  # Process at most 24 products
                elements = elements[:max_elements]
                
                for idx, element in enumerate(elements):
                    print(f"[DEBUG] Processing element {idx + 1}/{len(elements)} for {query.name}")
                    item_data = {}
                    
                    try:
                        for field in query.fields:
                            if isinstance(field, Field):
                                value, selector = await asyncio.wait_for(
                                    self.scrape_field(page, field, element),
                                    timeout=5.0  # 5 second timeout per field
                                )
                                # Store both value and element selector
                                item_data[field.name] = value
                                item_data[f"{field.name}_element"] = selector if selector else field.selector
                            else:  # Nested BaseQuery
                                nested_data = await asyncio.wait_for(
                                    self.scrape_base_query(page, field, element),
                                    timeout=10.0  # 10 second timeout for nested queries
                                )
                                item_data.update(nested_data)
                    except asyncio.TimeoutError:
                        print(f"[WARNING] Timeout processing element {idx + 1}")
                        continue
                    except Exception as e:
                        print(f"[ERROR] Error processing element {idx + 1}: {str(e)}")
                        continue
                    
                    if item_data:
                        result[query.name].append(item_data)
                        print(f"[DEBUG] Added data for element {idx + 1}")
                
                print(f"[DEBUG] Completed processing {len(result[query.name])} items for {query.name}")
                
            else:
                print(f"[DEBUG] Processing single item query: {query.name}")
                result[f"{query.name}_element"] = query.selector  # Store the base query selector
                element = await (container or page).query_selector(query.selector)
                if element:
                    for field in query.fields:
                        try:
                            if isinstance(field, Field):
                                value, selector = await asyncio.wait_for(
                                    self.scrape_field(page, field, element),
                                    timeout=5.0
                                )
                                # Store both value and element selector
                                result[field.name] = value
                                result[f"{field.name}_element"] = selector if selector else field.selector
                            else:
                                nested_data = await asyncio.wait_for(
                                    self.scrape_base_query(page, field, element),
                                    timeout=10.0
                                )
                                result.update(nested_data)
                        except asyncio.TimeoutError:
                            print(f"[WARNING] Timeout processing field {field.name}")
                            continue
                        except Exception as e:
                            print(f"[ERROR] Error processing field {field.name}: {str(e)}")
                            continue
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error in scrape_base_query for {query.name}: {str(e)}")
            return result

    async def scrape_page(self, page):
        print("[DEBUG] Starting page scraping")
        try:
            if self.config.include_html:
                self.html_content = await page.content()
                print("[DEBUG] HTML content captured")
            
            # Instead of generating combinations, we'll collect all data in one pass
            output_data = {
                "query": "{ " + self.format_query(self.config.fields) + " }",
                "output": {},
                "html": self.html_content if self.config.include_html else None
            }
            
            try:
                for field in self.config.fields:
                    if isinstance(field, Field):
                        value, selector = await asyncio.wait_for(
                            self.scrape_field(page, field),
                            timeout=5.0
                        )
                        if value is not None:
                            output_data["output"][field.name] = value
                            output_data["output"][f"{field.name}_element"] = selector
                    else:
                        result = await asyncio.wait_for(
                            self.scrape_base_query(page, field),
                            timeout=60.0
                        )
                        output_data["output"].update(result)
                
                if output_data["output"]:
                    self.data.append(output_data)
                    print(f"[DEBUG] Data collection completed successfully")
                
            except asyncio.TimeoutError:
                print(f"[WARNING] Timeout while processing page")
            except Exception as e:
                print(f"[ERROR] Error processing page: {str(e)}")
            
            print(f"[DEBUG] Page scraping completed. Total data points: {len(self.data)}")
            
        except Exception as e:
            print(f"[ERROR] Error in scrape_page: {str(e)}")
            import traceback
            print("[ERROR] Full traceback:")
            print(traceback.format_exc())

    def save_to_file(self, filename: str, append: bool = True):
        """Save the scraped data to a JSON file"""
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
            
            # Process new data - now much simpler since we're not dealing with combinations
            processed_data.extend(self.data)
            
            # Save all data back to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] Successfully saved {len(processed_data)} entries to {filename}")
            
        except Exception as e:
            print(f"[ERROR] Error saving to file: {str(e)}")
            import traceback
            print("[ERROR] Full traceback:")
            print(traceback.format_exc())

async def process_url(url: str, generator: TrainingDataGenerator, output_file: str):
    """Process a single URL and append results to the output file"""
    async with async_playwright() as p:
        print(f"[DEBUG] Processing URL: {url}")
        try:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                device_scale_factor=1,
                java_script_enabled=True,
                has_touch=False,
                is_mobile=False,
                locale='en-GB',
            )
            page = await context.new_page()
            
            try:
                page.set_default_timeout(60000)
                response = await page.goto(url, wait_until='load', timeout=60000)
                
                if response.status != 200:
                    print(f"[WARNING] Received non-200 status code: {response.status}")
                    return
                
                await page.wait_for_selector("div[data-component-type='s-search-result']", timeout=30000)
                products = await page.query_selector_all("div[data-component-type='s-search-result']")
                print(f"[DEBUG] Found {len(products)} products on the page")
                
                # Scroll sequence
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(1000)
                
                for i in range(10):
                    await page.evaluate(f"window.scrollTo(0, {(i + 1) * 1000})")
                    await page.wait_for_timeout(500)
                
                await page.wait_for_timeout(2000)
                
                # Scrape the page
                await asyncio.wait_for(generator.scrape_page(page), timeout=300.0)
                
                # Save data immediately after scraping each URL
                if generator.data:
                    # Read existing data
                    existing_data = []
                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                            print(f"[DEBUG] Loaded {len(existing_data)} existing entries")
                        except json.JSONDecodeError:
                            print(f"[WARNING] Could not read {output_file}, starting fresh")
                    
                    # Append new data
                    existing_data.extend(generator.data)
                    
                    # Save combined data
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    print(f"[DEBUG] Saved {len(generator.data)} new entries to {output_file}")
                    
                    # Clear generator data for next URL
                    generator.data = []
                
            except Exception as e:
                print(f"[ERROR] Error processing URL {url}: {str(e)}")
                traceback.print_exc()
            finally:
                await context.close()
                await browser.close()
                
        except Exception as e:
            print(f"[ERROR] Failed to initialize browser for URL {url}: {str(e)}")
            traceback.print_exc()

async def main():
    print("[DEBUG] Starting multi-URL scraping process...")
    
    # List of URLs to scrape
       # List of URLs to scrape
    urls = [
        # Search pages (commented out for now)
        # "https://www.amazon.co.uk/s?k=laptops",
        # "https://www.amazon.co.uk/s?k=monitors",
        # "https://www.amazon.co.uk/s?k=keyboards",
        # "https://www.amazon.co.uk/s?k=mice",
        # "https://www.amazon.co.uk/s?k=headphones",
        # "https://www.amazon.co.uk/s?k=speakers",
        # "https://www.amazon.co.uk/s?k=webcams",
        # "https://www.amazon.co.uk/s?k=microphones",
        # "https://www.amazon.co.uk/s?k=gaming+chairs",
        # "https://www.amazon.co.uk/s?k=gaming+desks",
        # "https://www.amazon.co.uk/s?k=mousepads",
        # "https://www.amazon.co.uk/s?k=computer+cases",
        # "https://www.amazon.co.uk/s?k=power+supplies",
        # "https://www.amazon.co.uk/s?k=graphics+cards",
        # "https://www.amazon.co.uk/s?k=processors",
        # "https://www.amazon.co.uk/s?k=motherboards",
        # Product detail pages
        "https://www.amazon.co.uk/LG-Electronics-24MR400-B-FreeSync-Anti-Glare/dp/B0CPGPF6LD",
        "https://www.amazon.co.uk/Logitech-Business-Keyboard-Windows-Linux/dp/B003ZY9Z40",
        "https://www.amazon.co.uk/Logitech-Wireless-Keyboard-Windows-Connection/dp/B00CL6353A",
        "https://www.amazon.co.uk/Sony-WH-CH720N-Cancelling-Bluetooth-Headphones-Black/dp/B0BTDX26B2",
        "https://www.amazon.co.uk/Sony-MDRZX110B-AE-Headphones-Black/dp/B00NBR70DO",
        "https://www.amazon.co.uk/JBL-Headphones-Microphone-Cancelling-Hands-Free-Black/dp/B096FYLJ6M",
        "https://www.amazon.co.uk/Razer-DeathAdder-Essential-programmable-mechanical/dp/B092R5MCB3",
        "https://www.amazon.co.uk/Bluetooth-Headphones-microphone-Earphones-Waterproof-Black/dp/B0D452MJV5",
        "https://www.amazon.co.uk/Noise-Cancelling-Headphones-Bluetooth-Microphone-1-Black-gold/dp/B0C2CNHX82",
        "https://www.amazon.co.uk/Amazon-Basics-Condenser-Microphone-Podcasting/dp/B0CL9BTQRF",
        "https://www.amazon.co.uk/Microphone-Veetop-Condenser-Podcasting-Compatible-Black/dp/B08YK68849",
        "https://www.amazon.co.uk/Fede-Microphone-Wireless-Bluetooth-Flashing/dp/B08CRHB6NM",
        "https://www.amazon.co.uk/office-L-Shaped-gaming-Corner-Storage/dp/B0DHK6SBBR",
        "https://www.amazon.co.uk/AEX-Stainless-Silverware-Heavy-duty-Dishwasher/dp/B098L7H3T1"
    ]
    
    output_file = "datasets/initial_dataset.json"
    
    for url in urls:
        if not url:  # Skip empty URLs
            continue
            
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
                    name="product_list",
                    selector="div[data-component-type='s-search-result']:not([data-component-id=''])",
                    fields=[
                        Field(
                            name="product_name",
                            selector="h2",
                        ),
                        Field(
                            name="product_price",
                            selector='a span[class="a-offscreen"]',
                        ),
                        Field(
                            name="product_image_url",
                            selector='a img',
                            value_type="attribute",
                            attribute_name="src"
                        ),
                        Field(
                            name="product_link",
                            selector="a",
                            value_type="attribute",
                            attribute_name="href",
                            preprocessing="lambda x: 'https://www.amazon.co.uk' + x if x.startswith('/') else x"
                        ),
                        Field(
                            name="product_rating",
                            selector="span.a-icon-alt",
                            preprocessing="lambda x: float(x.split(' ')[0]) if x else None"
                        ),
                        Field(
                            name="product_reviews_count",
                            selector="span.a-size-base.s-underline-text",
                            preprocessing="lambda x: int(x.replace(',', '')) if x and x.replace(',', '').isdigit() else None"
                        )
                    ]
                )
            ]
        )
        
        generator = TrainingDataGenerator(config)
        await process_url(url, generator, output_file)
        print(f"[DEBUG] Completed processing URL: {url}")
        await asyncio.sleep(2)  # Add a small delay between URLs to avoid rate limiting
    
    print("[DEBUG] All URLs processed successfully")

if __name__ == "__main__":
    import traceback
    print("[DEBUG] Script started")
    asyncio.run(main())
    print("[DEBUG] Script completed")
