from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import json
import os
from itertools import combinations

class HTMLContentScraper:
    def __init__(self, output_file="dataset.json"):
        self.output_file = output_file
        # Only create the file if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w") as file:
                json.dump([], file)

    def parse_html(self, html_content, query):
        """
        Parse the HTML and extract elements and values based on the input query.
        - html_content: The raw HTML content as a string.
        - query: A dictionary of queries to extract specific fields or elements.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        results = {}

        for key, value in query.items():
            if key.endswith("[]"):  # Handle array-like structures
                field_name = key[:-2]  # Remove the "[]" from the key
                item_query = value
                elements = soup.select(item_query["selector"])

                results[field_name] = []
                for element in elements:
                    item_result = {}
                    for sub_key, sub_value in item_query.items():
                        if sub_key != "selector":  # Skip the "selector" key
                            sub_element = element.select_one(sub_value["selector"])
                            item_result[sub_key] = {
                                "value": sub_element.get_text(strip=True) if sub_element else None,
                                "element": str(sub_element) if sub_element else None,
                            }
                    results[field_name].append(item_result)
            else:  # Handle single-value fields
                if "selector" in value:
                    element = soup.select_one(value["selector"])
                    results[key] = {
                        "value": element.get_text(strip=True) if element else None,
                        "element": str(element) if element else None,
                    }
                else:
                    results[key] = None

        return results

    def fetch_html(self, url):
        """
        Fetch HTML content using Playwright to bypass CAPTCHA.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            try:
                page.goto(url, timeout=60000)
                page.wait_for_timeout(5000)
                html_content = page.content()
                return html_content
            except Exception as e:
                print(f"Error fetching URL {url}: {e}")
            finally:
                browser.close()
        return None

    def save_to_file(self, data, html_content, input_query):
        """
        Save data in MarkupLM-compatible format (prettified HTML) with dynamic fields.
        """
        # Prettify the HTML for cleaner MarkupLM usage
        soup = BeautifulSoup(html_content, "html.parser")
        prettified_html = soup.prettify()

        def format_query_text(query):
            text_parts = []
            for key, value in query.items():
                if key.endswith("[]"):
                    field_name = key[:-2]
                    fields = [k for k in value.keys() if k != "selector"]
                    if fields:
                        field_text = f"  {field_name}[] {{\n"
                        field_text += "\n".join(f"    {field}" for field in fields)
                        field_text += "\n  }"
                        text_parts.append(field_text)
                else:
                    text_parts.append(f"  {key}")
            return "{\n" + "\n".join(text_parts) + "\n}"

        formatted_data = {
            "input": {
                # Store only the field names in text, not their selectors
                "text": format_query_text(input_query),
                # Use prettified HTML
                "html": prettified_html
            },
            "output": {}
        }

        # Process all fields from the input query
        for key, value in input_query.items():
            if key.endswith("[]"):  # Handle array fields
                field_name = key[:-2]
                formatted_data["output"][field_name] = []
                
                if field_name in data:
                    for item in data[field_name]:
                        item_data = {}
                        for sub_key in value.keys():
                            if sub_key != "selector":  # Skip the selector key
                                if sub_key in item:
                                    item_data[sub_key] = item[sub_key]
                        formatted_data["output"][field_name].append(item_data)
            else:  # Handle single value fields
                if key in data:
                    formatted_data["output"][key] = data[key]

        # Read existing data
        try:
            with open(self.output_file, "r") as file:
                existing_data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []

        # Append the new formatted_data
        existing_data.append(formatted_data)

        # Write back all data
        with open(self.output_file, "w") as file:
            json.dump(existing_data, file, indent=4)

    def generate_query_combinations(self, base_query):
        """
        Generate all possible combinations of query fields including both array and non-array fields.
        """
        # Separate array and single value queries
        array_queries = {k: v for k, v in base_query.items() if k.endswith('[]')}
        single_queries = {k: v for k, v in base_query.items() if not k.endswith('[]')}

        array_key = next(iter(array_queries))
        array_fields = {k: v for k, v in array_queries[array_key].items()
                        if k != 'selector'}

        print(f"\n🔍 Generating combinations:")
        print(f"   Array fields: {', '.join(array_fields.keys())}")
        print(f"   Single fields: {', '.join(single_queries.keys())}")

        array_field_names = list(array_fields.keys())
        single_field_names = list(single_queries.keys())
        total_fields = len(array_field_names) + len(single_field_names)
        print(f"   Expected combinations: 2^{total_fields} = {2**total_fields}")

        all_combinations = []
        
        for r_array in range(len(array_field_names) + 1):
            for r_single in range(len(single_field_names) + 1):
                for array_combo in combinations(array_field_names, r_array):
                    for single_combo in combinations(single_field_names, r_single):
                        if not array_combo and not single_combo:
                            # Skip empty combo
                            continue

                        # Build new query
                        new_query = {}

                        # Add array fields if selected
                        if array_combo:
                            new_array_query = {
                                "selector": array_queries[array_key]["selector"]
                            }
                            for field in array_combo:
                                new_array_query[field] = array_queries[array_key][field]
                            new_query[array_key] = new_array_query
                        
                        # Add single fields if selected
                        for field in single_combo:
                            new_query[field] = single_queries[field]

                        all_combinations.append(new_query)
                        print(f"   ✓ Generated query with: " + 
                              (f"array[{', '.join(array_combo)}] " if array_combo else "") +
                              (f"single[{', '.join(single_combo)}]" if single_combo else ""))

        print(f"\n📊 Total valid combinations generated: {len(all_combinations)}")
        return all_combinations

    def format_query_text(self, query):
        """
        (Optional) Additional function if needed to format the query text externally.
        """
        text_parts = []
        for key, value in query.items():
            if key.endswith("[]"):
                field_name = key[:-2]
                fields = [k for k in value.keys() if k != "selector"]
                if fields:
                    field_text = f"  {field_name}[] {{\n"
                    field_text += "\n".join(f"    {field}" for field in fields)
                    field_text += "\n  }"
                    text_parts.append(field_text)
            else:
                text_parts.append(f"  {key}")
        return "{\n" + "\n".join(text_parts) + "\n}"

    def process_url(self, url, base_query):
        """
        Process a single URL with the full query and then generate combinations.
        """
        print(f"\n🌐 Fetching URL: {url}")
        
        # Fetch HTML content once
        html_content = self.fetch_html(url)
        if not html_content:
            print(f"   ⚠️ Failed to fetch URL")
            return

        # First get the full data with all fields
        full_output = self.parse_html(html_content, base_query)
        
        # Generate all possible query combinations
        query_combinations = self.generate_query_combinations(base_query)
        print(f"\n   Processing {len(query_combinations)} query combinations:")

        # For each combination, filter the full output
        for query_index, query in enumerate(query_combinations, 1):
            filtered_output = {}
            
            # Filter array fields
            for key, value in query.items():
                if key.endswith("[]"):
                    field_name = key[:-2]
                    if field_name in full_output:
                        filtered_output[field_name] = []
                        for item in full_output[field_name]:
                            filtered_item = {}
                            for sub_key in value.keys():
                                if sub_key != "selector" and sub_key in item:
                                    filtered_item[sub_key] = item[sub_key]
                            if filtered_item:  # Only add if we have data
                                filtered_output[field_name].append(filtered_item)
                else:  # Handle single value fields
                    if key in full_output:
                        filtered_output[key] = full_output[key]

            # Get field summary for logging
            fields = []
            if any(k.endswith('[]') for k in query.keys()):
                array_key = next(k for k in query.keys() if k.endswith('[]'))
                fields.extend([k for k in query[array_key].keys() if k != 'selector'])
            fields.extend([k for k in query.keys() if not k.endswith('[]')])
            
            # Count products for logging
            product_count = len(filtered_output.get('products', [])) if 'products' in filtered_output else 0
            
            print(f"\n   → Combination {query_index}/{len(query_combinations)}:")
            print(f"     Fields: {', '.join(fields)}")
            print(f"     Products found: {product_count}")
            
            # Save this combination
            self.save_to_file(filtered_output, html_content, query)


if __name__ == "__main__":
    scraper = HTMLContentScraper(output_file="dataset.json")

    # Define the base input query with all fields
    base_query = {
        "products[]": {
            "selector": 'div[data-component-type="s-search-result"]',
            "product_name": {"selector": 'span[class="a-size-base-plus a-color-base"]'},
            "product_price": {"selector": 'span[class="a-offscreen"]'},
            "product_bottom_text": {"selector": 'a > h2 > span'}, 
            "product_image": {"selector": 'img[class="s-image"]'}
        },
        "search_bar": {"selector": 'input[id="twotabsearchtextbox"]'}
    }

    # List of target URLs
    urls = [
        'https://www.amazon.com/s?k=gaming&crid=3ELSHO9OECLAV&sprefix=gam%2Caps%2C327&ref=nb_sb_noss_2',
        'https://www.amazon.com/s?k=clothes&crid=1UJ547SLZMGH1&sprefix=%2Caps%2C334&ref=nb_sb_ss_recent_4_0_recent',
        'https://www.amazon.com/s?k=faber+castell+colored+pencils&crid=OPQRXY7VGUHJ&sprefix=faber%2Caps%2C460&ref=nb_sb_ss_ts-doa-p_1_5'
    ]

    print(f"\n📊 Processing {len(urls)} URLs:")
    for i, url in enumerate(urls, 1):
        print(f"   {i}. {url}")

    # Process each URL once
    for url_index, url in enumerate(urls, 1):
        print(f"\n🔄 Processing URL {url_index}/{len(urls)}")
        scraper.process_url(url, base_query)

    print(f"\n✅ All data saved to {scraper.output_file}")
