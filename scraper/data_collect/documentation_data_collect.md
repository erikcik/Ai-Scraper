Below is an example "documentation.md" file that explains how to use the data collection module provided in data_collect.py. You can adjust or extend these instructions as needed for your project.

---

# Documentation for Data Collection Module

This module provides a flexible, asynchronous web scraping solution using Playwright. It is designed to extract structured data from web pages using a configurable query-based approach. The output includes both the extracted values and their corresponding CSS selectors.

## Overview

The module is centered around several key components:

- **Field**: Represents a single data point to extract (e.g., product name, price).  
  - Each field includes a `name`, a CSS `selector`, and options such as `value_type` (e.g., "text" or "attribute"), and an optional `preprocessing` function.

- **BaseQuery**: Enables grouping of multiple fields into a structured query.  
  - Supports both list queries (for multiple items) and single queries. Nested queries are supported allowing the extraction of complex, hierarchical data.

- **ScrapingConfig**: Contains configuration related to a scraping session including:  
  - The URL from which to scrape data.  
  - A list of query definitions (Fields and/or BaseQueries).  
  - An option to include the full HTML content in the output.

- **TrainingDataGenerator**: Responsible for executing the scraping process using the specified configuration.  
  - It implements methods to generate field combinations, format GraphQL-style queries, scrape individual fields (including their CSS selectors), and compile the results.
  - The module logs detailed debug information for each step (e.g., when an element is found or if a timeout occurs).

- **process_url function**:  
  - Manages the overall scraping flow for a single URL.
  - Launches the browser using Playwright, configures the context and page, navigates to the target URL, waits for dynamic content, scrolls to load content, and then triggers the scraping process.
  - Finally, the scraped data is saved to a specified JSON file.

## Prerequisites

- **Python 3.7+**: Ensure you’re running a compatible Python version.
- **Playwright**: Used for controlling the headless browser.  
  - Install using: `pip install playwright` and then initialize with: `playwright install`
- Other standard libraries such as `asyncio`, `json`, and `dataclasses` are used.

## Installation

1. Clone or copy the module file to your working directory.
2. Install Playwright (if not already installed):
   ```bash
   pip install playwright
   playwright install
   ```
3. Ensure that any dependencies (such as Pydantic if you use additional modules) are installed.

## Configuration

To use the data collection module, first create a `ScrapingConfig` that defines:

- The target URL (for instance, a product page on Amazon).
- An array of Fields and BaseQueries which specify the items you wish to extract.  
  For example:
  - A Field for the search bar with a given selector.
  - A BaseQuery for a list of products, including fields such as product name, product price, image URL, etc.
  
Example configuration snippet:
```python
config = ScrapingConfig(
    url="https://www.example.com/products",
    fields=[
        Field(
            name="search_bar",
            selector='input[id="search"]',
            value_type="attribute",
            attribute_name="value"
        ),
        BaseQuery(
            name="product_list",
            selector="div.product-item",
            fields=[
                Field(name="product_name", selector="h1.product-title"),
                Field(name="product_price", selector="span.price"),
                Field(name="product_image_url", selector="img.product-image", value_type="attribute", attribute_name="src"),
                Field(name="product_link", selector="a.product-link", value_type="attribute", attribute_name="href"),
            ]
        )
    ],
    include_html=True  # Save the full HTML content along with the extracted data.
)
```

## How It Works

1. **Field Extraction**:  
   - The `scrape_field` method uses the provided CSS selector to locate the HTML element. If found, it extracts the field's value (text or attribute) and determines the element’s full CSS selector using `get_full_selector`.

2. **Handling Complex Queries**:  
   - The `scrape_base_query` method supports nested or list-based queries.  
   - For list queries, it limits the number of elements processed to avoid timeouts (for example, processing at most 24 items).

3. **Scraping the Page**:  
   - The `scrape_page` method combines all field extraction results. It calls either `scrape_field` or `scrape_base_query`, aggregates results, and appends them to a data list.
   - If the configuration is set to include the HTML content, it is stored along with the queries.

4. **Saving Data**:  
   - The `save_to_file` function writes the collected and processed data to a JSON file.  
   - The output includes both the raw extracted values and the associated CSS selectors (with keys following the naming convention, e.g., `product_name_element`).

## Running the Scraper

The entry point for the module is the `process_url` function, which is called for each URL to scrape. A sample main function is provided at the end of the file that:

1. Lists one or more URLs to scrape.
2. Iterates through each URL.
3. Calls `process_url` to perform scraping and save the results.
4. Implements delays between scrapes to avoid rate limiting.

To run the scraper:
```bash
python data_collect.py
```
Check the generated JSON file (for example, "amazon_gaming_monitors.json") for the structured output.

## Debugging and Logging

- The module is verbose. Debug messages trace significant events:
  - When an element is found or missing.
  - Timeouts when scraping fields or queries.
  - Details of each field processed.
- Review the console output for warnings or errors to troubleshoot any issues that occur during scraping.

## Customization

- **Selectors and Preprocessing**:  
  Adjust the selectors in each Field according to the target page’s structure. You can provide a custom preprocessing lambda to clean or transform values.
  
- **Timeouts**:  
  Modify the timeout parameters in `asyncio.wait_for` calls if you encounter delays in loading content or processing complex pages.

- **Field Combinations**:  
  The module supports generating different combinations of field queries based on the provided configuration. This can be useful if pages have diverse structures.

## Conclusion

This data collection module offers a robust foundation for web scraping and training data generation. By configuring Fields and BaseQueries appropriately, it is possible to collect rich, structured data, including not only the values but also their CSS selectors—a key feature for further automated data processing or model fine-tuning.

Happy scraping!

---

You can now include this `documentation.md` in your repository to help other developers understand and use the data collection functionality effectively.
