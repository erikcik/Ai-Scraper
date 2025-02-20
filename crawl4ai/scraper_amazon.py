import os
import json
import asyncio
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import json
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from bs4 import BeautifulSoup

class AmazonProduct(BaseModel):
    product_name: str = Field(..., description="Name/title of the Amazon product")
    price: str = Field(..., description="Price of the product including currency symbol")

async def extract_amazon_products(
    provider: str, api_token: str = None
):
    print(f"\n--- Extracting Amazon Products with {provider} ---")

    browser_config = BrowserConfig(
        browser_type="chromium",
        user_agent_mode="random",
        headless=False,
        light_mode=True,
        verbose=True,        
    )

    extra_args = {
        "temperature": 0.1,  # Slightly increased for better extraction
        "top_p": 0.95,
        "max_tokens": 2323  # Increased to handle more products
    }
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(),
        word_count_threshold=100,  # Increased to ensure we have enough content
        page_timeout=120000,  # Increased timeout to 120 seconds
        wait_for="css:div.s-result-item",  # Wait for product items to load
        js_code="""
            // Scroll down to load more products
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 2000));
            window.scrollTo(0, 0);
            await new Promise(r => setTimeout(r, 1000));
        """,
        
        extraction_strategy=LLMExtractionStrategy(
            provider=provider,
            api_token=api_token,
            schema=AmazonProduct.model_json_schema(),
            extraction_type="schema",
            instruction="""Analyze the Amazon product listing page and extract product information.
            For each product listing you can find:
            1. Extract the complete product name/title exactly as shown
            2. Extract the price including currency symbol (£). If multiple prices exist, use the main displayed price
            Focus only on actual products with visible prices.
            Ignore any sponsored or promotional content.
            If a product doesn't have both a clear name and price, skip it.""",
            extra_args=extra_args,
        ),
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.amazon.co.uk/gaming-monitor/s?k=gaming+monitor",
            config=crawler_config
        )
        formatted_html = BeautifulSoup(result.html, 'html.parser').prettify()
        
        result_dict = {
            'result': result.extracted_content
        }
        
        with open('result.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(
        extract_amazon_products(
            provider="openrouter/openrouter/auto", 
            api_token='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'
        )
    )