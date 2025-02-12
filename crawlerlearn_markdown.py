from crawl4ai import DefaultMarkdownGenerator, CrawlerRunConfig, AsyncWebCrawler, CacheMode, BM25ContentFilter, PruningContentFilter, LLMContentFilter, BrowserConfig
import json
import tiktoken


config = CrawlerRunConfig(
    markdown_generator= DefaultMarkdownGenerator(),
    cache_mode=CacheMode.BYPASS,
)
config2 = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=BM25ContentFilter(
            user_query='product',
            bm25_threshold=0.5
        )
    ),
    cache_mode=CacheMode.BYPASS,

)
config3 = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(
            threshold=0.5,
            threshold_type='dynamic'
        )
    ),
    cache_mode=CacheMode.BYPASS,
)
config4 = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=LLMContentFilter(
            provider='openrouter/openrouter/auto',
            api_token='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',
            instruction="""
    Extract detailed product information with a focus on product names and product prices. Ensure that:
    1. Product names are captured exactly as presented, preserving all original formatting and terminology.
    2. Prices (including currency symbols and numerical values) are accurately extracted.
    3. The output is formatted into structured markdown (using lists or tables) to clearly display products alongside their prices.
    4. Remove non-product related elements such as navigation menus, ads, and other irrelevant content.
    """,
    chunk_token_threshold=4096
        )
    )
)

async def main():
    async with AsyncWebCrawler(config=BrowserConfig(
        headless=False
    )) as crawler:
        configs = [config, config2, config3, config4]
        results = []
        for cfg in configs:

            result = await crawler.arun("https://www.amazon.co.uk/gaming-monitor/s?k=gaming+monitor", config=cfg)
            results.append(result)
        
        if all(r.success for r in results):
            encoding = tiktoken.get_encoding("cl100k_base")
            for i, result in enumerate(results, 1):
                tokens = len(encoding.encode(result.markdown_v2.fit_markdown))
                tokens2 = len(encoding.encode(result.markdown_v2.raw_markdown))
                print(f"Result {i} token count: {tokens}")
                print(f"Result raw {i} token count: {tokens2}")

            for i, result in enumerate(results, 1):
                with open(f'result{i}.txt', 'w', encoding='utf-8') as f:
                    f.write(result.markdown_v2.fit_markdown)
        else:
            for i, result in enumerate(results, 1):
                if not result.success:
                    print(f"Crawl {i} failed: {result.error_message}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())