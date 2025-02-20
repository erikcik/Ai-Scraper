from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import tiktoken
import random
import time
from typing import Optional, Dict, Union

def get_random_user_agent() -> str:
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Edge/122.0.0.0',
    ]
    return random.choice(user_agents)

def get_page_html(url: str, max_scroll: Optional[int] = 1) -> str:
    with sync_playwright() as p:
        # Browser arguments to avoid detection
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-extensions',
            '--disable-infobars',
            '--no-first-run',
            '--enable-webgl',
            f'--user-agent={get_random_user_agent()}'
        ]
        
        # Launch browser with custom arguments
        browser = p.chromium.launch(
            headless=False,  # True for production
            args=browser_args
        )
        
        # Create context with additional options
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent=get_random_user_agent(),
            java_script_enabled=True,
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1'
            }
        )
        
        page = context.new_page()
        
        # Randomize timing slightly
        page.set_default_timeout(10000)  # 10s timeout
        page.set_default_navigation_timeout(10000)
        
        # Navigate with smart waiting
        page.goto(url, wait_until='load')
        
        # Quick random mouse movement simulation
        if random.random() < 0.5:  # 50% chance to do mouse movements
            page.mouse.move(random.randint(100, 500), random.randint(100, 500))
            
        # Smart scrolling with minimal delay
        if max_scroll > 0:
            for _ in range(max_scroll):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight * Math.random());")
                time.sleep(random.uniform(0.1, 0.3))  # Minimal random delay
                
        # Get content after all operations
        html_content = page.content()
        
        # Clean up
        context.close()
        browser.close()
        
        return html_content

def clean_html(html: str) -> str:
    """
    Clean HTML content by removing unwanted elements.
    
    Args:
        html (str): Raw HTML content to clean
    
    Returns:
        str: Cleaned HTML content
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove <head> entirely
    if soup.head:
        soup.head.decompose()

    # Remove specific unwanted tags
    unwanted_tags = ['script', 'style', 'noscript', 'iframe']
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
    
    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment))):
        comment.extract()
    
    return str(soup)

# Example usage
if __name__ == "__main__":
    url = "https://www.amazon.co.uk/s?k=laptops"
    html = get_page_html(url)
    
    # Clean the HTML
    cleaned_html = clean_html(html)
    
    # Token counting logic
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(html))
    cleaned_token_count = len(encoding.encode(cleaned_html))
    
    print(f"Raw HTML token count: {token_count}")
    print(f"Cleaned HTML token count: {cleaned_token_count}")
    print(f"Token reduction: {token_count - cleaned_token_count} tokens ({((token_count - cleaned_token_count)/token_count)*100:.2f}%)")
    
    # Save files if needed
    with open('amazon_gaming_monitor.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    with open('amazon_gaming_monitor_pruned.html', 'w', encoding='utf-8') as f:
        f.write(cleaned_html)
