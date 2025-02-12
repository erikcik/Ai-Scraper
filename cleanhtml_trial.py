from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import tiktoken

def get_page_html(url):
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        
        # Create new page
        page = browser.new_page()
        
        # Go to URL
        page.goto(url)   
        
        # Get HTML content
        html_content = page.content()
        
        # Close browser
        browser.close()
        
        return html_content

def prune_with_beautifulsoup(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove <head> entirely
    if soup.head:
        soup.head.decompose()

    # Remove specific common tags outside of <head>
    unwanted_tags = ['script', 'style', 'noscript', 'iframe']
    for tag in soup.find_all(unwanted_tags):
        tag.decompose()

    # Optionally remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment))):
        comment.extract()
    return str(soup)

# Example usage
if __name__ == "__main__":
    import tiktoken

    url = "https://www.amazon.co.uk/gaming-monitor/s?k=gaming+monitor"
    html = get_page_html(url)
    
    # Count tokens for raw HTML
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(html))
    print(f"Raw HTML token count: {token_count}")
    
    with open('amazon_gaming_monitor.html', 'w', encoding='utf-8') as f:
        f.write(html)

    pruned_html = prune_with_beautifulsoup(html)
    
    # Count tokens for pruned HTML
    pruned_token_count = len(encoding.encode(pruned_html))
    print(f"Pruned HTML token count: {pruned_token_count}")
    print(f"Token reduction: {token_count - pruned_token_count} tokens ({((token_count - pruned_token_count)/token_count)*100:.2f}%)")
    
    with open('amazon_gaming_monitor_pruned.html', 'w', encoding='utf-8') as f:
        f.write(pruned_html)
