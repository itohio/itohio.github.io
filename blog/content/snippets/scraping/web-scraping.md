---
title: "Web Scraping with Python"
date: 2024-12-12
draft: false
category: "scraping"
tags: ["scraping-knowhow", "python", "beautifulsoup", "scrapy", "selenium", "crawl4ai"]
---


Web scraping tools and techniques with Python. Includes BeautifulSoup, Scrapy, Selenium, and crawl4ai patterns.

---

## BeautifulSoup - HTML Parsing

### Installation

```bash
pip install beautifulsoup4 requests lxml
```

### Basic Usage

```python
import requests
from bs4 import BeautifulSoup

# Fetch page
url = "https://example.com"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml')

# Find elements
title = soup.find('title').text
links = soup.find_all('a')
divs = soup.find_all('div', class_='content')

# CSS selectors
items = soup.select('.item')
first_item = soup.select_one('#first')

# Navigate
parent = soup.find('div').parent
siblings = soup.find('div').find_next_siblings()
children = soup.find('div').children

# Extract data
for link in soup.find_all('a'):
    href = link.get('href')
    text = link.text.strip()
    print(f"{text}: {href}")
```

### Complete Example

```python
import requests
from bs4 import BeautifulSoup
import csv

def scrape_products(url):
    """Scrape product information from e-commerce site"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    
    products = []
    
    for item in soup.select('.product-item'):
        product = {
            'name': item.select_one('.product-name').text.strip(),
            'price': item.select_one('.product-price').text.strip(),
            'rating': item.select_one('.product-rating')['data-rating'],
            'url': item.select_one('a')['href']
        }
        products.append(product)
    
    return products

def save_to_csv(products, filename='products.csv'):
    """Save products to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)

# Usage
products = scrape_products('https://example.com/products')
save_to_csv(products)
```

---

## Scrapy - Web Scraping Framework

### Installation

```bash
pip install scrapy
```

### Create Project

```bash
scrapy startproject myproject
cd myproject
scrapy genspider example example.com
```

### Spider Example

```python
# spiders/products_spider.py
import scrapy

class ProductsSpider(scrapy.Spider):
    name = 'products'
    allowed_domains = ['example.com']
    start_urls = ['https://example.com/products']
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 4,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def parse(self, response):
        """Parse product listing page"""
        for product in response.css('.product-item'):
            yield {
                'name': product.css('.product-name::text').get().strip(),
                'price': product.css('.product-price::text').get().strip(),
                'rating': product.css('.product-rating::attr(data-rating)').get(),
                'url': response.urljoin(product.css('a::attr(href)').get())
            }
        
        # Follow pagination
        next_page = response.css('a.next-page::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
    
    def parse_product(self, response):
        """Parse individual product page"""
        yield {
            'name': response.css('h1.product-title::text').get(),
            'price': response.css('.price::text').get(),
            'description': response.css('.description::text').get(),
            'images': response.css('.product-image::attr(src)').getall(),
            'specifications': {
                spec.css('.spec-name::text').get(): spec.css('.spec-value::text').get()
                for spec in response.css('.specification')
            }
        }
```

### Run Spider

```bash
# Run spider
scrapy crawl products

# Save to JSON
scrapy crawl products -o products.json

# Save to CSV
scrapy crawl products -o products.csv

# Save to JSON Lines
scrapy crawl products -o products.jl
```

### Scrapy Settings

```python
# settings.py
BOT_NAME = 'myproject'

# Obey robots.txt
ROBOTSTXT_OBEY = True

# Configure delays
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# Concurrent requests
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 4

# User agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Middleware
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
}

# Pipelines
ITEM_PIPELINES = {
    'myproject.pipelines.CleanDataPipeline': 300,
    'myproject.pipelines.SaveToDBPipeline': 400,
}

# AutoThrottle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
```

---

## Selenium - Browser Automation

### Installation

```bash
pip install selenium webdriver-manager
```

### Basic Usage

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Setup driver
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run in background
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

try:
    # Navigate to page
    driver.get('https://example.com')
    
    # Wait for element
    wait = WebDriverWait(driver, 10)
    element = wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, 'product-item'))
    )
    
    # Find elements
    products = driver.find_elements(By.CLASS_NAME, 'product-item')
    
    # Extract data
    for product in products:
        name = product.find_element(By.CLASS_NAME, 'product-name').text
        price = product.find_element(By.CLASS_NAME, 'product-price').text
        print(f"{name}: {price}")
    
    # Interact with page
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys('laptop')
    search_box.submit()
    
    # Wait for results
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'results')))
    
    # Scroll to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
    # Take screenshot
    driver.save_screenshot('screenshot.png')

finally:
    driver.quit()
```

### Handle Dynamic Content

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def scrape_infinite_scroll(url):
    """Scrape page with infinite scroll"""
    driver = webdriver.Chrome()
    driver.get(url)
    
    products = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Extract products
        items = driver.find_elements(By.CLASS_NAME, 'product-item')
        for item in items:
            products.append({
                'name': item.find_element(By.CLASS_NAME, 'name').text,
                'price': item.find_element(By.CLASS_NAME, 'price').text
            })
        
        # Check if reached bottom
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    driver.quit()
    return products
```

---

## crawl4ai - AI-Powered Scraping

### Installation

```bash
pip install crawl4ai
```

### Basic Usage

```python
from crawl4ai import WebCrawler

# Initialize crawler
crawler = WebCrawler()

# Crawl page
result = crawler.run(url="https://example.com")

# Extract content
print(result.markdown)  # Markdown content
print(result.cleaned_html)  # Cleaned HTML
print(result.media)  # Images, videos
print(result.links)  # All links

# Extract structured data
result = crawler.run(
    url="https://example.com/products",
    extraction_strategy="css",
    css_selector=".product-item"
)

for item in result.extracted_content:
    print(item)
```

### Advanced Usage

```python
from crawl4ai import WebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

# Define extraction schema
schema = {
    "name": "product",
    "baseSelector": ".product-item",
    "fields": [
        {
            "name": "title",
            "selector": ".product-name",
            "type": "text"
        },
        {
            "name": "price",
            "selector": ".product-price",
            "type": "text"
        },
        {
            "name": "image",
            "selector": "img",
            "type": "attribute",
            "attribute": "src"
        }
    ]
}

# Configure crawler
config = CrawlerRunConfig(
    extraction_strategy=JsonCssExtractionStrategy(schema),
    wait_for_selector=".product-item",
    screenshot=True,
    verbose=True
)

# Run crawler
crawler = WebCrawler()
result = crawler.run(url="https://example.com/products", config=config)

# Process results
for product in result.extracted_content:
    print(f"{product['title']}: {product['price']}")
```

---

## Anti-Scraping Bypass

### Rotate User Agents

```python
import random

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
]

headers = {
    'User-Agent': random.choice(USER_AGENTS)
}

response = requests.get(url, headers=headers)
```

### Use Proxies

```python
import requests

proxies = {
    'http': 'http://proxy:port',
    'https': 'http://proxy:port'
}

response = requests.get(url, proxies=proxies)

# Rotating proxies
PROXY_LIST = ['proxy1:port', 'proxy2:port', 'proxy3:port']

for url in urls:
    proxy = random.choice(PROXY_LIST)
    proxies = {'http': f'http://{proxy}', 'https': f'http://{proxy}'}
    response = requests.get(url, proxies=proxies)
```

### Handle Rate Limiting

```python
import time
from functools import wraps

def rate_limit(delay=1):
    """Decorator to add delay between requests"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(delay=2)
def fetch_page(url):
    return requests.get(url)
```

### Handle CAPTCHAs

```python
# Use 2captcha or similar service
from twocaptcha import TwoCaptcha

solver = TwoCaptcha('YOUR_API_KEY')

try:
    result = solver.recaptcha(
        sitekey='6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-',
        url='https://example.com'
    )
    print(f"Solved: {result['code']}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Best Practices

```python
# ✅ Respect robots.txt
from urllib.robotparser import RobotFileParser

rp = RobotFileParser()
rp.set_url("https://example.com/robots.txt")
rp.read()

if rp.can_fetch("*", "https://example.com/page"):
    # Scrape page
    pass

# ✅ Add delays
time.sleep(random.uniform(1, 3))

# ✅ Use session for efficiency
session = requests.Session()
response = session.get(url)

# ✅ Handle errors gracefully
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

# ✅ Save progress
import json

def save_checkpoint(data, filename='checkpoint.json'):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_checkpoint(filename='checkpoint.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
```

---