from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-web-security')
options.add_argument('--disable-software-rasterizer')
options.add_argument('--log-level=3')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115 Safari/537.36")

driver = webdriver.Chrome(options=options)
driver.get("https://finviz.com/news.ashx")

time.sleep(4)

html = driver.page_source
driver.quit()

soup = BeautifulSoup(html, 'html.parser')
news_blocks = soup.select("a.news-link")

if not news_blocks:
    print("❌ No news found. The structure may have changed or page failed to load.")
else:
    print("🔹 Latest News Headlines from Finviz 🔹\n")
    for block in news_blocks:
        try:
            headline = block.select_one(".news-link-right").text.strip()
            timestamp = block.select_one(".news-link-left").text.strip()
            link = "https://finviz.com" + block['href']
            print(f"[{timestamp}] {headline}\n→ {link}\n")
        except:
        
            print("❌ Error parsing a news block. The structure may have changed.")
