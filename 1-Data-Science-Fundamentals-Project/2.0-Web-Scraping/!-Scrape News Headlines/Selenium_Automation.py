from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time

# Optional: run browser maximized
options = Options()
options.add_argument("--start-maximized")

# Set up the Chrome browser
driver = webdriver.Chrome(options=options)

# Open Bing
driver.get("https://www.bing.com/")

# Find the search box
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("Uday Garg")
search_box.send_keys(Keys.RETURN)

# Wait for results to load
time.sleep(3)

# Scroll down a bit (optional: simulate user behavior)
driver.execute_script("window.scrollBy(0, 400)")

# Wait again for smooth scroll/render
time.sleep(2)

# Find all result links
links = driver.find_elements(By.XPATH, "//li[@class='b_algo']//a")

# Look for the link that contains 'udaygarg.com'
for link in links:
    href = link.get_attribute("href")
    if href and "udaygarg.com" in href:
        print("Opening:", href)
        link.click()
        break

# Let the page load
time.sleep(10)

# Close browser
driver.quit()
