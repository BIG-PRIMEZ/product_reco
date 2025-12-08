"""
Debug scraper to see what's happening with image URLs
"""
import sys
sys.path.insert(0, '.')

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# Initialize driver
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # Test with one product
    search_term = "LUNCH BAG PINK POLKADOT"
    search_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}&tbm=isch"

    print(f"Navigating to: {search_url}")
    driver.get(search_url)
    time.sleep(3)

    # Scroll
    print("Scrolling...")
    for i in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)

    # Find images
    print("\nTrying different selectors...")

    selectors = [
        "img.rg_i",
        "img.YQ4gaf",
        "img[data-src]",
        "div.isv-r img",
        "img"
    ]

    for selector in selectors:
        elements = driver.find_elements(By.CSS_SELECTOR, selector)
        print(f"\n{selector}: Found {len(elements)} elements")

        if elements and len(elements) > 0:
            print(f"  First 3 examples:")
            for i, elem in enumerate(elements[:3]):
                src = elem.get_attribute('src')
                data_src = elem.get_attribute('data-src')
                data_original = elem.get_attribute('data-original')

                print(f"    Element {i}:")
                print(f"      src: {src[:80] if src else 'None'}...")
                print(f"      data-src: {data_src[:80] if data_src else 'None'}...")
                print(f"      data-original: {data_original[:80] if data_original else 'None'}...")

    # Try clicking approach
    print("\n\nTrying click approach:")
    img_results = driver.find_elements(By.CSS_SELECTOR, "div.isv-r")
    print(f"Found {len(img_results)} clickable result divs")

    if img_results:
        print("\nClicking first result...")
        try:
            img_results[0].click()
            time.sleep(2)

            # Look for full-size image
            full_img = driver.find_elements(By.CSS_SELECTOR, "img.sFlh5c")
            if not full_img:
                full_img = driver.find_elements(By.CSS_SELECTOR, "img.iPVvYb")
            if not full_img:
                full_img = driver.find_elements(By.CSS_SELECTOR, "img[jsname]")

            print(f"Found {len(full_img)} full-size images")
            if full_img:
                src = full_img[0].get_attribute('src')
                print(f"Full-size src: {src[:100] if src else 'None'}...")

        except Exception as e:
            print(f"Error clicking: {e}")

finally:
    driver.quit()
    print("\nBrowser closed")
