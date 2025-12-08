"""
Image Scraper Service - Download product images using Selenium
Scrapes Google Images for e-commerce product training data
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import hashlib


class ScraperService:
    """
    Service for scraping product images from Google Images.
    Uses Selenium for automated browser interaction.
    """

    def __init__(self, base_dir="data/product_images", headless=True):
        """
        Initialize scraper service.

        Args:
            base_dir: Base directory to store scraped images
            headless: Run browser in headless mode (no GUI)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.driver = None

    def _init_driver(self):
        """Initialize Selenium WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def scrape_google_images(self, search_term: str, max_images: int = 150) -> list:
        """
        Scrape images from Google Images.

        Args:
            search_term: Product search term
            max_images: Maximum number of images to download

        Returns:
            List of downloaded image paths
        """
        print(f"\nScraping Google Images for: {search_term}")
        print(f"Target: {max_images} images")

        # Create product directory
        product_dir = self.base_dir / self._sanitize_filename(search_term)
        product_dir.mkdir(parents=True, exist_ok=True)

        # Initialize driver
        if not self.driver:
            self._init_driver()

        image_paths = []

        try:
            # Navigate to Google Images
            search_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}&tbm=isch"
            self.driver.get(search_url)
            time.sleep(2)

            # Scroll to load more images (aggressive scrolling to load ~150+ images)
            for _ in range(10):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(0.8)

                # Try to click "Show more results" button if it appears
                try:
                    show_more = self.driver.find_elements(By.CSS_SELECTOR, ".mye4qd")
                    if show_more:
                        show_more[0].click()
                        time.sleep(1)
                except:
                    pass

            # Find all images - Google Images loads thumbnails with encrypted-tbn URLs
            img_elements = self.driver.find_elements(By.TAG_NAME, "img")

            # Filter to only Google Images thumbnails (exclude logo, icons, etc.)
            valid_img_elements = []
            for img in img_elements:
                src = img.get_attribute('src') or ''
                # Google Images thumbnails are served from encrypted-tbn subdomain
                if 'encrypted-tbn' in src or 'gstatic.com/images' in src and 'encrypted-tbn' in src:
                    valid_img_elements.append(img)

            img_elements = valid_img_elements
            print(f"  Found {len(img_elements)} valid image elements")

            downloaded = 0
            for img_element in img_elements:
                if downloaded >= max_images:
                    break

                try:
                    # Get image source
                    img_url = img_element.get_attribute('src')

                    # Skip invalid URLs
                    if not img_url or img_url.startswith('data:'):
                        continue

                    # Download image
                    img_path = self._download_image(img_url, product_dir, downloaded)
                    if img_path:
                        image_paths.append(str(img_path))
                        downloaded += 1

                        if downloaded % 10 == 0:
                            print(f"  Downloaded: {downloaded}/{max_images}")

                    # Rate limiting
                    time.sleep(0.3)

                except Exception as e:
                    continue

        except Exception as e:
            print(f"  Error during scraping: {e}")

        print(f"  Total downloaded: {len(image_paths)} images")
        return image_paths

    def _download_image(self, url: str, save_dir: Path, index: int) -> Path:
        """
        Download and save an image.

        Args:
            url: Image URL
            save_dir: Directory to save image
            index: Image index for filename

        Returns:
            Path to saved image or None if failed
        """
        try:
            # Download image
            response = requests.get(url, timeout=10)

            # Validate image
            img = Image.open(BytesIO(response.content))

            # Skip very small images
            if img.width < 100 or img.height < 100:
                return None

            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Generate filename
            img_hash = hashlib.md5(response.content).hexdigest()[:8]
            filename = f"{index:04d}_{img_hash}.jpg"
            filepath = save_dir / filename

            # Save image
            img.save(filepath, 'JPEG', quality=85)

            return filepath

        except Exception:
            return None

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize string for use as filename."""
        return "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in name).strip()

    def validate_images(self, product_dir: Path) -> tuple:
        """
        Validate downloaded images.

        Args:
            product_dir: Directory containing images

        Returns:
            Tuple of (valid_count, invalid_files)
        """
        valid_count = 0
        invalid_files = []

        for img_file in product_dir.glob('*.jpg'):
            try:
                img = Image.open(img_file)
                img.verify()
                valid_count += 1
            except Exception:
                invalid_files.append(str(img_file))

        return valid_count, invalid_files

    def get_image_count(self, product_name: str) -> int:
        """Get count of downloaded images for a product."""
        product_dir = self.base_dir / self._sanitize_filename(product_name)
        if product_dir.exists():
            return len(list(product_dir.glob('*.jpg')))
        return 0

    def close(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
