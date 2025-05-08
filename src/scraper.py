import argparse
from bs4 import BeautifulSoup
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from tabulate import tabulate
import os
import time
import json

def focus_window(driver):
    # Get the window handle
    driver.minimize_window()
    driver.maximize_window()


def search_novels(query, website, driver):
    """Search for novels by name using Selenium and return matching results"""
    try:
        # Navigate to the search page
        driver.get(website["search_url"])
        print(f"Loaded search page: {website['search_url']}")
        focus_window(driver)
        time.sleep(6.5)
        
        # Find the search input field
        print("Waiting for search input field...")
        search_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, website["search_input_selector"]))
        )
        print("Found search input field.")
        
        # Enter the query into the search bar
        search_input.clear()
        search_input.send_keys(query)
        search_input.send_keys(Keys.RETURN)
        print(f"Submitted search query: {query}")
        
        # Wait for the search results to load
        print("Waiting for search results...")
        time.sleep(2)  # Wait for the search bar
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, website["search_item_selector"]))
        )
        print("Search results loaded successfully.")
        
        # Scroll to the bottom of the page to load all results (if needed)
        print("Scrolling to load more results...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)  # Wait for additional content to load
        
        # Parse the page source
        print("Parsing page source...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        print(f"Page source length: {len(driver.page_source)}")
        
        results = []
        for item in soup.select(website["search_item_selector"]):
            # print(f"Processing item: {item}")
            
            title_elem = item.select_one(website["search_title_selector"])
            if not title_elem:
                print("Skipping item due to missing title")
                continue

            author_elem = item.select_one(website["search_author_selector"])
            slug_elem = item.select_one(website["search_slug_selector"])
            slug = slug_elem['href'].split('/')[-1] if slug_elem else ""
            
            results.append({
                "title": title_elem.text.strip(),
                "author": author_elem.text.strip() if author_elem else "Unknown",
                "slug": slug,
                "website": website["name"]
            })
            
        return results
        
    except Exception as e:
        print(f"Search failed on {website['name']}: {str(e)}")
        return []

def select_novel(novel_name, website, driver):
    """Search across all websites and let user select a novel"""
    try:        
        # Search for novels
        results = search_novels(novel_name, website, driver)
            
        # Handle no results case
        if not results:
            print("No novels found with that name.")
            return None
            
        # Print results
        print(f"Found {len(results)} matches:")
        novels_found = [{"#": i, **result} for i, result in enumerate(results, 1)]
        print(tabulate(novels_found, headers="keys", tablefmt="pretty"))

        # Get user selection (or auto-select if only one result)
        if len(results) == 1:
            print("Only one novel found. Auto-selecting...")
            choice = 1  # Auto-select the first (and only) novel
        else:
            while True:
                try:
                    choice = int(input("Enter the number of the novel you want: "))
                    if 1 <= choice <= len(results):
                        break
                    print("Invalid selection. Try again.")
                except ValueError:
                    print("Please enter a valid number.")
        
        # Interact with the selected novel's element
        selected_novel = results[choice - 1]
        print(f"Selected: {selected_novel['title']}")
        time.sleep(2)  # Adjust the sleep time as needed

        # Get fresh reference to search results elements
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, website["search_item_selector"])))
        search_results = driver.find_elements(By.CSS_SELECTOR, website["search_item_selector"])

        if not search_results:
            print("Failed to locate search results on the page.")
            return None

    
        # Find the specific clickable element within the result item
        link_element = search_results[choice-1].find_element(
            By.CSS_SELECTOR, website["search_title_selector"]
        )
        
        # Scroll into view and click using JavaScript
        print(f"Navigating to: {link_element.get_attribute('href')}")
        novel_url = link_element.get_attribute('href')
        driver.execute_script("arguments[0].scrollIntoView(true);", link_element)
        driver.execute_script("arguments[0].click();", link_element)
        
        # Wait for new page to load
        time.sleep(2)
        print("Successfully loaded novel page")
        
        return {
        "novel": selected_novel,
        "url": novel_url
        }
    
    except Exception as e:
        print(f"Failed to select novel: {e}")
        return None

def get_novel_metadata(website, novel_url, driver):
    """
    Extract metadata for a novel from its dedicated page using Selenium.

    Args:
        website (dict): Website configuration containing CSS selectors.
        novel_url (str): URL of the novel's page.
        driver: Selenium WebDriver instance.

    Returns:
        dict: Novel metadata including title, author, description, chapters, etc.
    """
    try:
        # Navigate to the novel's page
        if driver.current_url != novel_url:
            print(f"Navigating to novel page: {novel_url}")
            driver.get(novel_url)
            time.sleep(7)  # Wait for the page to load

        # Parse the page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract novel title
        title = soup.select_one(website["title_selector"]).text.strip() if soup.select_one(website["title_selector"]) else "Unknown Title"

        # Extract novel author
        author_element = soup.select_one(website["author_selector"])
        author = author_element.get(website["author_selector_exact"], author_element.text.strip()) if author_element else "Unknown Author"
        
         # Extract novel description
        description = soup.select_one(website["description_selector"]).text.strip() if soup.select_one(website["description_selector"]) else "No description available."

        # Extract total number of chapters
        # Wait for the dropdown box to load
        dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, website["chapter_dropdown"]))
        )
        # Extract all options from the dropdown
        options = dropdown.find_elements(By.TAG_NAME, website["chapter_dropdown_options"])

        total_chapters = 0

        # Iterate through each option and calculate the number of chapters
        for option in options:
            text = option.text  # e.g., "C.1 - C.40"
            if " - " in text:
                start, end = text.split(" - ")
                start_num = int(start.lstrip("C."))
                end_num = int(end.lstrip("C."))
                chapters_in_range = (end_num - start_num) + 1  # Calculate chapters in this range
                total_chapters += chapters_in_range  # Add to the total

        print(f"Total number of chapters: {total_chapters}")

        # Extract additional metadata (e.g., genres, status)
        genres = []
        genre_elements = soup.select(website["genre_selector"])  # Locate all genre links
        if genre_elements:
            genres = [genre.text.strip() for genre in genre_elements]  # Extract the text content of each <a> tag
        else:
            print("No genres found.")

        status = "Unknown"
        status_element = soup.select_one(website["status_selector"])
        if status_element:
            status = status_element.text.strip()

        # Return the metadata
        return {
            "title": title,
            "author": author,
            "description": description,
            "total_chapters": total_chapters,
            "genres": genres,
            "status": status,
            "url": novel_url,
            "website": website["name"]
        }

    except Exception as e:
        print(f"Failed to fetch metadata for {novel_url}: {e}")
        return None

def navigate_to_latest_chapter(novel_url, driver):
    """
    Navigate to the last chapter from the novel homepage.

    Args:
        novel_url (str): URL of the novel's homepage.
        driver: Selenium WebDriver instance.
    """
    try:
        # Navigate to the novel homepage
        if driver.current_url != novel_url:
            print(f"Navigating to novel page: {novel_url}")
            driver.get(novel_url)
            time.sleep(7)  # Wait for the page to load

        # Locate the "Latest Chapters" section
        latest_chapters = driver.find_element(By.CLASS_NAME, "ul-list5")

        # Click the most recent chapter in the "Latest Chapters" section
        most_recent_chapter = latest_chapters.find_element(By.TAG_NAME, "a")
        most_recent_chapter.click()
        time.sleep(2.5)  # Wait for the chapter page to load

        print("Navigated to a random chapter from the homepage.")

    except Exception as e:
        print(f"Failed to navigate to a random chapter: {e}")
    
def navigate_to_chapter(driver, chapter_number):
    """
    Navigate to a specific chapter using the dropdown on the chapter page.

    Args:
        driver: Selenium WebDriver instance.
        chapter_number (int): The chapter number to navigate to.
    """
    try:
        print(f"Navigating to Chapter {chapter_number}")
        previous_url = driver.current_url

        # Keep window in the foreground
        driver.maximize_window()
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.NULL)  # Fake interaction
        
        # Wait for the dropdown icon to be clickable
        dropdown_icon = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "glyphicon-list-alt"))
        )
        
        # Click the dropdown icon to expand the list of chapters
        driver.execute_script("arguments[0].click();", dropdown_icon)
        time.sleep(1.0)  # Wait for the dropdown to load

        # Locate the dropdown list of chapters
        chapter_list = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "select-chapter"))
        )

        # Find the desired chapter option
        chapter_option = chapter_list.find_element(
            By.XPATH, f"//option[contains(text(), 'Chapter {chapter_number}')]"
        )

        # Scroll the dropdown to make the chapter option visible
        driver.execute_script("arguments[0].scrollIntoView();", chapter_option)
        time.sleep(0.5)  # Wait for scrolling to complete
        # Click the chapter option to navigate to the chapter
        chapter_option.click()
        time.sleep(2.5)  # Wait for the chapter page to load
        # Check if navigation was successful
        WebDriverWait(driver, 10).until(
            lambda d: d.current_url != previous_url
        )

        print(f"Navigated to Chapter {chapter_number}")
        return True

    except Exception as e:
        print(f"Failed to navigate to Chapter {chapter_number}: {e}")
        return False

def scrape_chapter(driver, website, novel_name, chapter_num, save_dir):
    """
    Scrape a specific chapter using Selenium and save it to a file.

    Args:
        driver: Selenium WebDriver instance.
        website (dict): Website configuration containing CSS selectors.
        novel_slug (str): Name of the novel (used for naming directories).
        chapter_num (int): Chapter number to scrape.
        save_dir (str): Root directory where the novel's chapters will be saved.

    Returns:
        dict: Chapter metadata (title, content, website).
    """
    try:
        # Wait for the chapter content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, website["content_selector"])))
        
        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract the chapter title
        title_tag = soup.select_one(website["chapter_title_selector"])
        title = title_tag.text.strip() if title_tag else f"Chapter {chapter_num}"

        # Extract the chapter content
        content_div = soup.select_one(website["content_selector"])
        if content_div:
            # Clean up content by removing scripts and ads
            for element in content_div.find_all(['script', 'div', 'aside']):
                element.decompose()
            
            paragraphs = content_div.find_all('p')
            content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        else:
            content = "Content not found."

        # Save the chapter to a file
        save_chapter_to_file(novel_name, chapter_num, title, content, save_dir)

        return {
            "title": title,
            "content": content,
            "website": website["name"]
        }

    except Exception as e:
        print(f"Failed to scrape {novel_name} - Chapter {chapter_num}: {e}")
        return None

def save_chapter_to_file(novel_name, chapter_num, title, content, save_dir = "novels"):
    """
    Save the chapter content to a file.

    Args:
        novel_name (str): Name of the novel (used for naming directories).
        chapter_num (int): Chapter number.
        title (str): Chapter title.
        content (str): Chapter content.
        save_dir (str): Root directory where the novel's chapters will be saved.
    """
    try:
        # Create the novel directory if it doesn't exist
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        novel_name = "-".join(novel_name.lower().split())
        novel_dir = os.path.join(root_dir, save_dir, novel_name)
        os.makedirs(novel_dir, exist_ok=True)

        # Create the file name
        file_name = f"chapter_{chapter_num}.txt"
        file_path = os.path.join(novel_dir, file_name)

        # Write the chapter content to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(f"{title}\n\n{content}")

        print(f"Saved chapter {chapter_num} to {file_path}")

    except Exception as e:
        print(f"Failed to save chapter {chapter_num}: {e}")

def scrape_all_chapters(driver, website, novel_metadata, save_dir = "novels"):
    """
    Scrape all chapters of a novel and save them to files.

    Args:
        driver: Selenium WebDriver instance.
        website (dict): Website configuration containing CSS selectors.
        novel_metadata (dict): Metadata of the novel, including 'title' and 'total_chapters'.
        save_dir (str): Directory to save the chapters.
    """
    for chapter_num in range(1, novel_metadata['total_chapters'] + 1):
        try:
            # Navigate to the chapter
            if navigate_to_chapter(driver, chapter_num):
                # Scrape and save the chapter
                scrape_chapter(driver, website, novel_metadata['title'], chapter_num, save_dir)
                print(f"Successfully scraped Chapter {chapter_num}")
            else:
                print(f"Failed to navigate to Chapter {chapter_num}")
        except Exception as e:
            print(f"Error scraping Chapter {chapter_num}: {e}")

def parse_chapter_input(chap_num, max_chapters=None):
    if chap_num is None:
        return list(range(1, max_chapters + 1)) if max_chapters else None
    
    chapters = set()
    
    # Split by commas first (handles cases like "1,3,5-7")
    parts = chap_num.split(",")
    
    for part in parts:
        part = part.strip()
        if "-" in part:     # Handle ranges (e.g., "10-15")
            start, end = map(int, part.split("-"))
            chapters.update(range(start, end + 1))
        else:
            # Handle single numbers (e.g., "5")
            chapters.add(int(part))
    
    # Filter out chapters beyond max_chapters (if provided)
    if max_chapters:
        chapters = {ch for ch in chapters if 0 <= ch <= max_chapters}
    
    return sorted(chapters)

def scrape_chapters(driver, website, novel_metadata, chap_num=None, save_dir="novels"):
    max_chapters = novel_metadata.get("total_chapters")
    
    # Get list of all chapter numbers to be scraped
    chapters_to_scrape = parse_chapter_input(chap_num, max_chapters)
    
    if not chapters_to_scrape:
        print("No valid chapters to scrape.")
        return
    
    # Case 1: Scrape all chapters (chap_num is None)
    if chap_num is None:
        scrape_all_chapters(driver, website, novel_metadata, save_dir=save_dir)
        return
    
    # Case 2: Scrape a single chapter (optimized path)
    if len(chapters_to_scrape) == 1:
        chapter_num = chapters_to_scrape[0]
        if navigate_to_chapter(driver, chapter_num):
            scrape_chapter(driver, website, novel_metadata['title'], chapter_num, save_dir=save_dir)
        return
    
    # Case 3: Scrape multiple chapters (individual or ranges)
    for chapter_num in chapters_to_scrape:
        if navigate_to_chapter(driver, chapter_num):
            scrape_chapter(driver, website, novel_metadata['title'], chapter_num, save_dir=save_dir)

def setup_driver():
    options = uc.ChromeOptions()
    driver = uc.Chrome(options=options)
    return driver


def load_config(config_file):
    """
    Load website configurations from a JSON file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def parse_arguments(websites):
    """
    Parse command-line arguments for the scraper.

    Args:
        websites (list): A list of dictionaries containing website configurations.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Scrape light novel chapters from multiple websites.")
    parser.add_argument(
        "--website",
        choices=[w["name"] for w in websites],
        required=False,
        default=None,
        help="Website to scrape",
        type=str,
    )
    parser.add_argument(
        "--novel",
        required=False,
        default=None,
        help="Novel slug/name",
        type=str,
    )
    parser.add_argument(
        "--chapter_num",
        type=int,
        required=False,
        default=None,
        help="Chapter number",
    )
    return parser.parse_args()

def get_missing_args(args, config):
    """
    Deals with missing arguments

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        argparse.Namespace: Updated arguments with missing values filled in.        
    """
    # Get the website name from args or config
    website_name = args.website.lower() if args.website else config.get("website", "").lower()
    # Find the matching website in the config, defaulting to the first website
    args.website = next((w for w in config["websites"] if w["name"].lower() == website_name), config["websites"][0])
    print(f"Selected `{args.website['name']}` website")
    
    if args.novel is None:
        args.novel = config["novel"] if config["novel"] else input("Enter the novel name: ")
    if args.chapter_num is None:
        args.chapter_num = config["chapter_num"] if config["chapter_num"] else str(input("Enter the chapter number: "))
    return args

def main():
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    config = load_config(config_path)
    websites = config["websites"]

    # Initialize the driver
    driver = setup_driver()

    try:
        # Get input args
        args = parse_arguments(websites)

        # Step 1: Select website, novel, and related variables
        args = get_missing_args(args, config)
        print("Input args selected")

        # Step 2: Search & select the novel
        # novels_found = search_novels(args.novel, website=args.website, driver=driver)
        # print(tabulate(novels_found, headers="keys", tablefmt="pretty"))
        selected_novel = select_novel(args.novel, website=args.website, driver=driver)
        if not selected_novel:
            print("Novel wasn't selected")
            return
        print(f"\nSelected: {selected_novel['novel']['title']}")

        # Step 3: Get novel metadata
        novel_metadata = get_novel_metadata(args.website, selected_novel['url'], driver=driver)
        if novel_metadata:
            print("\nNovel Metadata:")
            print(f"Title: {novel_metadata['title']}")
            print(f"Author: {novel_metadata['author']}")
            print(f"Description: {novel_metadata['description']}")
            print(f"Estimated Total Chapters: {novel_metadata['total_chapters']}")
            print(f"URL: {novel_metadata['url']}")
        else:
            print("Failed to fetch novel metadata.")
        
        # Step 4: Navigate to the last chapter
        navigate_to_latest_chapter(selected_novel['url'], driver)
        
        # Step 5: Find, scrape and save the chapter(s)
        scrape_chapters(driver, 
                        args.website, 
                        novel_metadata, 
                        chap_num=args.chapter_num, 
                        save_dir=config["save_dir"])

    except Exception as e:
        print(f"Exception: {str(e)}")
        return []

    finally:
        # Close the driver
         driver.quit()

if __name__ == "__main__":
    main()