import argparse
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium_stealth import stealth
import undetected_chromedriver as uc
from pprint import pprint
import time
import json


def search_novels(query, website):
    """Search for novels by name using Selenium and return matching results"""
    driver = setup_driver()
    
    try:
        # Navigate to the search page
        driver.get(website["search_url"])
        print(f"Loaded search page: {website['search_url']}")
        time.sleep(8)
        
        # Find the search input field
        print("Waiting for search input field...")
        search_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, website["search_input_selector"]))
        )
        print("Found search input field.")
        
        # Enter the query into the search bar
        print(f"Entering query: {query}")
        search_input.clear()
        search_input.send_keys(query)
        search_input.send_keys(Keys.RETURN)
        print(f"Submitted search query: {query}")
        
        # Wait for the search results to load
        time.sleep(5)  # Wait for the search bar
        print("Waiting for search results...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, website["search_item_selector"]))
        )
        print("Search results loaded successfully.")
        
        # Scroll to the bottom of the page to load all results (if needed)
        print("Scrolling to load more results...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # Wait for additional content to load
        
        # Parse the page source
        print("Parsing page source...")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        print(f"Page source length: {len(driver.page_source)}")
        
        results = []
        for item in soup.select(website["search_item_selector"]):
            print(f"Processing item: {item}")
            
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
            print(f"Added result: {results[-1]}")
            
        return results
        
    except Exception as e:
        print(f"Search failed on {website['name']}: {str(e)}")
        return []
    
    finally:
        print(driver.page_source)
        driver.quit()  # Close the browser when done


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
        "--chapter",
        type=int,
        required=False,
        default=None,
        help="Chapter number",
    )
    return parser.parse_args()

def get_missing_args(args, config):
    """
    Prompt the user for missing arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        argparse.Namespace: Updated arguments with missing values filled in.
    """
    if args.novel is None:
        args.novel = config["novel"] if config["novel"] else input("Enter the novel name: ")
    if args.chapter is None:
        args.chapter = config["chapter"] if config["chapter"] else input("Enter the chapter number: ")
    return args

def main():
    # Load config
    config = load_config("config.json")
    websites = config["websites"]

    # Get input args
    args =  parse_arguments(websites)

    # Handle missing arguments
    args = get_missing_args(args)

    # Step 1: Select the website
    website = next((w for w in websites if w["name"].lower() == (args.website.lower() if args.website else "")), websites[0])
    print(f"Selected `{website['name']}` website")
    pprint(website)

    # Step 2: Search for the novel
    print(search_novels(args.novel, website=website))

if __name__ == "__main__":
    main()