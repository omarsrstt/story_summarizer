import cloudscraper
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
        time.sleep(7)
        
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
        time.sleep(2.5)  # Wait for the search bar
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
        all_results = []
        
        # Search for novels
        results = search_novels(novel_name, website, driver)
        all_results.extend(results)
            
        # Handle no results case
        if not all_results:
            print("No novels found with that name.")
            return None
            
        # Print results
        print(f"Found {len(all_results)} matches:")
        novels_found = [{"#": i, **result} for i, result in enumerate(all_results, 1)]
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
    if args.chapter is None:
        args.chapter = config["chapter"] if config["chapter"] else input("Enter the chapter number: ")
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
            return
        print(f"\nSelected: {selected_novel['novel']['title']}")

    except Exception as e:
        print(f"Exception: {str(e)}")
        return []

    finally:
        # Close the driver
        driver.quit()

if __name__ == "__main__":
    main()