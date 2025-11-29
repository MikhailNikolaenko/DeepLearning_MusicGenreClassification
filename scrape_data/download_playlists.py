import json
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc


def setup_driver(download_dir: str):
    """
    Setup Chrome driver with custom download directory and Cloudflare bypass
    """
    chrome_options = uc.ChromeOptions()
    
    # Set download directory
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    
    # Add arguments to appear more human-like
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Use undetected-chromedriver to bypass Cloudflare
    driver = uc.Chrome(options=chrome_options, version_main=None)
    
    # Additional stealth settings
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """
    })
    
    return driver


def download_playlist(driver, playlist_url: str, genre: str, download_dir: str):
    """
    Download a single playlist from spotidownloader.com
    
    Args:
        driver: Selenium WebDriver instance
        playlist_url: Spotify playlist URL
        genre: Genre name for the playlist
        download_dir: Directory where files will be downloaded
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {genre}")
        print(f"URL: {playlist_url}")
        print(f"{'='*60}")
        
        # Navigate to spotidownloader
        driver.get("https://spotidownloader.com/en8")
        
        # Wait for page to load
        time.sleep(2)
        
        # Find the input field (adjust selector based on actual page structure)
        try:
            # Common selectors for URL input fields
            input_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], input[type='url'], input[placeholder*='URL'], input[placeholder*='url'], input[placeholder*='link']"))
            )
        except TimeoutException:
            # Try finding by ID or name
            input_field = driver.find_element(By.TAG_NAME, "input")
        
        # Clear and enter the playlist URL
        input_field.clear()
        input_field.send_keys(playlist_url)
        print(f"Entered URL into input field")
        
        # Find and click the download/submit button
        time.sleep(1)
        try:
            # Try common button selectors
            download_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit'], input[type='submit'], .download-btn, #download-btn")
        except NoSuchElementException:
            # Find any button on the page
            buttons = driver.find_elements(By.TAG_NAME, "button")
            if buttons:
                download_button = buttons[0]  # Use first button
            else:
                download_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
        
        download_button.click()
        print(f"✓ Clicked download button")
        
        # Wait for user to complete CAPTCHA
        print(f"⏳ Please complete CAPTCHA if present...")
        print(f"   Waiting 30 seconds for CAPTCHA verification...")
        time.sleep(10)  # Give user time to complete CAPTCHA
        
        # After CAPTCHA, find and click "Download ZIP" button
        print(f"🔍 Looking for 'Download ZIP' button...")
        try:
            # Try multiple selectors for "Download ZIP" button
            zip_button = None
            
            # Try by button text containing "ZIP"
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                if "zip" in btn.text.lower() or "download" in btn.text.lower():
                    zip_button = btn
                    print(f"✓ Found button with text: {btn.text}")
                    break
            
            # If not found, try by common selectors
            if not zip_button:
                selectors = [
                    "button.download-zip",
                    "button#download-zip",
                    "a.download-zip",
                    "a#download-zip",
                    ".btn-download",
                    "#btn-download"
                ]
                for selector in selectors:
                    try:
                        zip_button = driver.find_element(By.CSS_SELECTOR, selector)
                        print(f"✓ Found button with selector: {selector}")
                        break
                    except NoSuchElementException:
                        continue
            
            if zip_button:
                zip_button.click()
                print(f"✓ Clicked 'Download ZIP' button")
                
                # Wait a moment for the Continue button to appear
                time.sleep(2)
                
                # Now find and click "Continue" button
                print(f"🔍 Looking for 'Continue' button...")
                try:
                    continue_button = None
                    
                    # Try by button text containing "Continue"
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    for btn in buttons:
                        if "continue" in btn.text.lower():
                            continue_button = btn
                            print(f"✓ Found button with text: {btn.text}")
                            break
                    
                    # Try <a> tags as well
                    if not continue_button:
                        links = driver.find_elements(By.TAG_NAME, "a")
                        for link in links:
                            if "continue" in link.text.lower():
                                continue_button = link
                                print(f"✓ Found link with text: {link.text}")
                                break
                    
                    # If not found, try by common selectors
                    if not continue_button:
                        selectors = [
                            "button.continue",
                            "button#continue",
                            "a.continue",
                            "a#continue",
                            ".btn-continue",
                            "#btn-continue"
                        ]
                        for selector in selectors:
                            try:
                                continue_button = driver.find_element(By.CSS_SELECTOR, selector)
                                print(f"✓ Found button with selector: {selector}")
                                break
                            except NoSuchElementException:
                                continue
                    
                    if continue_button:
                        continue_button.click()
                        print(f"✓ Clicked 'Continue' button")
                    else:
                        print(f"⚠️  Could not find 'Continue' button")
                
                except Exception as e:
                    print(f"⚠️  Error clicking Continue: {str(e)}")
                
            else:
                print(f"⚠️  Could not find 'Download ZIP' button, continuing anyway...")
        
        except Exception as e:
            print(f"⚠️  Error clicking Download ZIP: {str(e)}")
        
        # Wait for ZIP file to appear in downloads directory
        print(f"⏳ Waiting for ZIP file to appear in downloads...")
        max_wait_time = 15*60  # Maximum wait time in seconds (15 minutes)
        check_interval = 2   # Check every 2 seconds
        elapsed_time = 0
        zip_found = False
        initial_files = set(os.listdir(download_dir))
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            current_files = set(os.listdir(download_dir))
            new_files = current_files - initial_files
            
            # Check if any new ZIP file appeared
            for file in new_files:
                if file.endswith('.zip') and not file.endswith('.crdownload'):
                    print(f"✓ ZIP file detected: {file} (after {elapsed_time}s)")
                    zip_found = True
                    break
            
            if zip_found:
                break
            
            # Show progress every 10 seconds
            if elapsed_time % 10 == 0:
                print(f"   Still waiting... ({elapsed_time}s elapsed)")
        
        if not zip_found:
            print(f"⚠️  No ZIP file detected after {max_wait_time} seconds")
        
        # Small buffer to ensure download is complete
        time.sleep(2)
        
        # Check if file was downloaded
        files = os.listdir(download_dir)
        print(f"✓ Files in download directory: {len(files)}")
        
        # Find the most recently downloaded file
        if files:
            newest_file = max([os.path.join(download_dir, f) for f in files], key=os.path.getctime)
            print(f"Most recent file: {os.path.basename(newest_file)}")
            
            # Rename file with genre name if it's not already named
            file_ext = os.path.splitext(newest_file)[1]
            new_name = os.path.join(download_dir, f"{genre}{file_ext}")
            
            # If file with genre name exists, add number
            counter = 1
            while os.path.exists(new_name):
                new_name = os.path.join(download_dir, f"{genre}_{counter}{file_ext}")
                counter += 1
            
            os.rename(newest_file, new_name)
            print(f"✓ Renamed to: {os.path.basename(new_name)}")
        
        print(f"✓ Completed processing for {genre}\n")
        return True
        
    except Exception as e:
        print(f"Error downloading {genre}: {str(e)}")
        return False


def main():
    # Load the playlists JSON
    json_file = "everynoise_playlists.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    playlists = data.get("playlists", {})
    
    if not playlists:
        print("No playlists found in JSON file!")
        return
    
    print(f"Found {len(playlists)} playlists to download")
    
    # Create downloads directory
    download_dir = os.path.abspath("downloads")
    os.makedirs(download_dir, exist_ok=True)
    print(f"Download directory: {download_dir}\n")
    
    # Setup Selenium driver
    print("Setting up Chrome driver...")
    driver = setup_driver(download_dir)
    
    try:
        successful = 0
        failed = 0
        
        for genre, playlist_url in playlists.items():
            success = download_playlist(driver, playlist_url, genre, download_dir)
            if success:
                successful += 1
            else:
                failed += 1
            
            # Small delay between downloads with progress indicator
            if genre != list(playlists.keys())[-1]:  # If not the last item
                print(f"⏳ Waiting 5 seconds before next download...")
                time.sleep(5)
        
        print(f"\n{'='*60}")
        print(f"Download Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(playlists)}")
        print(f"{'='*60}")
        
    finally:
        # Close the browser
        print("\nClosing browser...")
        driver.quit()
        print("Done!")


if __name__ == "__main__":
    main()
