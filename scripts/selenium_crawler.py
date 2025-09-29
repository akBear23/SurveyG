import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def download_with_selenium(pdf_url, file_path):
    """
    Download PDF from MDPI using Selenium with automatic ChromeDriver management
    
    Args:
        pdf_url (str): Direct PDF URL
        file_path (str): Full path including filename to save the PDF
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Set download directory to the target directory
    download_dir = os.path.dirname(file_path)
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": os.path.abspath(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    
    try:
        # Setup ChromeDriver automatically
        print("Setting up ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print(f"Accessing {pdf_url} with Selenium...")
        driver.get(pdf_url)
        
        # Wait for download to complete
        print("Waiting for download to complete...")
        time.sleep(15)  # Adjust based on file size and connection speed
        
        # Check if the file was downloaded
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ Downloaded: {file_path} ({file_size} bytes)")
            return file_path
        else:
            # Check for any downloaded PDF in the directory
            downloaded_files = [f for f in os.listdir(download_dir) if f.endswith('.pdf')]
            if downloaded_files:
                # Find the most recently downloaded PDF
                latest_file = max(
                    [os.path.join(download_dir, f) for f in downloaded_files],
                    key=os.path.getctime
                )
                # Rename it to the desired filename
                os.rename(latest_file, file_path)
                file_size = os.path.getsize(file_path)
                print(f"✅ Downloaded and renamed: {file_path} ({file_size} bytes)")
                return file_path
            else:
                print("❌ No PDF file found in download directory")
                return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None
    finally:
        try:
            driver.quit()
        except:
            pass


# Example usage
if __name__ == "__main__":
    pdf_url = "https://www.mdpi.com/2079-9292/14/4/725/pdf"
    downloaded_file = download_with_selenium(pdf_url, "pdfs/test.pdf")
    
    if downloaded_file:
        print(f"File available at: {os.path.abspath(downloaded_file)}")