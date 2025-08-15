import requests

def download_and_read_text(url):
    """
    Downloads text content from a given URL and returns it as a string.

    Args:
        url (str): The URL of the text file to download.

    Returns:
        str: The content of the text file, or an empty string if download fails.
    """
    try:
        response = requests.get(url)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        return ""

# Main execution block
if __name__ == '__main__':
    # The URL for the raw text of "The Verdict"
    file_url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    print("Downloading dataset...")
    raw_text = download_and_read_text(file_url)

    if raw_text:
        # Print the first 100 characters to verify
        print("Successfully downloaded dataset.")
        print("First 100 characters:\n", raw_text[:100])
        print("\nTotal number of characters:", len(raw_text))