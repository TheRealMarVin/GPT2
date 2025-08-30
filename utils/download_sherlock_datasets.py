import requests
import os

from os import path

# URLs of Sherlock Holmes books from Project Gutenberg
books = {
    "A Study in Scarlet": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "The Sign of the Four": "https://www.gutenberg.org/files/2097/2097-0.txt",
    "The Adventures of Sherlock Holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "The Memoirs of Sherlock Holmes": "https://www.gutenberg.org/files/834/834-0.txt",
    "The Hound of the Baskervilles": "https://www.gutenberg.org/files/2852/2852-0.txt",
    "The Return of Sherlock Holmes": "https://www.gutenberg.org/files/108/108-0.txt",
    "The Valley of Fear": "https://www.gutenberg.org/files/3289/3289-0.txt",
    "His Last Bow": "https://www.gutenberg.org/files/2350/2350-0.txt",
    "The Case-Book of Sherlock Holmes": "https://www.gutenberg.org/files/221/221-0.txt"
}

def download_sherlock_dataset(out_folder, out_name, override_files=False):
    if not path.exists(out_folder):
        os.makedirs(out_folder)

    filename = os.path.join(out_folder, out_name)
    if path.exists(filename) and not override_files:
        return filename

    data = []

    # Download each book
    for title, url in books.items():
        response = requests.get(url)

        if response.status_code == 200:
            text = response.text
            text = text.replace("\r\n\r\n", "\n")
            text = text.replace("\r\n", " ")
            text = text.replace("  ", " ")
            data.append(text)
            print(f"Downloaded: {title}")
        else:
            print(f"Failed to download: {title}")

        with open(filename, "w", encoding="utf-8") as file:
            full_content = "\n".join(data)
            file.write(full_content)

    return filename

if __name__ == "__main__":
    download_sherlock_dataset("data", "test.txt")
