import json
import os
import urllib.request

url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json"

def download_instruct_dataset_file(file_path, count=-1, override=False):
    if not os.path.exists(file_path) or override:
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)

    if count > 0 and count <= len(data):
        data = data[:count]

    return data


if __name__ == "__main__":
    filename = "../data/instruction-data.json"

    instruct_data = download_instruct_dataset_file(filename)
    print("Number of entries:", len(instruct_data))
