import os
import requests

# Directly hit Llama
# url = "https://model-8w6z6553.api.baseten.co/development/predict"

# Hit Chain (that hits Llama as stub)
url = "https://chain-v31d613z.api.baseten.co/development/run_remote"

headers = {"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
payload = {"text" : "this is a test"}

with requests.post(url, headers=headers, json=payload, stream=True) as response:

    response.raise_for_status()

    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            print(chunk)
