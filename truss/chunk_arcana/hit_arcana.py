import os
import requests

url = "https://chain-dq41xdq2.api.baseten.co/development/run_remote"

headers = {"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
payload = {"text" : "this is a test. this is another test.", "speaker": "luna"}

with requests.post(url, headers=headers, json=payload, stream=True) as response:

    response.raise_for_status()

    with open("/workspace/tmp/baseten-two-sentence-chunked-arcana-audio.wav", "wb") as f:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
