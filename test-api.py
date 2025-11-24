import json

import requests

port = 12000
prompt = "The capital of France is"

response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 512,
        },
        "stream": True,
    },
    stream=True,
)

print(f"Prompt: {prompt}")
print(prompt, end="")

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    if chunk and chunk.startswith("data:"):
        if chunk == "data: [DONE]":
            break
        data = json.loads(chunk[5:].strip("\n"))
        output = data["text"]
        print(output[prev:], end="", flush=True)
        prev = len(output)
