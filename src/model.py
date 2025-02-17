import openai
from openai import OpenAI
import os
import requests
import json

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import time

# client = OpenAI(api_key=openai.api_key)

def call_chat_gpt(message, args):
    wait = 1
    while True:
        try:
            ans = client.chat.completions.create(model=args.model,
            max_tokens=args.max_tokens,
            messages=message,
            temperature=args.temperature,
            n=1)
            return ans.choices[0].message.content, ans.usage.prompt_tokens, ans.usage.completion_tokens
        except openai.RateLimitError as e:
            time.sleep(min(wait, 60))
            wait *= 2

def query_firework(message, args, model="deepseek-v3"):
    api_key = os.environ.get("FIREWORK_API_KEY")

    if model == "deepseek-v3":

        url = "https://api.fireworks.ai/inference/v1/chat/completions"

        payload = {
            "model": f"accounts/fireworks/models/{model}",
            "max_tokens": 16384,
            "temperature": args.temperature,
            "messages": message
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        response = requests.request("POST", url, json=payload, headers=headers)

        if response.status_code == 200:
            try:
                data = response.json()
                # print(data)
                # Extract content
                content = data["choices"][0]["message"]["content"]
                input_token = data["usage"]["prompt_tokens"]
                output_token = data["usage"]["completion_tokens"]
                return content, input_token, output_token
            except json.JSONDecodeError as e:
                # Return an error message if JSON decoding fails
                return f"JSONDecodeError: {e} - Response text: {response.text}"
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    elif model == "starcoder":
        url = "https://api.fireworks.ai/inference/v1/completions"
        payload = {
        "model": "accounts/chungyuwang5507-f1662b/deployedModels/starcoder2-15b-bb7b2085",
        "max_tokens": 2048,
        "temperature": args.temperature,
        "prompt": message[1]['content']
        }
        headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            try:
                data = response.json()
                print(data)
                # Extract content
                content = data["choices"][0]["text"]
                input_token = data["usage"]["prompt_tokens"]
                output_token = data["usage"]["completion_tokens"]
                return content, input_token, output_token
            except json.JSONDecodeError as e:
                # Return an error message if JSON decoding fails
                return f"JSONDecodeError: {e} - Response text: {response.text}"
        else:
            return f"Error: {response.status_code}, {response.text}"
        
    

    
    

def get_embedding(text, model='text-embedding-3-large'):
    response = client.embeddings.create(input=text, model = model)
    return response.data[0].embedding