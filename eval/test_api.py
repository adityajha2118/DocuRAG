from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
c = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

models_to_try = [
    "meta/llama-4-scout-17b-16e-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "google/gemma-3-27b-it",
    "mistralai/mistral-small-24b-instruct",
    "qwen/qwen2.5-72b-instruct",
    "deepseek-ai/deepseek-v4.1-flash",
]

for m in models_to_try:
    try:
        r = c.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "Reply ONLY with: OK"}],
            temperature=0,
            max_tokens=10,
        )
        content = r.choices[0].message.content
        print(f"  OK  {m} -> {repr(content)}")
    except Exception as e:
        err = str(e)[:80]
        print(f"  ERR {m} -> {err}")
