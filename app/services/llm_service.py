import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_PROVIDER = os.getenv("HF_PROVIDER", "cerebras")


def ask_huggingface_llm(prompt: str) -> str:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing in .env")

    client = InferenceClient(
        provider=HF_PROVIDER,
        model=HF_MODEL,
        token=HF_TOKEN,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI codebase assistant. "
                "Answer only from the provided code context. "
                "If the answer is not in the context, say clearly that it was not found."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=700,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()