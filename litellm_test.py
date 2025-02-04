import litellm
import asyncio

async def main():
    try:
        response = await litellm.acompletion(
            model="ollama/llama3.2:3b",
            messages=[{"content": "Write a haiku about sunsets", "role": "user"}],
            api_base="http://localhost:11434", # Explicitly set api_base
            timeout=300 # Set timeout here as well
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 