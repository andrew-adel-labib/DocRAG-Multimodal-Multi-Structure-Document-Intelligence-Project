from openai import OpenAI


class VLLMClient:
    def __init__(
        self,
        base_url="http://localhost:8000/v1",
        model="meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )

        self.model = model

    def generate(
        self,
        prompt,
        temperature=0,
        max_tokens=500
    ):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()