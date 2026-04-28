import os
from anthropic import Anthropic


class ClaudeClient:
    def __init__(
        self,
        model="claude-3-haiku-20240307"
    ):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.model = model

    def generate(
        self,
        prompt,
        temperature=0,
        max_tokens=500
    ):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text.strip()