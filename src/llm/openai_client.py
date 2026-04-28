from openai import OpenAI


class OpenAIClient:
    def __init__(
        self,
        model="gpt-4o-mini"
    ):
        self.client = OpenAI()
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