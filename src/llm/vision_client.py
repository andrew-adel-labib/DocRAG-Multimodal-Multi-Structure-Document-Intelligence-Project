from openai import OpenAI


class VisionClient:
    def __init__(
        self,
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2-VL-7B-Instruct"
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )

        self.model = model

    def analyze_image(
        self,
        image_path,
        question
    ):
        prompt = f"Analyze this image and answer: {question}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()