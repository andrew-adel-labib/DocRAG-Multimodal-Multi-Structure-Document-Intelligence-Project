from transformers import pipeline

judge = pipeline("text-generation", model="gpt2")

def llm_judge(question, pred, truth):
    try:
        prompt = f"""
        Question: {question}
        Ground Truth: {truth}
        Prediction: {pred}

        Is the prediction correct? yes or no
        """

        out = judge(prompt, max_length=50)[0]["generated_text"]
        return "yes" in out.lower()

    except:
        return False