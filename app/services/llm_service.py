import os
from groq import Groq

class LLMService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)

        self.model_name = "llama-3.1-8b-instant"

    def generate_answer(self, question: str, retrieved_chunks):
        context = "\n\n".join(chunk[0].text for chunk in retrieved_chunks)
        
        prompt = f"""
Use the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content