from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config import GOOGLE_API_KEY
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image-preview",
    api_key=GOOGLE_API_KEY
)

def generate_improvements(question: str, answer: str,feedback: str):
    prompt_template = """
    You are an IELTS writing examiner.Suggest **specific and practical improvements** the student can make to reach a higher band.  

    Question: {question}
    Answer: {answer}
    Examiner Feedback: {feedback}

      Rules:
    - Address the student directly using "you" (e.g., "You should…")
    - Improvements must be practical and specific.
    - Give **at least 3-5 concrete improvements**.
    - Cover areas like task response, coherence, vocabulary, and grammar.
    - Keep each suggestion short (1 sentence max).
    -  Return ONLY a JSON object.
    - Do not include Markdown formatting, code fences, or the word "json".
    - JSON format must be:
    {{
  "improvements": [
    "<improvement 1>",
    "<improvement 2>",
    "<improvement 3>",
    "<improvement 4>",
    "<improvement 5>"
  ]
}}"""

    improvement_prompt = PromptTemplate(
        input_variables=["question", "answer", "feedback"],
        template=prompt_template
    )

    formatted_prompt = improvement_prompt.format(
        question=question,
        answer=answer,
        feedback=feedback
    )

    response = llm.invoke(formatted_prompt)
    print("Calling improvement LLM...")
    try:
        return json.loads(response.content)
    except Exception:
        return {"improvements": [response.content.strip() or "No improvements generated"]}
