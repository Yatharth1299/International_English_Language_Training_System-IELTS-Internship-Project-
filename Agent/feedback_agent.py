from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config import GOOGLE_API_KEY
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image-preview",
    api_key=GOOGLE_API_KEY
)

def generate_feedback(question: str, answer: str, band: float):
    prompt_template = """
    You are an experienced IELTS examiner. Your task is to provide **examiner-style feedback** directly to the student in an interactive way (use "you").
    The feedback must cover **Task 1, Task 2, and an overall comment**, referring to the band score.  

    Question: {question}
    Answer: {answer}
    Band Score: {band}

    Rules:
    - Feedback must be clear, concise, and examiner-style.
    - Focus on strengths and weaknesses across BOTH tasks.
    - Mention how the band score reflects performance.
    - Write feedback as a single plain text string (no JSON inside).
    - Do not include Markdown formatting, code fences, or the word "json".
    - Do not include extra quotation marks, braces, or line breaks inside the JSON structure.
    - Return ONLY valid JSON in this exact format:

    {{
      "feedback": "Your examiner-style feedback here as one plain string"
    }}
    """

    feedback_prompt = PromptTemplate(
        input_variables=["question", "answer", "band"],
        template=prompt_template
    )

    formatted_prompt = feedback_prompt.format(
        question=question,
        answer=answer,
        band=band,
    )

    print("Calling feedback LLM...")
    response = llm.invoke(formatted_prompt)
    raw_output = getattr(response, "content", "").strip()

    # Clean fenced code output if present
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        raw_output = raw_output.replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw_output)

        # If feedback is itself a JSON string, unwrap it
        if isinstance(parsed.get("feedback"), str):
            try:
                inner = json.loads(parsed["feedback"])
                if isinstance(inner, dict) and "feedback" in inner:
                    parsed["feedback"] = inner["feedback"]
            except json.JSONDecodeError:
                pass

        return parsed
    except Exception:
        return {"feedback": raw_output or "Unable to generate feedback."}
