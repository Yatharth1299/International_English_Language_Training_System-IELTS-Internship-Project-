from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config import GOOGLE_API_KEY
from services.evaluation_service import get_rubric
import json

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image-preview",
    api_key=GOOGLE_API_KEY)


def score_task(task_type: str, test_type: str, question: str, answer: str = None, image_b64: str = None):
    rubric_type=get_rubric(task_type,test_type)

    prompt_template_score = """You are an expert IELTS examiner.Evaluate the following IELTS Writing {task_type} answer.
    Question: {question}
    Your answer: {answer}
    Band descriptors to guide scoring:{rubric_type}
    Return a valid JSON object, exactly in this format:
    {{"band": <float 0.0-9.0, step 0.5>}}
    
    Rules:
    - Return ONLY the JSON object
    - Do not include the word "json" anywhere
    - Do not include line breaks inside JSON
    - Format example: {{"band": 6.5}}
"""
    

    
    score_prompt = PromptTemplate(
    input_variables=["task_type", "question", "answer", "rubric_type"],
    template=prompt_template_score)
    
    formatted_prompt = score_prompt.format(
        task_type=task_type,
        question=question,
        answer=answer if answer else "[Answer provided in image]",   #format the text even if image is present
        rubric_type=json.dumps(rubric_type, ensure_ascii=False, indent=2)
    )
    print("format prompt",formatted_prompt)
    
    if not image_b64:
        response = llm.invoke(formatted_prompt)
        return json.loads(response.content)
    #text+image
    response = llm.invoke(
        [
            {"role": "user", "content": [
                {"type": "text", "text": formatted_prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"}
            ]}
        ]
    )
    print("successfully sent response")
    return json.loads(response.content)


def combine_results(task1_result: dict, task2_result: dict):
    prompt_template = """
    You are an IELTS examiner. Combine the Task 1 and Task 2 evaluations into a single final assessment.

    Task 1:
    {task1}

    Task 2:
    {task2}

    Rules:
    - Task 2 is weighted more heavily than Task 1 (about 2x).
    - Final band must be between 0.0 and 9.0 (step 0.5).
    
    
    Return a valid JSON object in this format:
    {{
      "band": <float 0.0-9.0, step 0.5>
      
    }}
    Rules:
    - Return ONLY the JSON object
    - Do not include the word "json" anywhere
    - Do not include line breaks inside JSON
    - Format example: {{"band": 6.5}}

    """
    prompt = PromptTemplate(
        input_variables=["task1","task2"],
        template=prompt_template
    )
    formatted_prompt = prompt.format(
        task1=json.dumps(task1_result, ensure_ascii=False),
        task2=json.dumps(task2_result, ensure_ascii=False)
    )

    print("final prompt",formatted_prompt)
    print("Calling scoring LLM...")
    response = llm.invoke(formatted_prompt)
    return json.loads(response.content)


