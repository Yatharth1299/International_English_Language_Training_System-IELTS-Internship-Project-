from config import GOOGLE_API_KEY
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,END
from typing import TypedDict, List
from agents.scoring_agent import combine_results,score_task
from agents.feedback_agent import generate_feedback
from agents.improvement_agent import generate_improvements




llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-image-preview",
    api_key=GOOGLE_API_KEY
)
state={
    "mode":"",
    "test_type":"",
    "task1_question":"",
    "task1_image":"",
    "task2_question":"",
    "task1_answer":"",
    "task2_answer":""
    
}

Task1_template="""
You are an IELTS test generator.
The user selected: {test_type} {mode}.
Generate **one IELTS Writing Task 1 question** that looks like a real exam question.

If Academic:
- Generate a one clear **chart, graph, table, map, or process diagram** as an image that matches the question.
- Avoid repeating the same topic everytime.
- Include numbers, trends, or comparisons in the data.
- Use IELTS-style wording like: 
  "Summarize the information by selecting and reporting the main features, and make comparisons where relevant."


If General Training:
- Include a context for the letter (why it is written).
- Specify recipient type (friend, manager, company, council, etc.).
- Specify purpose (request, complaint, apology, suggestion, enquiry, etc.).
- Do NOT include "Begin with Dear Sir/Madam".
- ABSOLUTELY ONLY return the question text and bullet points.
- Use IELTS-style wording like: 
  "Write a letter to ... explaining the situation and what action you would like to be taken."

Do NOT include the answer.
Output:
DO not include any line breaks (\n or \n\n) in your response.
- For Academic: Return **the question text and the generated chart image**.
- For General Training: Return **the question text**.

"""
task1_prompt=PromptTemplate(
    input_variables=["test_type","mode"],
    template=Task1_template
)

Task2_template="""
You are an IELTS examiner.
The user is preparing for the IELTS {test_type} Writing {mode} test.

Your job:
- Generate one IELTS Writing Task 2 question.
- Task 2 always essay question
- The question must realistic and written in authentic IELTS exam style.
- Do not provide answer,only the question.
- Categories are below:
  1.Agree/Disagree
  2.Advantages/Disadvantages
  3.Causes/Effects
  4.Causes/Solutions
  5.Discuss both views and give your own opinion
  6.Double question

If Academic - use more formal and academic-style topics.
If General Training - use more practical, everyday life topics.

Topics(choose ONE topic yourself BELOW):
Education, Technology, Environment, Health, Government spending, Socialogy, Work & Employment, Culture & Globalization, Family & Children, Media & Advertising.
Final Output:
DO not include any line breaks (\n or \n\n) in your response.
Write only the Task 2 question text
The question must end with:
Give reasons for your answer and include any relevant examples from your own knowledge or experience."""

task2_prompt=PromptTemplate(
    input_variables=["test_type","mode"],
    template=Task2_template
)



def evaluate_task(request):
    # 1. Score
    task1_result, task2_result = None, None

    # --- Task 1 ---
    if request.test_type == "academic" and request.task1_answer and request.task1_image:
        task1_result = score_task("task1", request.test_type, request.task1_question, request.task1_answer, request.task1_image)
    elif request.test_type == "general training" and request.task1_answer:
        task1_result = score_task("task1", request.test_type, request.task1_question, request.task1_answer)

    # --- Task 2 ---
    if request.task2_answer and request.task2_question:
        task2_result = score_task("task2", request.test_type, request.task2_question, request.task2_answer)

    if not task1_result and not task2_result:
        return {"error": "No valid tasks submitted"}

    
    if task1_result and task2_result:
        final_result = combine_results(task1_result, task2_result)
    elif task2_result:
        final_result = task2_result
    else:
        final_result = task1_result
    final_band = final_result["band"]
    print("final score",final_band)

 


    
    # 2. Feedback
    combined_question = ""
    combined_answer = ""

    if request.task1_question and request.task1_answer:
        combined_question += f"Task 1 Question: {request.task1_question}\n"
        combined_answer += f"Task 1 Answer: {request.task1_answer}\n\n"

    if request.task2_question and request.task2_answer:
        combined_question += f"Task 2 Question: {request.task2_question}\n"
        combined_answer += f"Task 2 Answer: {request.task2_answer}\n"

    # --- Feedback ---
    feedback_obj = generate_feedback(combined_question, combined_answer, final_band)
    feedback = feedback_obj.get("feedback", "")
    print("feedback_combined",feedback)

    # 3. Improvements
    improvement_obj = generate_improvements(combined_question, combined_answer, feedback)
    improvements = improvement_obj.get("improvements", [])
    print(improvements)

    return {
        "band": final_band,
        "feedback": feedback,
        "improvements": improvements
    }
