import json
import os

# Load rubrics
RUBRICS_PATH = r"C:\IELTS_modules_app\data\prompts\rubrics\Band_descriptors.json"

try:
    with open(RUBRICS_PATH, encoding="utf-8") as f:
        rubrics = json.load(f)
        print("Rubrics JSON loaded successfully")
except json.JSONDecodeError as e:
    print("JSON error:", e)
    rubrics = {}
except FileNotFoundError:
    print(f"File not found: {RUBRICS_PATH}")
    rubrics = {}

def get_rubric(task_type: str, test_type: str = None):
    """
    Fetch the correct rubric section for a task.
    - task_type: "task1" or "task2"
    - test_type: "academic" or "general training" (only needed for task1)
    """
    if task_type == "task1":
        if not test_type:
            raise ValueError("test_type is required for task1 rubric lookup")
        test_type_key = test_type.lower().replace(" ", "_")
        return rubrics["task1"].get(test_type_key, {})
    elif task_type == "task2":
        return rubrics.get("task2", {})
    else:
        raise ValueError(f"Invalid task_type: {task_type}")
