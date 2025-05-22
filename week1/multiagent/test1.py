import os
from openai import OpenAI
import re

# Bosch LLM Farm credentials
FARM_KEY = "0c9f370035e0436989bb962b0d1bb9d0"
BASE_URL = "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18"

# Create OpenAI client for Bosch LLM Farm
client = OpenAI(
    api_key="dummy",
    base_url=BASE_URL,
    default_headers={"genaiplatform-farm-subscription-key": FARM_KEY}
)

# Create /code directory
os.makedirs("code", exist_ok=True)

# === LLM Call Wrapper ===
def ask_llm(system_msg, user_msg):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        extra_query={"api-version": "2024-08-01-preview"}
    )
    return response.choices[0].message.content.strip()

# === Step 1: UserAgent gives task ===
task = "Write a Python function that returns the factorial of a number."
print(f"\n👤 UserAgent Task: {task}")

# === Step 2: CoderAgent writes code ===
coder_prompt = f"Write only the Python code to solve this: {task}\nDo not include explanations."
code = ask_llm("You are a Python coding assistant.", coder_prompt)

# Remove markdown code fences and stray 'python' lines
code = re.sub(r"```(?:python)?\\n|```", "", code).strip()
code_lines = code.splitlines()
code = "\n".join(line for line in code_lines if line.strip().lower() != "python").strip()

print(f"\n💻 CoderAgent wrote code:\n{code}")

# === Step 3: Save to /code/test.py ===
code_path = "code/test.py"

# Append execution logic to the code
code += "\n\nif __name__ == '__main__':\n    print(factorial(5))\n"

with open(code_path, "w") as f:
    f.write(code)
print(f"\n📁 Code saved to: {code_path} with execution logic")

# === Step 4: ExecutorAgent simulates running it ===
executor_prompt = f"""You are a Python code runner.
The user gave you this code:

{code}

Show only the expected output of running this file."""
output = ask_llm("You are a Python execution assistant.", executor_prompt)
print(f"\n🧪 ExecutorAgent Expected Output:\n{output}")

# === Step 5: Actually run the saved script ===
print("\n🚀 Real Execution Output:")
os.system(f"python3 {code_path}")
