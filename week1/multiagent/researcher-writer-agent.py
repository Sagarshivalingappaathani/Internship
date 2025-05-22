from openai import OpenAI

FARM_KEY = "0c9f370035e0436989bb962b0d1bb9d0"
BASE_URL = "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18"

client = OpenAI(
    api_key="dummy",  
    base_url=BASE_URL,
    default_headers={"genaiplatform-farm-subscription-key": FARM_KEY}
)

def ask_agent(role_name, system_prompt, user_message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are {role_name}. {system_prompt}"},
            {"role": "user", "content": user_message}
        ],
        extra_query={"api-version": "2024-08-01-preview"}
    )
    return response.choices[0].message.content

# Step 1: Researcher agent
topic = "How electric vehicles work"
research_output = ask_agent(
    "a researcher",
    "Your job is to gather factual information in bullet points.",
    f"Find important facts about: {topic}"
)

print("\n🔍 Researcher found:")
print(research_output)

# Step 2: Writer agent
writer_output = ask_agent(
    "a writer",
    "You write a short paragraph based on given research.",
    f"Write a paragraph based on these notes:\n{research_output}"
)

print("\n📝 Writer generated:")
print(writer_output)
