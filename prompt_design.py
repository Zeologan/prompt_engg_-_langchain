#--- A strong prompt usually has these five parts --

# Role
# Who should the AI act as?
# "You are a career counsellor."

# Task
# What exactly should it do?
# "Suggest 3 career options."

# Context
# What background does it need?
# "The student is in final-year BCom and likes working with numbers."

# Format
# How should the answer look?
# "Give a table with option, reason and salary range."

# Constraints
# Any limits or rules?
# "Keep it under 100 words. Use simple English."

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0,
)


# ❌ Bad prompt: vague, no context, no format
# bad = model.invoke("Tell me about careers.")
# print(bad.content)

# print("-" * 50)


# ✅ Good prompt: role, task, context, format and constraints
# good = model.invoke([
#     SystemMessage("You are a career counsellor for Indian college students. Use simple English."),
#     HumanMessage(
#         "Suggest 3 career options for a final-year BCom student who enjoys working with numbers. "
#         "Give the answer as a table with columns: Career, Why it fits, Starting salary in India. "
#         "Keep it under 100 words."
#     ),
# ])
# print(good.content)


# Role / Persona Prompting

# question = "What is cloud computing?"

# roles = [
#     "You are a teacher explaining to a 10-year-old child. Use a fun example.",
#     "You are a senior cloud architect talking to a fellow engineer. Be technical and brief.",
#     "You are a sales person explaining to a small shop owner why it helps their business.",
# ]

# for role in roles:
#     response = model.invoke([
#         SystemMessage(role),
#         HumanMessage(question),
#     ])
#     print(response.content)
#     print("-" * 50)


persona = """You are Priya, a friendly interview coach with 10 years of experience in IT hiring.
You are helping freshers prepare for their first technical interview.
Rules:
- Encourage the student, never make them feel bad.
- Give practical tips, not theory.
- End every answer with one quick practice question."""

response = model.invoke([
    SystemMessage(persona),
    HumanMessage("How should I introduce myself in an interview?"),
])
print(response.content)
