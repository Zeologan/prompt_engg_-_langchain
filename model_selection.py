import os
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

load_dotenv()

small_model = init_chat_model(model=os.getenv("MODEL_SMALL"), model_provider="groq", temperature=0)
large_model = init_chat_model(model=os.getenv("MODEL_LARGE"), model_provider="groq", temperature=0)


#  - - - Step 1: Compare Models on the Same Test Set

# test_set = [
#     {"text": "Money deducted but order shows failed.", "expected": "Urgent"},
#     {"text": "Parcel is 2 days late.", "expected": "Normal"},
#     {"text": "Please add UPI autopay option.", "expected": "Low"},
#     {"text": "Refund not received after 12 days.", "expected": "Urgent"},
#     {"text": "Delivery boy came one day late.", "expected": "Normal"},
#     {"text": "Your app colours look boring.", "expected": "Low"},
# ]

# system = SystemMessage(
#     "Classify the complaint. Money issues are Urgent, delivery delays are Normal, "
#     "suggestions are Low. Reply with only one word: Urgent, Normal or Low."
# )

# def evaluate(model, name):
#     correct, total_tokens = 0, 0
#     start = time.time()

#     for item in test_set:
#         response = model.invoke([system, HumanMessage(item["text"])])
#         answer = response.content.strip().rstrip(".")
#         if answer.lower() == item["expected"].lower():
#             correct += 1
#         total_tokens += response.usage_metadata["total_tokens"]

#     seconds = time.time() - start
#     print(f"{name:6} → Accuracy: {correct}/{len(test_set)} | Tokens: {total_tokens} | Time: {seconds:.2f} sec")

# evaluate(small_model, "Small")
# evaluate(large_model, "Large")


# - - - Step 2: Use Different Models for Different Jobs


# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# parser = StrOutputParser()

# # Easy job → small model
# classify = ChatPromptTemplate.from_messages([
#     ("system", "Classify the complaint as Urgent, Normal or Low. One word only."),
#     ("human", "{complaint}"),
# ]) | small_model | parser

# # Hard job → large model
# reply = ChatPromptTemplate.from_messages([
#     ("system", "You are a polite customer support agent. Write a short, caring reply. "
#                "If priority is Urgent, promise a callback within 2 hours."),
#     ("human", "Complaint: {complaint}\nPriority: {priority}"),
# ]) | large_model | parser

# support_chain = RunnablePassthrough.assign(priority=classify) | reply

# print(support_chain.invoke({"complaint": "₹2,500 deducted twice for one order. Very angry!"}))


# - - - Step 3: Add a Backup Model (Fallbacks)

# main_model = init_chat_model(model=os.getenv("MODEL_LARGE"), model_provider="groq", max_retries=1)
# backup_model = init_chat_model(model=os.getenv("MODEL_SMALL"), model_provider="groq")

# safe_model = main_model.with_fallbacks([backup_model])

# response = safe_model.invoke("Give one tip to save money as a college student.")
# print(response.content)


# Cost for One Request 

# Sample prices in USD per 1 million tokens (replace with current real prices)
PRICES = {
    "small": {"input": 0.05, "output": 0.08},
    "large": {"input": 0.59, "output": 0.79},
}

USD_TO_INR = 88   # Sample rate, check the current rate

def estimate_cost(input_tokens, output_tokens, model_type):
    price = PRICES[model_type]
    usd = (input_tokens / 1_000_000 * price["input"]) + (output_tokens / 1_000_000 * price["output"])
    return usd * USD_TO_INR

response = large_model.invoke("Explain mutual funds to a beginner in 5 lines.")
u = response.usage_metadata

cost = estimate_cost(u["input_tokens"], u["output_tokens"], "large")
print(f"Tokens: {u['total_tokens']} | Cost: ₹{cost:.5f}")



# Monthly Cost for a Real App
def monthly_cost(avg_input, avg_output, requests_per_day, model_type, days=30):
    per_request = estimate_cost(avg_input, avg_output, model_type)
    return per_request * requests_per_day * days

# Measured averages from testing (example values)
avg_input = 600     # Includes system prompt + history + question
avg_output = 250
requests_per_day = 5_000   # 500 students × 10 questions each

for model_type in ["small", "large"]:
    total = monthly_cost(avg_input, avg_output, requests_per_day, model_type)
    print(f"{model_type.capitalize():6} model → ₹{total:,.0f} per month")

# Sample free plan limits (check your real limits in the Groq console)
free_rpm = 30
free_rpd = 1_000
free_tpm = 6_000

avg_tokens_per_request = avg_input + avg_output   # 850 from our example

max_by_tokens = free_tpm // avg_tokens_per_request
max_per_minute = min(free_rpm, max_by_tokens)

print(f"Max requests per minute: {max_per_minute}")
print(f"Max requests per day:    {free_rpd}")
print(f"Daily need for our app:  {requests_per_day}")
print("Free plan enough?", "Yes" if requests_per_day <= free_rpd else "No, you need a paid plan")
