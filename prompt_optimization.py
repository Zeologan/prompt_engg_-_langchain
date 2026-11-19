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

def measure(label, response):
    u = response.usage_metadata
    print(f"{label} → input: {u['input_tokens']}, output: {u['output_tokens']}, total: {u['total_tokens']}")



# - - - Part A: Prompt Optimization
#  - - - Technique 1: Remove Filler From the System Prompt

# long_system = SystemMessage(
#     "Hello! You are a very helpful, kind and friendly assistant. Your job is to help users "
#     "with their questions about personal finance. Please always make sure that you answer "
#     "in a way that is simple and easy to understand. It is very important that you use "
#     "simple words. Please do not use difficult or complicated words. Always try your best "
#     "to keep the answer short. Thank you so much for your help!"
# )

# short_system = SystemMessage(
#     "You are a personal finance assistant. Use simple words. Keep answers short."
# )

# question = HumanMessage("What is an emergency fund?")

# measure("Long system prompt ", model.invoke([long_system, question]))
# measure("Short system prompt", model.invoke([short_system, question]))


# - - - Technique 2: Control the Output Length
# q = "What is inflation?"

# measure("No limit        ", model.invoke(q))
# measure("'In 2 sentences'", model.invoke(q + " Answer in 2 sentences."))


# - - - Technique 3: Ask for Data, Not Paragraphs

# review = "Food was tasty but delivery took almost 90 minutes."

# measure("Open question  ", model.invoke(f"What is the sentiment of this review? {review}"))
# measure("Label only     ", model.invoke(f"Sentiment of this review as Positive, Negative or Mixed. Reply with one word only.\n{review}"))


#  - - - Technique 5: Group Small Tasks Into One Request
# reviews = [
#     "Loved the biryani, will order again!",
#     "Cold food and rude delivery person.",
#     "Okay taste, nothing special.",
#     "Packaging was great and food was hot.",
#     "Waited 1 hour, very disappointed.",
# ]

# # One request per review
# total = 0
# for r in reviews:
#     resp = model.invoke(f"Sentiment as Positive, Negative or Neutral. One word only.\n{r}")
#     total += resp.usage_metadata["total_tokens"]
# print("5 separate requests → total tokens:", total)

# # One request for all reviews
# numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(reviews, start=1))
# resp = model.invoke(
#     "For each review, give the sentiment as Positive, Negative or Neutral. "
#     "Reply as: number. label\n\n" + numbered
# )
# measure("1 grouped request  ", resp)
# print(resp.content)

# Technique 6: Write Instructions in English, Even for Indian-Language Replies

# hindi_instructions = SystemMessage("आप एक सहायक हैं। हमेशा सरल हिंदी में छोटा उत्तर दें।")
# english_instructions = SystemMessage("You are an assistant. Always reply in simple, short Hindi.")

# question = HumanMessage("SIP क्या है?")

# measure("Hindi instructions  ", model.invoke([hindi_instructions, question]))
# measure("English instructions", model.invoke([english_instructions, question]))



# - - - Technique 7: Cache Repeated Questions
# import time
# from langchain_core.globals import set_llm_cache
# from langchain_core.caches import InMemoryCache

# set_llm_cache(InMemoryCache())   # Turn on caching

# start = time.time()
# r1 = model.invoke("What is the full form of KYC?")
# print(f"First call: {time.time() - start:.2f} sec →", r1.content)

# start = time.time()
# r2 = model.invoke("What is the full form of KYC?")
# print(f"Second call: {time.time() - start:.4f} sec →", r2.content)



# Part B: Context Compression 

# Technique 1: Clean the Text With Python First (Free Compression)
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# def clean_text(data):
#     text = data["text"]
#     text = re.sub(r"<[^>]+>", " ", text)       # Remove HTML tags
#     text = re.sub(r"\s+", " ", text)           # Turn many spaces/new lines into one space
#     text = text.replace("Click here to unsubscribe", "")
#     return {"text": text.strip()}

# raw = """<div><p>Dear   Customer,</p>


# <p>Your   order #4521   has been   <b>shipped</b> and will arrive by Friday.</p>
# </div>
# Click here to unsubscribe"""

# prompt = ChatPromptTemplate.from_template("Summarize this message in one line:\n{text}")

# chain = RunnableLambda(clean_text) | prompt | model

# measure("Without cleaning", (prompt | model).invoke({"text": raw}))
# measure("With cleaning   ", chain.invoke({"text": raw}))


# - - - Technique 2: Send Only the Fields You Need

# order = {
#     "order_id": "OD4521", "customer_id": "C9981", "warehouse_code": "WH-PUNE-07",
#     "item": "Wireless earbuds", "status": "Shipped", "expected_delivery": "Friday",
#     "gst_number": "27ABCDE1234F1Z5", "internal_notes": "Packed by team B, batch 112",
#     "payment_gateway_ref": "PG-77812-XYZ", "created_at": "2026-11-10T10:24:00Z",
# }

# question = "Where is my order and when will it come?"

# # ❌ Sending everything
# full = model.invoke(f"Order data: {order}\nCustomer question: {question}\nAnswer in one line.")

# # ✅ Sending only what's needed
# needed = {k: order[k] for k in ["item", "status", "expected_delivery"]}
# small = model.invoke(f"Order data: {needed}\nCustomer question: {question}\nAnswer in one line.")

# measure("All fields   ", full)
# measure("Needed fields", small)


# Technique 3: Compress a Long Document Once, Reuse It Many Times 

policy = """
Leave Policy - ABC Technologies Pvt Ltd (Updated 2026)
All full-time employees are entitled to 18 days of paid leave per calendar year.
Leave is credited at 1.5 days per month. Unused leave up to 10 days can be carried
forward to the next year; anything beyond 10 days will lapse. Sick leave is separate:
employees get 8 days of sick leave per year, and a medical certificate is required for
sick leave longer than 2 consecutive days. Maternity leave is 26 weeks as per Indian law.
Paternity leave is 5 working days, to be taken within 3 months of the child's birth.
Employees on probation (first 6 months) can take a maximum of 6 days of paid leave.
Leave requests must be applied on the HR portal at least 3 days in advance, except
for sick leave. For any queries, contact hr@abctech.example.
"""

# Step 1: Compress once
compress_prompt = ChatPromptTemplate.from_template(
    "Convert this document into short bullet-point facts. Keep every number and rule. "
    "Remove everything else.\n\n{doc}"
)
compressed = (compress_prompt | model | StrOutputParser()).invoke({"doc": policy})
print(compressed)

# Step 2: Reuse for many questions
qa_prompt = ChatPromptTemplate.from_template(
    "Answer using only these facts. Answer in one line.\n\nFacts:\n{facts}\n\nQuestion: {question}"
)
qa_chain = qa_prompt | model

for q in ["How many paid leaves do I get?", "Can I carry forward 12 days?", "How long is paternity leave?"]:
    full_resp = qa_chain.invoke({"facts": policy, "question": q})
    small_resp = qa_chain.invoke({"facts": compressed, "question": q})
    print(f"\n{q}")
    print("  Full doc   →", full_resp.usage_metadata["input_tokens"], "input tokens |", full_resp.content)
    print("  Compressed →", small_resp.usage_metadata["input_tokens"], "input tokens |", small_resp.content)

