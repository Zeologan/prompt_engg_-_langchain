import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0,
)


#  - - - Tracking Tokens for One Call
# response = model.invoke("Explain what GST is in 2 lines.")

# print(response.content)
# print(response.usage_metadata)



# - - - Experiment 1: English vs Hindi 
# english = "Please tell me the train timings from Pune to Mumbai."
# hindi = "कृपया मुझे पुणे से मुंबई की ट्रेन का समय बताइए।"

# for label, text in [("English", english), ("Hindi", hindi)]:
#     response = model.invoke(text, max_tokens=1)   # max_tokens=1 so we mainly measure input
#     print(label, "→ input tokens:", response.usage_metadata["input_tokens"])



#  - - - Experiment 2: How Few-shot Examples Add Up
# system = SystemMessage("Classify complaints as Urgent, Normal or Low. Reply with only the label.")
# question = HumanMessage("My parcel is 3 days late.")

# examples = [
#     HumanMessage("My refund has not come even after 10 days."), AIMessage("Urgent"),
#     HumanMessage("My parcel is 2 days late."), AIMessage("Normal"),
#     HumanMessage("Please add a dark mode to your app."), AIMessage("Low"),
# ]

# zero_shot = model.invoke([system, question])
# few_shot = model.invoke([system] + examples + [question])

# print("Zero-shot input tokens:", zero_shot.usage_metadata["input_tokens"])
# print("Few-shot input tokens: ", few_shot.usage_metadata["input_tokens"])



# - - - Experiment 3: How Chat History Grows 
history = [SystemMessage("You are a helpful assistant. Keep answers to 2 lines.")]

# questions = [
#     "Suggest a weekend trip from Pune.",
#     "How do I reach there?",
#     "What should I pack?",
#     "Any good local food?",
#     "What is the best time to go?",
# ]

# for i, q in enumerate(questions, start=1):
#     history.append(HumanMessage(q))
#     response = model.invoke(history)
#     history.append(response)
#     print(f"Turn {i} → input tokens: {response.usage_metadata['input_tokens']}")



#  - - - Estimating Tokens Before Sending

# from langchain_core.messages.utils import count_tokens_approximately

# estimate = count_tokens_approximately(history)
# print("Estimated tokens in history:", estimate)

# if estimate > 1000:
#     print("History is too long, trim it before sending.")



# - - - Tracking Tokens Across a Whole Chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel
# from langchain_core.callbacks import get_usage_metadata_callback

# parser = StrOutputParser()

# marketing = RunnableParallel(
#     tagline=ChatPromptTemplate.from_template("One-line tagline for: {product}") | model | parser,
#     description=ChatPromptTemplate.from_template("2-line description for: {product}") | model | parser,
#     hashtags=ChatPromptTemplate.from_template("5 hashtags for: {product}") | model | parser,
# )

# with get_usage_metadata_callback() as tracker:
#     result = marketing.invoke({"product": "a solar-powered phone charger"})

# print(tracker.usage_metadata)



#  - - - A Simple Token Tracker for Your App
class TokenTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, response):
        usage = response.usage_metadata
        self.input_tokens += usage["input_tokens"]
        self.output_tokens += usage["output_tokens"]
        self.calls += 1

    def report(self):
        total = self.input_tokens + self.output_tokens
        print(f"Calls: {self.calls} | Input: {self.input_tokens} | Output: {self.output_tokens} | Total: {total}")

tracker = TokenTracker()

for q in ["What is SIP?", "What is a mutual fund?", "What is an index fund?"]:
    response = model.invoke(q, max_tokens=60)
    tracker.add(response)

tracker.report()
