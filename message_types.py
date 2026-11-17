import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0,      # 0 = focused and consistent, higher = more creative
    max_tokens=200,     # Limits how long the answer can be
    max_retries=2,      # Automatically tries again if the request fails
)

# messages = [
#     SystemMessage("You are a friendly maths tutor. Explain in simple words with one example."),
#     HumanMessage("What is a percentage?"),
# ]


# response = model.invoke(messages)
# print(response.content)


### --- The Model Does NOT Remember: Conversations
# messages = [
#     SystemMessage("You are a helpful travel guide."),
#     HumanMessage("Suggest one hill station near Mumbai."),
#     AIMessage("Lonavala is a great choice. It's about 2 hours from Mumbai."),
#     HumanMessage("What is the best time to visit there?"),  # "there" = Lonavala
# ]

# response = model.invoke(messages)
# print(response.content)          # The answer text
# print(response.usage_metadata)   # Tokens used: input, output, total


### --- The Response Object
response = model.invoke("Say hello in Marathi.")

print(response.content)          # The answer text
print(response.usage_metadata)   # Tokens used: input, output, total




# Stream: print the answer word by word
# for chunk in model.stream("Write a 4-line poem about Mumbai rain."):
#     print(chunk.content, end="", flush=True)

# Batch: ask multiple questions in one go

# Batch: ask multiple questions in one go
# questions = ["Capital of Kerala?", "Capital of Punjab?", "Capital of Assam?"]

# answers = model.batch(questions)
# for ans in answers:
#     print(ans.content)


