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

# ---Zero Shot 
# response = model.invoke(
#     "Classify this complaint as Urgent, Normal or Low priority.\n"
#     "Complaint: I was charged twice for my order."
# )
# print(response.content)


# ----One Shot 
# response = model.invoke([
#     SystemMessage("Classify customer complaints as Urgent, Normal or Low. Reply with only the label."),
#     HumanMessage("My refund has not come even after 10 days."),
#     AIMessage("Urgent"),
#     HumanMessage("I was charged twice for my order."),
# ])
# print(response.content)

# ----Few-shot 
# response = model.invoke([
#     SystemMessage("Classify customer complaints as Urgent, Normal or Low. Reply with only the label."),

#     HumanMessage("My refund has not come even after 10 days."),
#     AIMessage("Urgent"),

#     HumanMessage("My parcel is 2 days late."),
#     AIMessage("Normal"),

#     HumanMessage("Please add a dark mode to your app."),
#     AIMessage("Low"),

#     HumanMessage("Money was deducted but the order shows failed."),
#     AIMessage("Urgent"),

#     # Actual question
#     HumanMessage("The delivery boy said he will come tomorrow instead of today."),
# ])
# print(response.content)   # Normal


#---- ChatPromptTemplate 

# from langchain_core.prompts import ChatPromptTemplate

# translator = ChatPromptTemplate.from_messages([
#     ("system", "You are a translator. Translate the user's text into {language}. Reply with only the translation."),
#     ("human", "{text}"),
# ])

# # Step 1: Fill the template with values
# prompt_value = translator.invoke({
#     "language": "Hindi",
#     "text": "Where is the nearest railway station?",
# })

# # Step 2: Send the filled prompt to the model
# response = model.invoke(prompt_value)
# print(response.content)



# ------- FewShotChatMessagePromptTemplate: Few-shot + Templates Together

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Step 1: Examples as a simple list of dictionaries
examples = [
    {"complaint": "My refund has not come even after 10 days.", "priority": "Urgent"},
    {"complaint": "My parcel is 2 days late.", "priority": "Normal"},
    {"complaint": "Please add a dark mode to your app.", "priority": "Low"},
    {"complaint": "Money was deducted but the order shows failed.", "priority": "Urgent"},
]

# Step 2: How ONE example should look as messages
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{complaint}"),
    ("ai", "{priority}"),
])

# Step 3: Combine examples with the example format
few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# Step 4: Final template = system rule + examples + new question
classifier = ChatPromptTemplate.from_messages([
    ("system", "Classify customer complaints as Urgent, Normal or Low. Reply with only the label."),
    few_shot,
    ("human", "{complaint}"),
])

# Use it
prompt_value = classifier.invoke({"complaint": "I got the wrong size shoes."})
print(model.invoke(prompt_value).content)
