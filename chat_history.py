import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0.3,
)


# --- A Simple Chatbot With Manual History

# history = [SystemMessage("You are a friendly travel guide for India. Keep answers short.")]

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break

#     history.append(HumanMessage(user_input))   # Save the user's message
#     response = model.invoke(history)            # Send the FULL history
#     history.append(response)                    # Save the AI's reply

#     print("Bot:", response.content)


# --- Adding History to a Prompt Template 

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a friendly travel guide for India. Keep answers short."),
#     MessagesPlaceholder("history"),     # Past messages go here
#     ("human", "{question}"),            # New question goes here
# ])

# chain = prompt | model

# past_messages = [
#     HumanMessage("Suggest one beach in Goa."),
#     AIMessage("Try Baga Beach in South Goa."),
# ]

# response = chain.invoke({"history": past_messages, "question": "Best time to visit there?"})
# print(response.content)


# --- Automatic Memory for Many Users 
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# chatbot = create_agent(
#     model=model,
#     tools=[],                         # No tools, just a chatbot
#     system_prompt="You are a friendly travel guide for India. Keep answers short.",
#     checkpointer=InMemorySaver(),     # The "filing cabinet" that saves every chat
# )

# def chat(question, user_id):
#     config = {"configurable": {"thread_id": user_id}}   # Which user's file to open
#     result = chatbot.invoke({"messages": [{"role": "user", "content": question}]}, config)
#     return result["messages"][-1].content

# User 1: Rahul
# print(chat("Suggest a hill station in Himachal.", "rahul"))
# print(chat("What food is famous there?", "rahul"))

# User 2: Sneha (separate conversation)
# print(chat("What food is famous there?", "sneha"))

# saved = chatbot.get_state({"configurable": {"thread_id": "rahul"}})

# for msg in saved.values["messages"]:
#     print(type(msg).__name__, ":", msg.content)



# --- Managing the Size of Memory 

# from langchain_core.messages import trim_messages
# long_chat = [
#     SystemMessage("You are a helpful assistant."),
#     HumanMessage("My name is Arjun."),
#     AIMessage("Nice to meet you, Arjun!"),
#     HumanMessage("I live in Nashik."),
#     AIMessage("Nashik is famous for grapes and vineyards."),
#     HumanMessage("I like cricket."),
#     AIMessage("Great! Who is your favourite player?"),
#     HumanMessage("Virat Kohli."),
# ]

# trimmed = trim_messages(
#     long_chat,
#     strategy="last",       # Keep the most recent messages
#     token_counter=len,     # Count each message as 1, so max_tokens means "max messages"
#     max_tokens=5,          # Keep at most 5 messages
#     include_system=True,   # Always keep the system message
#     start_on="human",      # History should start with a user message
# )

# for m in trimmed:
#     print(type(m).__name__, ":", m.content)



#  --- Adding trimming to your chatbot
# from langchain.agents import AgentState
# from langchain.agents.middleware import before_model
# from langchain.messages import RemoveMessage
# from langgraph.graph.message import REMOVE_ALL_MESSAGES

# @before_model
# def keep_recent_messages(state: AgentState, runtime) -> dict | None:
#     messages = state["messages"]
#     if len(messages) <= 10:
#         return None                    # Short chat, nothing to trim

#     recent = trim_messages(
#         messages,
#         strategy="last",
#         token_counter=len,
#         max_tokens=10,                 # Keep only the last 10 messages
#         start_on="human",
#     )
#     # Replace the saved history with only the recent messages
#     return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent]}

# chatbot = create_agent(
#     model=model,
#     tools=[],
#     system_prompt="You are a friendly travel guide for India. Keep answers short.",
#     middleware=[keep_recent_messages],
#     checkpointer=InMemorySaver(),
# )


# Summarizing old messages (built-in) 
from langchain.agents.middleware import SummarizationMiddleware

chatbot = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a friendly travel guide for India. Keep answers short.",
    middleware=[
        SummarizationMiddleware(
            model=model,               # Model used to write the summary
            trigger=("tokens", 500),   # Summarize when history crosses ~500 tokens
            keep=("messages", 6),      # Always keep the last 6 messages as they are
        )
    ],
    checkpointer=InMemorySaver(),
)

print(chat("Hi, my name is Arjun and I live in Nashik.", "arjun"))
print(chat("Suggest a weekend trip from my city.", "arjun"))
# ... after many more messages ...
print(chat("What's my name and where do I live?", "arjun"))
