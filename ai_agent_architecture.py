import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_LARGE"),
    model_provider="groq",
    temperature=0,
)

@tool
def calculate_emi(principal: float, annual_rate: float, years: int) -> str:
    """Calculate the monthly EMI for a loan.

    Args:
        principal: Loan amount in rupees, e.g. 500000
        annual_rate: Yearly interest rate in percent, e.g. 9.5
        years: Loan duration in years
    """
    r = annual_rate / 12 / 100
    n = years * 12
    emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return f"Monthly EMI is ₹{emi:,.0f} for {n} months."

@tool
def calculate_gst(amount: float, gst_rate: float) -> str:
    """Calculate GST and the final price for a product.

    Args:
        amount: Price before GST in rupees
        gst_rate: GST rate in percent, e.g. 18
    """
    gst = amount * gst_rate / 100
    return f"GST: ₹{gst:,.2f}, Final price: ₹{amount + gst:,.2f}"

tools = [calculate_emi, calculate_gst]



# - - - Step 1: Building the Agent Loop Yourself

# def simple_agent(question, max_steps=5):
#     tools_by_name = {t.name: t for t in tools}
#     model_with_tools = model.bind_tools(tools)

#     state = [HumanMessage(question)]           # State: the agent's notebook

#     for step in range(max_steps):
#         ai_message = model_with_tools.invoke(state)   # LLM: decide next step
#         state.append(ai_message)

#         if not ai_message.tool_calls:                  # No tool needed → done
#             return ai_message.content

#         for tool_call in ai_message.tool_calls:        # Tools: do the work
#             print(f"Step {step + 1} → {tool_call['name']}({tool_call['args']})")
#             result = tools_by_name[tool_call["name"]].invoke(tool_call)
#             state.append(result)                       # Save result in state

#     return "Stopped: too many steps."

# print(simple_agent(
#     "I want to buy a laptop for ₹60,000 plus 18% GST, on EMI at 12% interest for 1 year. "
#     "What's my monthly EMI?"
# ))



# Step 2: Using LangChain's create_agent 
from langchain.agents import create_agent

# agent = create_agent(
#     model=model,
#     tools=tools,
#     system_prompt="You are a helpful finance assistant for Indian users. "
#                   "Use tools for all calculations. Never calculate in your head.",
# )

# result = agent.invoke({
#     "messages": [{"role": "user", "content":
#         "I want to buy a laptop for ₹60,000 plus 18% GST, on EMI at 12% for 1 year. "
#         "What's my monthly EMI?"}]
# })

# for msg in result["messages"]:
#     msg.pretty_print()

# print(result["messages"][-1].content)


# - - - Watching the agent live 
# for step in agent.stream(
#     {"messages": [{"role": "user", "content": "GST on ₹25,000 at 12%, and EMI on that total at 10% for 2 years?"}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()



# Step 4: Continuing a Conversation 

from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful finance assistant for Indian users. "
                  "Use tools for all calculations. Never calculate in your head.",
    checkpointer=InMemorySaver(),
)

# config = {"configurable": {"thread_id": "priya"}}

# first = agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the final price of a ₹40,000 phone with 18% GST?"}]},
#     config,
# )
# print(first["messages"][-1].content)

# follow_up = agent.invoke(
#     {"messages": [{"role": "user", "content": "What would the EMI be on that at 14% for 1 year?"}]},
#     config,
# )
# print(follow_up["messages"][-1].content)


# Limit the number of steps 

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "EMI on ₹2,00,000 at 9% for 3 years?"}]},
#     config = {"configurable": {"thread_id": "priya"}, "recursion_limit": 10},   # Stop after 10 internal steps
# )

# print(result["messages"][-1].content)

from langchain_core.callbacks import get_usage_metadata_callback

with get_usage_metadata_callback() as tracker:
    agent.invoke({"messages": [{"role": "user", "content":
        "Laptop ₹60,000 + 18% GST, EMI at 12% for 1 year?"}]},
        config = {"configurable": {"thread_id": "priya"}, "recursion_limit": 10},
        )

print(tracker.usage_metadata)


