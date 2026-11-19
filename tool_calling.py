import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_LARGE"),
    model_provider="groq",
    temperature=0,
)


# Step 1: Create a Tool 

from langchain.tools import tool

@tool
def calculate_emi(principal: float, annual_rate: float, years: int) -> str:
    """Calculate the monthly EMI for a loan.

    Args:
        principal: Loan amount in rupees, e.g. 500000
        annual_rate: Yearly interest rate in percent, e.g. 9.5
        years: Loan duration in years
    """
    monthly_rate = annual_rate / 12 / 100
    months = years * 12
    emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
    return f"Monthly EMI is ₹{emi:,.0f} for {months} months."

# print(calculate_emi.name)
# print(calculate_emi.description)
# print(calculate_emi.args)



model_with_tools = model.bind_tools([calculate_emi])

# response = model_with_tools.invoke("What will be my EMI for a ₹5 lakh loan at 9.5% for 5 years?")

# print("Content:", response.content)
# print("Tool calls:", response.tool_calls)


# response = model_with_tools.invoke("What does EMI stand for?")

# print("Content:", response.content)
# print("Tool calls:", response.tool_calls)


# - - - Step 3 and 4: Run the Tool and Send the Result Back


# question = HumanMessage("What will be my EMI for a ₹5 lakh loan at 9.5% for 5 years?")

# # Step 3: AI asks for the tool
# ai_message = model_with_tools.invoke([question])
# tool_call = ai_message.tool_calls[0]

# # Step 4: Our code runs the tool
# tool_message = calculate_emi.invoke(tool_call)
# print(tool_message)

# # Step 5: Send everything back to the AI for the final answer
# final = model_with_tools.invoke([question, ai_message, tool_message])
# print(final.content)


# - - -- Working With More Than One Tool


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
tools_by_name = {t.name: t for t in tools}       # Look up a tool by its name
model_with_tools = model.bind_tools(tools)


def ask(question_text):
    messages = [HumanMessage(question_text)]
    ai_message = model_with_tools.invoke(messages)
    messages.append(ai_message)

    # Run every tool the AI asked for
    for tool_call in ai_message.tool_calls:
        selected_tool = tools_by_name[tool_call["name"]]
        print(f"→ AI chose: {tool_call['name']} with {tool_call['args']}")
        messages.append(selected_tool.invoke(tool_call))

    # If no tool was needed, the first reply is already the answer
    if not ai_message.tool_calls:
        return ai_message.content

    return model_with_tools.invoke(messages).content


print(ask("A laptop costs ₹60,000 before GST. What's the final price at 18% GST? In short, please."))
print()
print(ask("EMI for ₹3,00,000 at 10% for 2 years? In short, please."))
print()
print(ask("Tell me one tip for saving money. In short, please."))
