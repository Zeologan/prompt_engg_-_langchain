import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0,   # 0 is best for maths and logic
)


# -- Chain-of-Thought (CoT) Prompting

# question = (
#     "A shirt costs ₹800. The shop gives 25% off. "
#     "Then on the reduced price, there is an extra 10% off. "
#     "What is the final price? Let's think step by step."
# )

# response = model.invoke(question)
# print(response.content)


# question = (
#     "A shirt costs ₹800. The shop gives 25% off. "
#     "Then on the reduced price, there is an extra 10% off. "
#     "What is the final price? Let's think step by step."
# )

# response = model.invoke(question)
# print(response.content)

# system = SystemMessage(
#     "Solve the problem step by step. "
#     "At the end, write the final answer on a new line in this format:\n"
#     "Final Answer: <answer>"
# )

# response = model.invoke([system, HumanMessage(question)])

# # Pick out only the final answer
# full_text = response.content
# final = full_text.split("Final Answer:")[-1].strip()

# print("\nOnly the answer:", final)


# --- few shot with cot 
# response = model.invoke([
#     SystemMessage("Solve the problem step by step, then give the final answer."),

#     HumanMessage("A train ticket costs ₹500. Seniors get 40% off. What does a senior pay?"),
#     AIMessage("40% of ₹500 = ₹200. Senior pays ₹500 - ₹200 = ₹300.\nFinal Answer: ₹300"),

#     HumanMessage(
#         "A mobile costs ₹20,000. There is a ₹2,000 cashback, then 18% GST is added "
#         "on the price after cashback. What is the final amount?"
#     ),
# ])
# print(response.content)




# from langchain_core.prompts import ChatPromptTemplate

# # Step 1: Draft
# draft_template = ChatPromptTemplate.from_messages([
#     ("system", "You write short professional emails."),
#     ("human", "Write an email to an HR manager asking for a {role} internship. I am a {background}."),
# ])

# # Step 2: Critique (with a clear checklist)
# critique_template = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a strict HR reviewer. Review the email using this checklist:\n"
#      "1. Is the subject line clear?\n"
#      "2. Is it under 120 words?\n"
#      "3. Does it mention specific skills?\n"
#      "4. Does it have a clear request at the end?\n"
#      "5. Is the tone polite but confident?\n"
#      "List only the problems you find. Be specific."),
#     ("human", "Email:\n\"\"\"{draft}\"\"\""),
# ])

# # Step 3: Revise
# revise_template = ChatPromptTemplate.from_messages([
#     ("system", "You improve emails based on reviewer feedback. Return only the improved email."),
#     ("human", "Original email:\n\"\"\"{draft}\"\"\"\n\nReviewer feedback:\n{feedback}"),
# ])

# # Run all three steps
# inputs = {"role": "Data Analyst", "background": "final-year BSc IT student who knows Python and Excel"}

# draft = model.invoke(draft_template.invoke(inputs)).content
# feedback = model.invoke(critique_template.invoke({"draft": draft})).content
# final_email = model.invoke(revise_template.invoke({"draft": draft, "feedback": feedback})).content

# print("DRAFT:\n", draft)
# print("\nFEEDBACK:\n", feedback)
# print("\nFINAL:\n", final_email)


response = model.invoke([
    SystemMessage(
        "Solve the problem step by step. "
        "Then check your answer by working backwards. "
        "If you find a mistake, correct it. "
        "End with: Final Answer: <answer>"
    ),
    HumanMessage("If 5 pens cost ₹75, how much do 12 pens cost?"),
])
print(response.content)
