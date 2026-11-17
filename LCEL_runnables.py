# LCEL and Runnables 

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0,
)


# Your First Chain 
translator = ChatPromptTemplate.from_messages([
    ("system", "Translate the user's text into {language}. Reply with only the translation."),
    ("human", "{text}"),
])

# prompt_value = translator.invoke({"language": "Hindi", "text": "Where is the bus stop?"})
# response = model.invoke(prompt_value)
# print(response.content)


#  --- After: One chain with LCEL
# chain = translator | model | StrOutputParser()

# result = chain.invoke({"language": "Hindi", "text": "Where is the bus stop?"})
# print(result)


# Stream the answer piece by piece
# for chunk in chain.stream({"language": "Marathi", "text": "Good night, see you tomorrow."}):
#     print(chunk, end="", flush=True)

# Run many inputs together
# results = chain.batch([
#     {"language": "Tamil", "text": "Thank you"},
#     {"language": "Bengali", "text": "Thank you"},
#     {"language": "Gujarati", "text": "Thank you"},
# ])
# print(results)



#  ------- RunnableLambda: Adding Your Own Python Functions
# from langchain_core.runnables import RunnableLambda

# def word_count(text):
#     return f"{text}\n\n(Word count: {len(text.split())})"

# summary_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Summarize the text in 2 simple sentences."),
#     ("human", "{text}"),
# ])

# chain = summary_prompt | model | StrOutputParser() | RunnableLambda(word_count)

# result = chain.invoke({"text": "Mumbai local trains carry over 7 million passengers daily. "
#                                "They are called the lifeline of the city because most people "
#                                "depend on them to reach work, college and markets."})
# print(result)


#  --- Connecting One Chain to Another

# idea_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Write a one-line story idea."),
#     ("human", "Topic: {topic}"),
# ])

# idea_chain = idea_prompt | model | StrOutputParser()

# full_chain = (
#     idea_chain
#     | (lambda idea: {"text": idea, "language": "Hindi"})   # Convert text into a dictionary
#     | translator
#     | model
#     | StrOutputParser()
# )

# print(full_chain.invoke({"topic": "a lost cricket ball"}))



# --- RunnableParallel: Running Steps at the Same Time 

# from langchain_core.runnables import RunnableParallel

# tagline_chain = ChatPromptTemplate.from_template(
#     "Write a catchy one-line tagline for: {product}"
# ) | model | StrOutputParser()

# description_chain = ChatPromptTemplate.from_template(
#     "Write a 2-line product description for: {product}"
# ) | model | StrOutputParser()

# hashtag_chain = ChatPromptTemplate.from_template(
#     "Give 5 Instagram hashtags for: {product}. Only hashtags, separated by spaces."
# ) | model | StrOutputParser()

# marketing = RunnableParallel(
#     tagline=tagline_chain,
#     description=description_chain,
#     hashtags=hashtag_chain,
# )

# result = marketing.invoke({"product": "a steel water bottle that keeps water cold for 24 hours"})

# print(result["tagline"])
# print(result["description"])
# print(result["hashtags"])




#  ----- RunnablePassthrough: Keeping the Original Input 

from langchain_core.runnables import RunnablePassthrough

draft_template = ChatPromptTemplate.from_messages([
    ("system", "You write short professional emails."),
    ("human", "Write an email to an HR manager asking for a {role} internship. I am a {background}."),
])

critique_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict HR reviewer. Check: 1) clear subject line, 2) under 120 words, "
     "3) specific skills mentioned, 4) clear request at the end. List only the problems."),
    ("human", "Email:\n\"\"\"{draft}\"\"\""),
])

revise_template = ChatPromptTemplate.from_messages([
    ("system", "Improve the email using the feedback. Return only the improved email."),
    ("human", "Email:\n\"\"\"{draft}\"\"\"\n\nFeedback:\n{feedback}"),
])

parser = StrOutputParser()

self_critique_chain = (
    RunnablePassthrough.assign(draft=draft_template | model | parser)
    | RunnablePassthrough.assign(feedback=critique_template | model | parser)
    | revise_template | model | parser
)

final_email = self_critique_chain.invoke({
    "role": "Data Analyst",
    "background": "final-year BSc IT student who knows Python and Excel",
})
print(final_email)
