import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
    temperature=0,
)

#  --- The Problem: Just Asking for JSON

# import json

job_post = """We are hiring a Junior Python Developer at TechNova, Pune.
Freshers or up to 1 year experience. Skills: Python, SQL, Git.
Salary: ₹3.5 to 4.5 LPA."""

# response = model.invoke(f"Extract job title, company, city and skills from this post as JSON:\n{job_post}")
# print(response.content)

# data = json.loads(response.content)


#  -- Method 1: with_structured_output()
from pydantic import BaseModel, Field

# class JobPost(BaseModel):
#     """Details extracted from a job posting."""
#     job_title: str = Field(description="The job role, e.g. 'Data Analyst'")
#     company: str = Field(description="Name of the hiring company")
#     city: str = Field(description="City where the job is located")
#     skills: list[str] = Field(description="List of required skills")

# Attach it to the model 
# structured_model = model.with_structured_output(JobPost)

# result = structured_model.invoke(job_post)

# print(result)
# print(result.job_title)
# print(result.skills)

# print(result.model_dump())


#  --- Handling Missing Information and Fixed Choices
from typing import Literal

class JobPost(BaseModel):
    """Details extracted from a job posting."""
    job_title: str = Field(description="The job role")
    company: str = Field(description="Name of the hiring company")
    city: str = Field(description="City where the job is located")
    skills: list[str] = Field(description="List of required skills")
    min_salary_lpa: float | None = Field(default=None, description="Minimum salary in LPA, if mentioned")
    max_salary_lpa: float | None = Field(default=None, description="Maximum salary in LPA, if mentioned")
    level: Literal["Fresher", "Experienced"] = Field(description="Fresher if 0-1 year experience, else Experienced")


# structured_model = model.with_structured_output(JobPost)
# result = structured_model.invoke(job_post)
# print(result.model_dump())


#  --- Using Structured Output in a Chain
# extract_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Extract job details from the posting. Only use information present in the text."),
#     ("human", "{post}"),
# ])

# extract_chain = extract_prompt | model.with_structured_output(JobPost)

# posts = [
#     "Hiring Data Analyst at FinEdge, Mumbai. 2+ years experience. Skills: Excel, Power BI, SQL.",
#     "Frontend intern needed at PixelCraft, Bengaluru. Freshers welcome. React, HTML, CSS. Stipend ₹15k/month.",
# ]

# results = extract_chain.batch([{"post": p} for p in posts])

# for r in results:
#     print(r.job_title, "|", r.city, "|", r.level, "|", r.skills)




#  --- Method 2: Output Parsers

# JsonOutputParser
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=JobPost)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract job details from the posting.\n{format_instructions}"),
    ("human", "{post}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

# result = chain.invoke({"post": job_post})
# print(result)
# print(type(result))


# PydanticOutputParser 

from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=JobPost)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract job details from the posting.\n{format_instructions}"),
    ("human", "{post}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

result = chain.invoke({"post": job_post})
print(result.job_title, result.skills)



