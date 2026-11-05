import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # Reads GROQ_API_KEY and MODEL_ID from the .env file

# Create the model (LangChain handles the connection to Groq)
model = init_chat_model(
    model=os.getenv("MODEL_ID"),
    model_provider="groq",
)

# Ask a question
response = model.invoke("Explain what an API is in 2 simple lines.")

# Print only the answer text
print(response.content)

