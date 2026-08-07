from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv

success = load_dotenv(find_dotenv())
print(f"did .env load successfully: {success}")


MODEL = "gpt-4o-mini"

# Naive Fix: Use a placeholder slot to dump the entire message history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{messages}"),  # History gets injected here
])

chain = prompt | ChatOpenAI(model=MODEL, temperature=0.0)

# Manually passing the entire history with every call
response = chain.invoke({
    "messages": [
        ("human", "Translate 'I love programming' to French."),
        ("ai", "J'adore programmer."),
        ("human", "What did you just say?"),
    ]
})

print(response.content) # Output: I said 'J'adore programmer'...
