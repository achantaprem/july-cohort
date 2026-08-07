from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    messages : Annotated[list, add_messages]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def chat_bot(state : State):
    answer = model.invoke(state["messages"])
    return{"messages": [answer]}

builder = StateGraph(State)
builder.add_node("chatbot", chat_bot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

result = graph.invoke({"messages":[{"role":"user", "content":"who am I"}]})
print(result["messages"][-1].content)



