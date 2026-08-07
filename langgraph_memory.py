from typing import TypedDict, Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages : Annotated[list, add_messages]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages":[answer]}

memory = MemorySaver()

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)

thread1 = {"configurable": {"thread_id":"1"}}

response = graph.invoke({"messages":[{"role":"user", "content":"Hi, my name is prem"}]}
,config=thread1)
print("AI_Message: ",response["messages"][-1].content)

result = graph.invoke({"messages":[HumanMessage("who am I?")]}, config=thread1)
print("AI_response: ", result["messages"][-1].content)




