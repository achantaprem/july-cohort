from typing import TypedDict, Annotated
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages : Annotated[list, add_messages]

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def chatbot(state:State):
    persona = SystemMessage(content="You are a senior software architect at IT Solutions. Always answer with a highly technical, professional tone, focus on system design, and warmly welcome the user to IT Solutions on the first interaction.")
    message_with_persona = [persona] + state["messages"]
    answer = model.invoke(message_with_persona)
    return {"messages" : [answer]}

memory = MemorySaver()
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)

thread_1 = {"configurable" : {"thread_id" : "1"}}

print("Responses with persona and user messages\n")

response1 = graph.invoke({"messages":[HumanMessage("what is the best way to structure AI agents?")]}, thread_1)

print(response1["messages"][-1].content)
print("\n")

response2 = graph.invoke({"messages":[HumanMessage("can you summarize in one sentence")]}, thread_1)
print(response2["messages"][-1].content)

