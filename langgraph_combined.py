from typing import TypedDict, Annotated
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
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

graph.update_state(thread1, {"messages":[HumanMessage("I Love building AI agents.")]})

messages1=[
    SystemMessage("You are a helpful AI assistant"),
    HumanMessage("Hi! I am starting a new project today."),
    AIMessage("That's fantastic, what kind of project are you building."),
    HumanMessage("I am buliding an AI agent using LangGraph."),
    AIMessage("LangGraph is fantastic for stateful agents. Do you need help with the stepup?"),
    HumanMessage("Yes, how do I define a basic StateGraph?"),
    AIMessage("You define a state TypedDict, add nodes and connect with edges."),
    HumanMessage("Got it, what if conversation gets too long"),
    AIMessage("If it gets too long, it will reach token limit and it will crash."),
    HumanMessage("How do I prevent that token limit crash?"),
]

graph.update_state(thread1, {"messages" : messages1})

current_state = graph.get_state(thread1)
print("Current Messages: ", len(current_state.values["messages"]))
for m in current_state.values["messages"]:
    print(f"{m.type}, {m.content}")

trimmer = trim_messages(
    max_tokens = 80,
    strategy = "last",
    token_counter = model,
    include_system = True,
    start_on = "human"
)

history = current_state.values["messages"]
trimmed_history = trimmer.invoke(history)

print(f"Original message count: {len(current_state.values["messages"])}")
print(f"Trimmed message count: {len(trimmed_history)}\n")

for msg in trimmed_history:
    print(f"{msg.type.upper()} : {msg.content}")



