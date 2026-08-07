from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, trim_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage("You are a helpful AI assistant."),
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

model = ChatOpenAI(model=MODEL, temperature=0.0)

trimmer = trim_messages(
    max_tokens = 65,
    strategy = "last",
    token_counter = model,
    include_system = True,
    start_on = "human",
)

trimmed_history = trimmer.invoke(messages)

print(f"Original messages: {len(messages)}")
print(f"Trimmed messages: {len(trimmed_history)}\n")

for msg in trimmed_history:
  print(f"{msg.type.upper()} : {msg.content}")