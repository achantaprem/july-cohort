import asyncio
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient 

load_dotenv()

# 1. DEFINE STATE
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def main():
    # 2. INITIALIZE MCP CLIENT
    client = MultiServerMCPClient({
        # "math": {
        #     "transport": "stdio",
        #     "command": "python",
        #     "args": ["class_assignment.py"], 
        # },
        "docs-langchain": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp"
        }
    })
    
    # 3. GET TOOLS
    mcp_tools = await client.get_tools()

    # Manually subset the tools you actually want this specific graph to use
    duck_tool = mcp_tools[0]
    calculator_tool = mcp_tools[1]
    subset_tools = [duck_tool, calculator_tool]
    
    # 4. SETUP LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = llm.bind_tools(subset_tools)

    # 5. DEFINE GRAPH NODE
    # (Moved inside main() so it has access to the locally scoped `chain` and `subset_tools`)
    async def assistant(state: State) -> dict:
        """The 'brain' node: one model call over the running conversation."""
        # FIXED: Changed .invoke to .ainvoke for async execution
        response = await chain.ainvoke(state["messages"])
        return {"messages": [response]}

    # 6. BUILD GRAPH
    builder = StateGraph(State)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(subset_tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)  # tools? : END
    builder.add_edge("tools", "assistant")                       # loop back

    graph = builder.compile(checkpointer=MemorySaver())
    thread = {"configurable": {"thread_id": "demo-3"}}

    # 7. EXECUTE GRAPH
    print("--- turn 1: needs the calculator ---")
    
    # FIXED: Changed .stream to .astream because we are in an async function
    async for chunk in graph.astream(
        {"messages": [HumanMessage("What is 1234 * 5678?")]}, thread
    ):
        for node, update in chunk.items():
            print(f"[{node}] {update['messages'][-1].content[:120]}")

    print("\n--- turn 2: relies on MEMORY of turn 1 ---")
    
    # FIXED: Changed .invoke to .ainvoke
    result = await graph.ainvoke(
        {"messages": [HumanMessage("Now divide that result by 2.")]}, thread
    )
    print("bot >", result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())