from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, AIMessage, RemoveMessage
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph


# 短期记忆
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def my_node_1(state: State):
    # 添加一条AIMessage到messages列表
    return {"messages": [AIMessage(content="你好")]}


def my_node_2(state: State):
    # 从messages列表中删除除最后两条以外的所有消息（只保留最后两条消息）
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"messages": delete_messages}


builder = StateGraph(State)
builder.add_node(my_node_1)
builder.add_node(my_node_2)
builder.add_edge(START, "my_node_1")
builder.add_edge("my_node_1", "my_node_2")
builder.add_edge("my_node_2", END)

graph = builder.compile()
result = graph.invoke({
    "messages": [AIMessage(content=str(i)) for i in range(5)]
})
print(result)
