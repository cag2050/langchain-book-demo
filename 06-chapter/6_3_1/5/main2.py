from typing import TypedDict

from langgraph.constants import START
from langgraph.graph import StateGraph


# 父图和子图使用不同的状态模式
class ParentState(TypedDict):
    parent_input: str
    parent_output: str


class SubState(TypedDict):
    sub_input: str  # 完全不同的键
    sub_output: str


def process_in_sub(state: SubState):
    return {"sub_output": f"子图处理：{state['sub_input']}"}


# 调用子图的函数
def call_subgraph(state: ParentState):
    # 转换状态到子图格式
    sub_result = sub_graph.invoke({
        "sub_input": state["parent_input"],
    })
    # 转换结果到父图格式
    return {"parent_output": sub_result["sub_output"]}


# 定义子图
sub_builder = StateGraph(SubState)
sub_builder.add_node("process", process_in_sub)
sub_builder.add_edge(START, "process")
sub_graph = sub_builder.compile()

# 在父图中使用子图
parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub_process", call_subgraph)  # 调用子图
parent_builder.add_edge(START, "sub_process")

parent_graph = parent_builder.compile()
result = parent_graph.invoke(
    {"parent_input": 123}
)
print(result)
