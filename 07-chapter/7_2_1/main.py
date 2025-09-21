import uuid
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph

load_dotenv()

workflow = StateGraph(state_schema=MessagesState)
llm = ChatDeepSeek(model="deepseek-chat")


# 定义调用模型的函数
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# 初始化记忆
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 线程ID是一个唯一标识符，用于标识这次特定的对话
# 这是的单个应用能够管理多个用户之间的对话
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}
message1 = HumanMessage(content="我是李四")
for event in app.stream({"messages": message1}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# 询问之前提到的名字
message2 = HumanMessage(content="我的名字是什么？")
for event in app.stream({"messages": message2}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
