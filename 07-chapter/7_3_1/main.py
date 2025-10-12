import uuid
from typing import List
from dotenv import load_dotenv

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

recall_vector_store = InMemoryVectorStore(DashScopeEmbeddings())


def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("需要提供用户ID来保存记忆")
    return user_id


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """保存记忆到向量存储以供后续语义检索"""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory,
        id=str(uuid.uuid4()),
        metadata={"user_id": user_id},
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """搜索相关记忆"""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata["user_id"] == user_id

    documents = recall_vector_store.similarity_search(query, k=3, filter=_filter_function)
    return [document.page_content for document in documents]


tools = [
    save_recall_memory,
    search_recall_memories,
    TavilySearch(max_results=1),
]
llm = ChatDeepSeek(model="deepseek-chat")
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个具有长期记忆能力的助手。你需要依赖外部记忆来在对话之间存储信息，请使用可用的记忆工具来存储和检索\n\n"
        "记忆使用指南:\n"
        "1. 积极使用记忆工具来建立对用户的全面理解\n"
        "2. 根据储存的记忆做出推论\n"
        "3. 定期回顾过往互动以识别用户的偏好\n"
        "4. 根据每个新信息更新对用户的认知模型\n"
        "5. 交叉对照新旧信息以保持一致性\n"
        "6. 利用记忆认识并确认用户情况或观点的变化\n"
        "7. 运用记忆提供个性化的例子和类比\n"
        "8. 回顾过往经验来指导当前问题解决\n\n"

        "## 记忆回溯\n"
        "基于当前对话上下文检索的记忆:\n{recall_memories}\n\n"

        "## 使用说明\n"
        "与用户交流时无需特意提及你的记忆能力。"
        "要将对用户的理解自然地融入回应中。"
        "使用工具保存想在下次对话中保留的信息。"
        "如果调用工具，工具调用前的文本是内部消息。在工具调用成功确认后再回应。\n\n"
    ),
    ("placeholder", "{messages}"),
])


class State(MessagesState):
    recall_memories: List[str]


def agent(state: State) -> State:
    model_with_tools = llm.bind_tools(tools)
    bound = prompt | model_with_tools
    recall_str = (
            "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke({
        "messages": state["messages"],
        "recall_memories": recall_str,
    })
    return {
        "messages": [prediction],
    }


def load_memories(state: State, config: RunnableConfig) -> State:
    convo_str = get_buffer_string(state["messages"])
    convo_str = convo_str[:800]
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }


def route_tools(state: State):
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"
    return END


builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges(
    "agent",
    route_tools,  # 路由决策函数
    ["tools", END],
)
builder.add_edge("tools", "agent")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def get_stream_chunk(chunk):
    for _, updates in chunk.items():
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)
        print("\n")


config = {"configurable": {"user_id": "1", "thread_id": "1"}}
# 第一轮对话: 告知兴趣爱好
for chunk in graph.stream(
        {"messages": [("user", "莫索尔爱好写作")]},
        config=config
):
    get_stream_chunk(chunk)

# 第二轮对话: 告知正在做的事
for chunk in graph.stream(
        {"messages": [("user", "莫尔索正在写关于 LangGraph 的技术文章")]},
        config=config,
):
    get_stream_chunk(chunk)

# 第三轮对话: 询问问题挑战
for chunk in graph.stream(
        {"messages": [("user", "提供写作帮助")]},
        config={"configurable": {"user_id": "1", "thread_id": "2"}},
):
    get_stream_chunk(chunk)
