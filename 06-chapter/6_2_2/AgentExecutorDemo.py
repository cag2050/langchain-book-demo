from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

load_dotenv()


@tool
def count_unique_chinese_characters(sentence_param):
    """用于计算句子中不同汉字的数量"""
    unique_characters = set()
    for char in sentence_param:
        if '\u4e00' <= char <= '\u9fff':
            unique_characters.add(char)
    return len(unique_characters)


sentence = "'如何用LangChain实现一个代理'这句话共包含多少个不同的汉字"

# 创建一个聊天提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_output"),
    ]
)

llm = ChatDeepSeek(model="deepseek-chat")
# 将工具函数绑定到模型上
llm_with_tools = llm.bind(functions=[convert_to_openai_function(count_unique_chinese_characters)])

# 构建一个代理，它将处理输入、提示词、模型响应和输出解析
agent = (
            {
                "input": lambda x: x["input"],
                "agent_output": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
        ) | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

agent_executor = AgentExecutor(
    agent=agent,
    tools=[count_unique_chinese_characters],
    verbose=True,
)
print(agent_executor.invoke({
    "input": sentence
}))
