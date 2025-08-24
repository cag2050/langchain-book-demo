from typing import List
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser

load_dotenv()

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        return text.strip().split(",")

template = """你是一个能生成以逗号分隔的列表的助手，用户会传入一个类别，你应该生成该类别下的5个对象，并以逗号分隔的形式返回。
只返回以逗号分隔的内容，不要包含其他内容。"""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

if __name__ == "__main__":
    chain = chat_prompt | ChatDeepSeek(model="deepseek-chat") | CommaSeparatedListOutputParser()
    print(chain.invoke({"text": "动物"}))
