from typing import List
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_deepseek import ChatDeepSeek
from langserve import add_routes

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
first_chain = chat_prompt | ChatDeepSeek(model="deepseek-chat") | CommaSeparatedListOutputParser()

app = FastAPI(
    title="第一个LangChain应用",
    version="0.0.1",
    description="LangChain应用接口",
)

add_routes(app, first_chain, path="/first-app")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
