from dotenv import load_dotenv

from langchain_community.llms.tongyi import Tongyi
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

load_dotenv()

llm = Tongyi()
chat_model = ChatDeepSeek(model="deepseek-chat")

text = "给生产杯子的公司取一个名字，直接输出最终名字。"
messages = [HumanMessage(content=text)]

if __name__ == "__main__":
    print(llm.invoke(text))
    print(chat_model.invoke(messages))
