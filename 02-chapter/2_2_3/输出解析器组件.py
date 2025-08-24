from dotenv import load_dotenv

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.llms.tongyi import Tongyi

load_dotenv()

llm = Tongyi()

text = "给生产杯子的公司取三个合适的中文名字，以逗号分隔的形式输出。"
messages = [HumanMessage(content=text)]

class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(",")

if __name__ == "__main__":
    llms_response = llm.invoke(text)
    print(CommaSeparatedListOutputParser().parse(llms_response))
