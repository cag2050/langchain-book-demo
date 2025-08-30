from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek


def test():
    load_dotenv()

    prompt = PromptTemplate.from_template("给生产{product}的公司取一个名字")
    runnable = prompt | ChatDeepSeek(model="deepseek-chat") | StrOutputParser()
    print(runnable.input_schema.model_json_schema())
    print(runnable.output_schema.model_json_schema())


if __name__ == '__main__':
    test()
