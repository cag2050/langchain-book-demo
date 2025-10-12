from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

load_dotenv()

prompt = ChatPromptTemplate.from_template("给做{product}的公司，取一个名字，不超过5个字")


class MyRequestCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain start")

    def on_chain_end(self, outputs, **kwargs):
        print("Chain end")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM start")

    def on_llm_end(self, response, **kwargs):
        print("LLM end")


def request_test():
    callbacks = [MyRequestCallbackHandler()]
    llm = ChatDeepSeek(model="deepseek-chat")
    chain = prompt | llm
    # 在请求中使用回调处理器
    output = chain.invoke(
        {
            "product": "杯子"
        },
        config={"callbacks": callbacks})
    print(output)


if __name__ == "__main__":
    request_test()
