from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

load_dotenv()

prompt = ChatPromptTemplate.from_template("给做{product}的公司，取一个名字，不超过5个字")


class MyConstructorCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain start")

    def on_chain_end(self, outputs, **kwargs):
        print("Chain end")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM start")

    def on_llm_end(self, response, **kwargs):
        print("LLM end")


def constructor_test():
    callbacks = [MyConstructorCallbackHandler()]
    # 在构造器中使用回调处理器
    llm = ChatDeepSeek(model="deepseek-chat", callbacks=callbacks)
    chain = prompt | llm
    # 这次运行将使用构造器中定义的回调
    output = chain.invoke({
        "product": "杯子"
    })
    print(output)


if __name__ == "__main__":
    constructor_test()
