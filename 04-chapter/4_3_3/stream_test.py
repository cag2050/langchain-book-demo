from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate


def test():
    load_dotenv()

    model = ChatDeepSeek(model="deepseek-chat")

    prompt = PromptTemplate.from_template(template="讲一个{story_type}的故事")

    runnable = prompt | model

    for s in runnable.stream({"story_type": "爱情"}):
        print(s.content, end="", flush=True)


if __name__ == "__main__":
    test()
