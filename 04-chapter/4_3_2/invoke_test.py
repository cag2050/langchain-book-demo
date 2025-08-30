from dotenv import load_dotenv

from langchain_community.llms.tongyi import Tongyi

load_dotenv()

model = Tongyi()
response = model.invoke("什么是机器学习？")

if __name__ == "__main__":
    print(response)
