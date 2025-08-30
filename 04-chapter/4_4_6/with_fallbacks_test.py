from dotenv import load_dotenv
import logging

from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models.tongyi import ChatTongyi

load_dotenv()
logging.basicConfig(level=logging.INFO)

deepseek_llm = ChatDeepSeek(model="deepseek-chat", request_timeout=0.1, max_retries=0)
ali_llm = ChatTongyi(max_retries=0)
llm = deepseek_llm.with_fallbacks([ali_llm])

if __name__ == "__main__":
    try:
        # 直接调用ChatDeepSeek，因为request_timeout的限制，会报错：Request timed out.
        print(deepseek_llm.invoke("鲁迅和周树人是一个人吗？"))
    except Exception as e:
        print("deepseek_llm 执行失败：", e)

    try:
        # ChatDeepSeek如果报错，再去请求ChatTongyi
        print(llm.invoke("鲁迅和周树人是一个人吗？"))
    except Exception as e:
        print("llm 执行失败：", e)
