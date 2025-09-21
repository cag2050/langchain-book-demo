from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

messages = [
    HumanMessage("你好AI，我想聊聊我的爱好和职业。"),
    AIMessage("你好李四，很高兴和你聊天。你的爱好是什么呢？"),
    HumanMessage("我喜欢阅读和徒步旅行。你呢？"),
    AIMessage("作为一个AI，我‘喜欢’处理数据和帮助解决问题。你从事什么职业？"),
    HumanMessage("我是一名软件工程师。总是有很多问题需要解决。"),
    AIMessage("那听起来很有趣！你最喜欢编程的哪个部分？"),
    HumanMessage("我最喜欢解决复杂的算法问题。那对你来说，最大的挑战是什么？"),
    AIMessage("对我来说，最大的挑战是如何更自然地与人类沟通。"),
]

"""
下面设置：token_counter=ChatDeepSeek(model="deepseek-chat"), 的时候，报错如下，不要使用这个ChatDeepSeek模型：
File "/Users/cag2050/Documents/PycharmProjects/langchain-book-demo/venv/lib/python3.11/site-packages/langchain_openai/chat_models/base.py", line 1543, in get_num_tokens_from_messages
    raise NotImplementedError(
NotImplementedError: get_num_tokens_from_messages() is not presently implemented for model deepseek-chat. See https://platform.openai.com/docs/guides/text-generation/managing-tokens for information on how messages are converted to tokens.
"""

"""
下面设置：token_counter=ChatTongyi(model="qwen-plus"), 或 token_counter=ChatTongyi(model="qwen-max"), 的时候，报错如下，不要使用这个ChatTongyi模型：
  File "/Users/cag2050/Documents/PycharmProjects/langchain-book-demo/venv/lib/python3.11/site-packages/transformers/models/gpt2/tokenization_gpt2.py", line 153, in __init__
    with open(vocab_file, encoding="utf-8") as vocab_handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: expected str, bytes or os.PathLike object, not NoneType
"""

# token_counter使用模型来计算
selectd_messages = trim_messages(
    messages,
    # token_counter=ChatDeepSeek(model="deepseek-chat"),  # 使用模型的词法分析器计算消息中的 token 数，报错信息如上，不要使用
    # token_counter=ChatTongyi(model="qwen-plus"),  # 使用模型的词法分析器计算消息中的 token 数，报错信息如上，不要使用
    # token_counter=ChatTongyi(model="qwen-max"),  # 使用模型的词法分析器计算消息中的 token 数，报错信息如上，不要使用
    token_counter=ChatOpenAI(model="gpt-4o"),  # 使用模型的词法分析器计算消息中的 token 数
    max_tokens=100,  # # 设置令牌数量的上限
    strategy="last",  # 选择策略为“最后”，即优先保留列表末尾的消息
    start_on="human",  # 确保对话历史以人类消息开始
    include_system=True,  # 如果原始对话历史中包含系统消息，则保留它，因为系统消息可能包含对模型的特殊指令
)

for msg in selectd_messages:
    msg.pretty_print()
