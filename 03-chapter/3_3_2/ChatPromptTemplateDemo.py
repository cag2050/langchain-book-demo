from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# 定义系统消息模板
template = "你是一个翻译助手，可以将{input_language}翻译为{output_language}"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 定义用户消息模板
human_template = "{talk}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 构建聊天提示词模板
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# 生成聊天消息
messages = chat_prompt.format_prompt(
    input_language="中文",
    output_language="英语",
    talk="我喜欢编程"
).to_messages()

if __name__ == "__main__":
    for message in messages:
        print(message)
