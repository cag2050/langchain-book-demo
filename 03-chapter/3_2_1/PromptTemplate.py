from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("翻译这段文字：{text}，风格：{style}")
formatted_prompt = template.format(text="我爱编程", style="诙谐有趣")

if __name__ == "__main__":
    print(formatted_prompt)
