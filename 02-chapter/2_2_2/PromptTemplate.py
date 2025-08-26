from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("给生产{product}的公司取一个名字。")

if __name__ == "__main__":
    print(prompt.format(product="杯子"))
