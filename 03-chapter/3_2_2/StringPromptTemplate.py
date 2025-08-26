import json
from pydantic import BaseModel, field_validator
from langchain_core.prompts import StringPromptTemplate

delimiter = "####"
PROMPT = f"""将每个用户的信息用{delimiter}字符分割，并按照JSON格式提取姓名、职业和爱好信息。
示例如下："""


class PersonInfoPromptTemplate(StringPromptTemplate, BaseModel):
    # 验证输入变量
    @field_validator("input_variables")
    def validate_input_variables(cls, input_variables):
        if "name" not in input_variables:
            raise ValueError("name is required")
        if "occupation" not in input_variables:
            raise ValueError("occupation is required")
        if "fun_fact" not in input_variables:
            raise ValueError("fun_fact is required")
        return input_variables

    # 格式化输入，生成JSON格式输出
    def format(self, **kwargs):
        person_info = {
            "name": kwargs.get("name"),
            "occupation": kwargs.get("occupation"),
            "fun_fact": kwargs.get("fun_fact"),
        }
        return PROMPT + json.dumps(person_info, ensure_ascii=False)

    # 指定模板类型
    def _prompt_type(self) -> str:
        return "person-info"


# 使用模板
person_info_template = PersonInfoPromptTemplate(input_variables=["name", "occupation", "fun_fact"])
prompt_output = person_info_template.format(
    name="张三",
    occupation="软件工程师",
    fun_fact="喜欢攀岩"
)

if __name__ == "__main__":
    print(prompt_output)
