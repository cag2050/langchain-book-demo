from langchain_core.example_selectors import BaseExampleSelector
from typing_extensions import Dict, List
import numpy as np


class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: dict[str, str]) -> None:
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict]:
        return np.random.choice(self.examples, size=2, replace=False)


if __name__ == "__main__":
    examples = [
        {"foo": "1"},
        {"foo": "2"},
        {"foo": "3"},
    ]
    example_selector = CustomExampleSelector(examples=examples)

    select_examples_1 = example_selector.select_examples({"foo": "foo"})
    print(select_examples_1)

    example_selector.add_example({"foo": "4"})
    print(example_selector.examples)

    select_examples_2 = example_selector.select_examples({"foo": "foo"})
    print(select_examples_2)
