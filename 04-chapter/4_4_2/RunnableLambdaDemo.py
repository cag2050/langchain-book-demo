from langchain_core.runnables import RunnableLambda


def add(x):
    return x + x


def multiply(x):
    return x * 2


add_runnable = RunnableLambda(add)
multiply_runnable = RunnableLambda(multiply)

chain = add_runnable | multiply_runnable

if __name__ == '__main__':
    print(chain.invoke(3))
    print(chain.invoke(4))
