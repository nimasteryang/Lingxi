from langchain_anthropic import ChatAnthropic
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

# from langchain_openai.chat_models.base import BaseChatOpenAI
from agent.runtime_config import load_env_config

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


load_env_config()


def create_llm():
    llm = ChatAnthropic(model="claude-3-5-sonnet-latest",temperature=0.0,max_tokens=2048,cache=True)

    return llm


llm = create_llm()

if __name__ == "__main__":
    print(llm.invoke("Tell me a joke"))
    # pass
    # from typing import Optional

    # from pydantic import BaseModel, Field

    # class Joke(BaseModel):
    #     '''Joke to tell user.'''

    #     setup: str = Field(description="The setup of the joke")
    #     punchline: str = Field(description="The punchline to the joke")
    #     rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

    # # print(llm.invoke("你好"))
    # structured_llm = llm.with_structured_output(Joke)
    # ai_msg = structured_llm.invoke("Tell me a joke about cats")
    # print(ai_msg)

    # import json
    # from openai import OpenAI

    # client = OpenAI(
    #     api_key=os.getenv("SILICONFLOW_API_KEY"), # 从https://cloud.siliconflow.cn/account/ak获取
    #     base_url="https://api.siliconflow.cn/v1"
    # )

    # response = client.chat.completions.create(
    #         model="deepseek-ai/DeepSeek-V3",
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    #             {"role": "user", "content": "? 2020 年世界奥运会乒乓球男子和女子单打冠军分别是谁? "
    #             "Please respond in the format {\"男子冠军\": ..., \"女子冠军\": ...}"}
    #         ],
    #         response_format={"type": "json_object"}
    #     )

    # print(response.choices[0].message.content)

    # print(llm.with_structured_output(Joke,method="json_mode").invoke("Tell me a joke about cats"))

    # from pydantic import BaseModel, Field

    # class GetWeather(BaseModel):
    #     '''Get the current weather in a given location'''

    #     location: str = Field(
    #         ..., description="The city and state, e.g. San Francisco, CA"
    #     )

    # class GetPopulation(BaseModel):
    #     '''Get the current population in a given location'''

    #     location: str = Field(
    #         ..., description="The city and state, e.g. San Francisco, CA"
    #     )

    # llm_with_tools = llm.bind_tools(
    #     [GetWeather, GetPopulation]
    # # strict = True  # enforce tool args schema is respected
    # )
    # ai_msg = llm_with_tools.invoke(
    #     "Which city is hotter today and which is bigger: LA or NY?"
    # )
    # print(ai_msg)
