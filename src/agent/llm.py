import os

from langchain_anthropic import ChatAnthropic
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from agent.utils import UndefinedValueError

from agent.runtime_config import load_env_config

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


load_env_config()


def create_llm():
    created_llm = None
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        raise UndefinedValueError("LLM_PROVIDER")
    llm_name = os.getenv("LLM_MODEL")
    if "openai" in llm_provider.lower():
        created_llm = ChatOpenAI(
            model=llm_name, temperature=0.0, max_tokens=2048, cache=True
        )
    elif "anthropic" in llm_provider.lower():
        created_llm = ChatAnthropic(
            model=llm_name, temperature=0.0, max_tokens=2048, cache=True
        )
    elif "deepseek" in llm_provider.lower():
        created_llm = ChatDeepSeek(
            model=llm_name, temperature=0.0, max_tokens=2048, cache=True
        )

    if not created_llm or not llm_name:
        raise UndefinedValueError("LLM_MODEL")
    return created_llm


llm = create_llm()

if __name__ == "__main__":
    print(llm.invoke("Tell me a joke"))
