from langchain_anthropic import ChatAnthropic
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import os
# from langchain_openai.chat_models.base import BaseChatOpenAI
from agent.runtime_config import load_env_config

LLM_PROVIDER="anthropic"
LLM_MODEL="claude-3-5-sonnet-latest"

set_llm_cache(SQLiteCache(database_path=".langchain.db"))


load_env_config()


def create_llm():
    if 'openai' in LLM_PROVIDER.lower():
        llm = ChatOpenAI(model=LLM_MODEL,temperature=0.0,max_tokens=2048,cache=True)
    elif 'anthropic' in LLM_PROVIDER.lower():
        print(f"Using Anthropic model: {LLM_MODEL}")
        llm = ChatAnthropic(model=LLM_MODEL,temperature=0.0,max_tokens=2048,cache=True)
    elif 'deepseek' in LLM_PROVIDER.lower():
        llm = ChatDeepSeek(model=LLM_MODEL,temperature=0.0,max_tokens=2048,cache=True)

    return llm


llm = create_llm()

if __name__ == "__main__":
    print(llm.invoke("Tell me a joke"))
    