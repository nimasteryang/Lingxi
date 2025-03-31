from string import Template


FILTER_CONVO_HISTORY_SYSTEM_PROMPT = Template(
    """The ${next_agent} will perform its task next. Based on the chat history, filter out irrelevant messages so that the agent can focus on performing its task with useful information.
Here are the previous messages in JSON:\n```json\n${messages}```

Only respond with your result as a JSON-compliant list of strings for each of the ids of the relevant messages remaining after your filtering. An example of the format is below:
```["id1", "id2"]```"""
)
FILTER_CONVO_HISTORY_SYSTEM_PROMPT_V2 = Template(
    """The ${next_agent} will perform its task next. Based on the chat history, filter out irrelevant messages so the agent can focus on highly relevant information while performing its task.
Here is the task description of the agent:
```
${next_agent_prompt}
```

Here are the previous messages in JSON:
```json
${messages}
```

Only respond with your result as a JSON-compliant list of strings for each of the ids of the relevant messages remaining after your filtering. An example of the format is below:
```["id1", "id2"]```"""
)

RELEVANT_FILE_EXPLANATION_SYSTEM_PROMPT = Template(
    """Given a search term ${search_term}, a vector database performing similarity search of embeddings between the search term and code snippets of files and functions/methods in the project returned ${k} relevant documents. For each document, provide a description explaining why the search term is relevant to the code retrieved from the database. Below is a list of the filepaths and their corresponding code snippets in JSON format:
```${full_result}```

Only respond with your result as a list of JSON with the "file_path" key and the "explanation" key for your corresponding explanation. An example of the format is below:
```[{\"file_path\": \"filepath1/file1.py\", \"explanation\": \"This file contains the keyword \"UIButton\" from the search term\"}]```"""
)