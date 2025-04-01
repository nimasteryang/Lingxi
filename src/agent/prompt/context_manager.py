from string import Template

RELEVANT_FILE_EXPLANATION_SYSTEM_PROMPT = Template(
    """Given a search term ${search_term}, a vector database performing similarity search of embeddings between the search term and code snippets of files and functions/methods in the project returned ${k} relevant documents. For each document, provide a description explaining why the search term is relevant to the code retrieved from the database. Below is a list of the filepaths and their corresponding code snippets in JSON format:
```${full_result}```

Only respond with your result as a list of JSON with the "file_path" key and the "explanation" key for your corresponding explanation. An example of the format is below:
```[{\"file_path\": \"filepath1/file1.py\", \"explanation\": \"This file contains the keyword \"UIButton\" from the search term\"}]```"""
)
