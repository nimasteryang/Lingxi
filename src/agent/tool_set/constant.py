
MAX_LIST_FILES: int = 400

MAX_RESPONSE_LEN_CHAR: int = 32000

SNIPPET_CONTEXT_WINDOW = 4

CONTENT_TRUNCATED_NOTICE: str = '<response clipped><NOTE>Due to the max output limit, only part of the full response has been shown to you.</NOTE>'

FILE_CONTENT_TRUNCATED_NOTICE: str = '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. To locate specific content, consider using the `search_file_by_keywords` tool or review the file structure below to identify the relevant line numbers. Then, re-run this tool using the `view_range` parameter to retrieve the desired content.</NOTE>'

DIRECTORY_CONTENT_TRUNCATED_NOTICE: str = '<response clipped><NOTE>Due to the max output limit, only part of this directory has been shown to you. You should use `view_directory` tool instead to view large directories incrementally.</NOTE>'