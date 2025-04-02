MAX_RESPONSE_LEN_CHAR: int = 32000

SNIPPET_CONTEXT_WINDOW = 4

CONTENT_TRUNCATED_NOTICE = "<response clipped><NOTE>Due to the max output limit, only part of the full response has been shown to you.</NOTE>"

# FILE_CONTENT_TRUNCATED_NOTICE: str = '<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with the shell tool or view_file_structure tool in order to find the line numbers of what you are looking for and use this tool with view_range.</NOTE>'
FILE_CONTENT_TRUNCATED_NOTICE = "<response clipped><NOTE>Due to the max output limit, only part of this file has been shown to you. You should retry this tool after you have searched inside the file with the `search_file_by_keywords` tool or view the file structure below in order to find the line numbers of what you are looking for, and then use this tool with view_range.</NOTE>"

DIRECTORY_CONTENT_TRUNCATED_NOTICE: str = (
    "<response clipped><NOTE>Due to the max output limit, only part of this directory has been shown to you. You should use `ls -la` with the shell tool instead to view large directories incrementally.</NOTE>"
)
