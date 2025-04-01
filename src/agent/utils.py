class UndefinedValueError(ValueError):
    """
    A custom exception raised when a variable is not defined.

    Args:
        variable_name (str): The name of the undefined variable
        message (str, optional): Custom error message
    """

    def __init__(self, variable_name, message=None):
        if message is None:
            message = f"`{variable_name}` is required and not defined in `.env` environment variables."

        self.variable_name = variable_name

        super().__init__(message)


def stage_message_processor(messages):
    """Based on XY's `message_processor` with few additions. It is used for newest version of summarizer/summary."""
    message_json = {}
    step = 0

    index = 0
    found_human_feedback = False
    messages = [
        vars(message) if not isinstance(message, dict) else message
        for message in messages
    ]
    for message in messages:
        if message["name"] == "human_feedback":
            found_human_feedback = True
            index = messages.index(message)

    if not found_human_feedback:
        index = 0

    messages = messages[index:]

    for message in messages:
        if message["type"] != "tool":
            # if message['content'] is string:
            if isinstance(message["content"], str):
                if step == 0:
                    step += 1
                    continue  # skip human input as its duplicated
                name = message["name"] if "name" in message else ""
                message_json[f"Step {step} {name}:"] = {"detail": message["content"]}
            else:
                detail_cnt = 1
                details = {}
                for content in message["content"]:
                    details[f"Detail {detail_cnt} {content['type']}:"] = content
                    detail_cnt += 1
                name = message["name"] if "name" in message else ""
                message_json[f"Step {step} {name}:"] = {"detail": details}

            step += 1
    return message_json
