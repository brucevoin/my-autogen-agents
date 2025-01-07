from autogen_core import CancellationToken
from typing_extensions import Annotated
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage


async def execute_code(
    code: Annotated[str, "Code to execute"],
    language: Annotated[str, "Language of the code"] = "python",
):
    code_executor = LocalCommandLineCodeExecutor(work_dir="coding")
    code_executor_agent = CodeExecutorAgent(
        "code_executor", code_executor=code_executor
    )

    code = f"```{language}\n{code}```"

    # Run the agent with a given code snippet.
    task = TextMessage(
        content=code,
        source="user",
    )
    response = await code_executor_agent.on_messages([task], CancellationToken())
    return response.chat_message.content
