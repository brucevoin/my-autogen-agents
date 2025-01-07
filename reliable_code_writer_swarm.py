from asyncio import subprocess
from typing import AsyncGenerator, List, Sequence
import os
import subprocess
import psutil
import asyncio
import platform
import socket

from autogen_agentchat.ui import Console
from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent, UserProxyAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import AgentMessage, ChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import (
    HandoffTermination,
    MaxMessageTermination,
    TextMentionTermination,
)


async def main():
    # Create the agents
    os.environ["OPENAI_API_KEY"] = "sk-fb600a333ea14bfc97c83cbc3ac27417"
    api_key = os.getenv("OPENAI_API_KEY")
    model_client = OpenAIChatCompletionClient(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        model_capabilities={
            "vision": False,
            "function_calling": True,
            "json_output": True,
        },
    )

    async def execute_code(
        code: Annotated[str, "Code to execute"],
        language: Annotated[str, "Language of the code"] = "python",
    ):
        executor = LocalCommandLineCodeExecutor(work_dir="coding")
        code_executor = CodeExecutorAgent(
            "code_executor",
            code_executor=executor,
        )

        code = f"```{language}\n{code}```"

        # Run the agent with a given code snippet.
        task = TextMessage(
            content=code,
            source="user",
        )
        response = await code_executor.on_messages([task], CancellationToken())
        return response.chat_message.content

    execute_code_tool = FunctionTool(
        execute_code,
        name="execute_code",
        description="Execute the provided Python or Bash code and return the result.",
    )

    # code_executor_agent = AssistantAgent(
    #     name="executor_agent",
    #     model_client=model_client,
    #     system_message="""Execute the code provided by the coder_agent.
    #         - Don't write or modify any code yourself.
    #         - Handoff to reviewer_agent for review after execution.""",
    #     handoffs=["reviewer_agent"],
    #     tools=[execute_code_tool],
    # )

    async def save_code(code: str, file_path: str) -> str:
        """Save the provided code to a file at the specified path."""
        try:
            with open(file_path, "w") as f:
                f.write(code)
            return f"Code successfully saved to {file_path}"
        except Exception as e:
            return f"Error saving code: {e}"

    # save_code_tool = FunctionTool(
    #     save_code,
    #     name="save_code",
    #     description="Save the provided code to a file at the specified path.",
    # )

    async def delete_code(file_path: str) -> str:
        """Delete the code file at the specified path."""
        try:
            os.remove(file_path)
            return f"Code file successfully deleted at {file_path}"
        except Exception as e:
            return f"Error deleting code file: {e}"

    code_writer_agent = AssistantAgent(
        name="code_writer_agent",
        model_client=model_client,
        system_message="""You are a helpful AI assistant. Your task is:
        First, write code based on the user's request and the feedback from code_tester_agent. 
        Second, save the code to a file at the specified path. 
        
        You have access the following tools:
        - save_code_tool: save the code to a file at the specified path.
        
        Handoff the code file to code_tester_agent for testing.
        Handoff to user for clarification if needed.
        """,
        # - code_base_tool: read the codebase and understand the context.
        # - delete_code_tool: delete the code file at the specified path.
        # - documentation_tool: read the documentation and understand the context.
        tools=[save_code, delete_code],
        handoffs=["code_tester_agent", "user"],
    )

    async def read_code(file_path: str) -> str:
        """Read the code file at the specified path."""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading code file: {e}"

    async def write_test_code(file_path: str, test_code: str) -> str:
        """Write the test code to a file at the specified path."""
        try:
            with open(file_path, "w") as f:
                f.write(test_code)
            return f"Test code written to {file_path}"
        except Exception as e:
            return f"Error writing test code: {e}"

    async def execute_test_code(file_path: str) -> str:
        """Execute the code file at the specified path."""
        try:
            result = subprocess.run(
                ["python", file_path], capture_output=True, text=True
            )
            return result.stdout
        except Exception as e:
            return f"Error executing code: {e}"

    code_tester_agent = AssistantAgent(
        name="code_tester_agent",
        model_client=model_client,
        system_message="""You are a helpful AI assistant. Your task is:
        1. Read the code from file that was written by code_writer_agent;
        2. Write test code for the code and save it to a file;
        3. Execute the test code to check the results;
        
        You have access the following tools:
        - read_code_tool: read the code written by code_writer_agent.
        - code_writer_tool: write test code for the code written by code_writer_agent. 
        - execute_code_tool: execute the test code and check the results.
      
        Provide feedback and handoff to code_writer_agent If the code fails the test. 
        Handoff to the user If the code passes the test, .
        """,
        tools=[read_code, write_test_code, execute_test_code],
        handoffs=["code_writer_agent", "user"],
    )

    # code_file_manager_agent = AssistantAgent(
    #     name="code_file_manager_agent",
    #     model_client=model_client,
    #     system_message="""You are a helpful AI assistant. You manage code files written by code_writer_agent. You have access the following tools:
    #     - save_code_tool: save the code written by code_writer_agent to a file.
    #     - read_code_tool: read the code from a file.
    #     - create_directory_tool: create a directory to save the code files.
    #     - delete_code_file_tool: delete the code file.
    #     """,
    # )

    user_proxy_agent = UserProxyAgent(
        name="uer",
    )

    termination = HandoffTermination(target="user")

    # response = await asyncio.create_task(
    #                 user_proxy_agent.on_messages(
    #                     [TextMessage(content="Input your task: ", source="user")],
    #                     cancellation_token=CancellationToken(),
    #                 )
    #             )
    # task = response.chat_message.content

    team = Swarm(
        [code_writer_agent, code_tester_agent, user_proxy_agent],
        termination_condition=termination,
    )

    while True:
        task = input("Enter task: ")
        if task == "exit":
            break
        if task == "reset":
            await team.reset()
            continue
        if task == "":
            continue
        last_processed = await Console(team.run_stream(task=task))
        await team.reset()


if __name__ == "__main__":
    asyncio.run(main())
