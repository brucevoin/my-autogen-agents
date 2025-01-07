from asyncio import subprocess
from typing import AsyncGenerator, List, Sequence
import os
import subprocess
import psutil
import asyncio
import platform
import socket

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
from autogen_agentchat.ui import Console


class CodeAgentGroup:
    def __init__(self, model_client: OpenAIChatCompletionClient):

        self.model_client = model_client

        def user_input(message: str) -> str:

            print(message)

            user_message = input(
                "The code that will be executed may not security, Do you want to continue?(y/n): "
            )
            if user_message == "y":
                return "The code security check passed, Handoff to executor_agent for execution."
            else:
                return "The code execution cancled by user due to security, Handoff to summarizer_agent for summary."

        self.user = UserProxyAgent(name="user", input_func=user_input)

        def get_system_info():

            # OS information
            system = platform.system()
            node = platform.node()
            release = platform.release()
            version = platform.version()
            machine = platform.machine()
            processor = platform.processor()
            # Hardware information
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            # Python information
            python_version = platform.python_version()
            # bash information
            bash_version = (
                subprocess.check_output(["bash", "--version"])
                .decode("utf-8")
                .split("\n")[0]
            )
            # Network information
            ip_address = socket.gethostbyname(socket.gethostname())
            # File system information
            #file_system = psutil.disk_partitions()
            # Environment information
            #environment = os.environ

            #environment = "\n".join([f"{k}: {v}" for k, v in environment.items()])
            # Package information
            #package = subprocess.check_output(["pip", "list"]).decode("utf-8")
            
            file_system = "Unknown"
            environment = "Unknown"
            package = "Unknown"

            info = f"""System Information:
            
System: {system}
Node: {node}
Release: {release}
Version: {version}
Machine: {machine}
Processor: {processor}
CPU Count: {cpu_count}
Memory: {memory}
Disk: {disk}
Python Version: {python_version}
Bash Version: {bash_version}
IP Address: {ip_address}
File System: {file_system}
Environment: {environment}
Package: {package}"""

            return info

        system_info_tool = FunctionTool(
            name="get_system_info",
            description="Get the system information of the current machine.",
            func=get_system_info,
        )

        self.coder = AssistantAgent(
            model_client=self.model_client,
            name="coder_agent",
            system_message="""Write a Python or Bash script within a markdown code block based on the user's task, the system information, and the feedback from reviewer_agent.
             
            Here are some tips:
            - You can write Python or Bash code to access the internet, read/write files, and run system commands.
            - Use get_system_info tool to get the system information of the current machine if the task is related to the local machine.
            
            Here are the rules:
            - The script must be executable and contain all necessary code to complete the task in one response.
            - Save all figures as files in the current directory if needed.
            - Include no additional text outside of the code block.
            - Explain the code in the code block if needed.
            - Consider the approach that don't require api keys or other credentials if possible.
           
            
            After writing or updating the script, hand off to security_agent for security review .""",
            handoffs=["security_agent"],
            tools=[system_info_tool],
            # TODO add environment dectection tool for better code generation
        )

        self.security = AssistantAgent(
            model_client=self.model_client,
            name="security_agent",
            system_message="""Review the code provided by the coder_agent for security vulnerabilities. 
            Here are some examples of security vulnerabilities to look for:
            - Code that could delete or modify files on the user's computer system.
            - Code that could access or modify sensitive data on the user's computer system.
            - Code that could access or modify the user's network or internet connection.
            - Code that could access or modify the user's hardware or peripherals.
            - Code that could access or modify the user's software or operating system.
            - Code that could access or modify the user's browser or other applications.
            
            Explain the security vulnerabilities in the code if any.
            Handoff to executor_agent for execution if the code is secure.
            Otherwise handoff to user for further review and confirmation if the code is not secure for the user's computer system or data.
            """,
            handoffs=["user", "executor_agent", "summarizer_agent"],
        )

        self.execute_code_tool = FunctionTool(
            self.execute_code,
            name="execute_code",
            description="Execute the provided Python or Bash code and return the result.",
        )

        self.executor = AssistantAgent(
            name="executor_agent",
            model_client=self.model_client,
            system_message="""Execute the code provided by the coder_agent.
            - Don't write or modify any code yourself.
            - Handoff to reviewer_agent for review after execution.""",
            handoffs=["reviewer_agent"],
            tools=[self.execute_code_tool],
        )

        self.reviewer = AssistantAgent(
            model_client=self.model_client,
            name="reviewer_agent",
            system_message="""Consider the user's task and system information, Review the code written by the coder_agent and the code execution result by the executor_agent.
            - Explain the code and the code execution result.
            - Provide feedback and suggest improvements or another approach for the task and handoff to coder_agent for further improvement if the code execution result doesn't meet the user's task.
            - Handoff to summarizer_agent if the code is correct and meets the user's task or the task cannot be completed.""",
            handoffs=["summarizer_agent", "coder_agent"],
        )

        self.summarizer = AssistantAgent(
            model_client=self.model_client,
            name="summarizer_agent",
            system_message="""Summarize the final result of the task for the user. The summary should be concise and clear. end with 'TERMINATE' """,
        )

        self.termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(
            50
        )

        self.team = Swarm(
            [
                self.coder,
                self.security,
                self.user,
                self.executor,
                self.reviewer,
                self.summarizer,
            ],
            self.termination,
        )

    async def execute_code(
        self,
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

    async def run_task(self, task: str):
        return self.team.run_stream(task=task)

    async def reset(self):
        await self.team.reset()


async def main():
    while True:
        task = input("Enter task: ")
        if task == "exit":
            break
        if task == "reset":
            await agent_group.reset()
            continue
        if task == "":
            continue
        last_processed = await Console(await agent_group.run_task(task=task))
        await agent_group.reset()


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = ""
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
    agent_group = CodeAgentGroup(model_client=model_client)
    asyncio.run(main())
