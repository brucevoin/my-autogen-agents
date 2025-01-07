from dataclasses import dataclass
import json
import re
import os
import asyncio
import tempfile
from typing import List
from typing_extensions import Annotated
from autogen_core.tools import FunctionTool
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
    ClosureAgent,
    ClosureContext,
    DefaultSubscription,
)
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
)
from autogen_core import CancellationToken, FunctionCall
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import AgentMessage, ChatMessage, TextMessage

from execute_tool_call import execute_tool_call
from execute_code_tool import execute_code

@dataclass
class UserTaskMessage:
    user_task: str


@dataclass
class WorkerTaskMessage:
    user_task: str


@dataclass
class TaskResultMessage:
    user_task: str
    result: str


@dataclass
class TaskReviewMessage:
    user_task: str
    result: str
    review: str


@dataclass
class FinalResultMessage:
    user_task: str
    result: str


@dataclass
class BrodcastMessage:
    message: str

@default_subscription
class MetaAgent(RoutedAgent):
    def __init__(
        self, model_client: ChatCompletionClient, min_agent_count=1, max_agent_count=2
    ) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=f""" You are a meta agent that can make other agents to solve problems. 
                
                - Use make_agent tool to make an agent to solve the problem. 
                - Use make_reviewer_agent tool to make a reviewer agent to review the result of the worker agent. 
                
                Do not solve the problem directly.""",
            )
        ]
        self._tools = []
        make_agent_tool = FunctionTool(
            self.make_agent,
            name="make_agent",
            description="Make an agent to solve the problem.",
        )
        self._tools.append(make_agent_tool)

        make_reviewer_agent_tool = FunctionTool(
            self.make_reviewer_agent,
            name="make_reviewer_agent",
            description="Make a reviewer agent to review the result of the worker agents.",
        )
        self._tools.append(make_reviewer_agent_tool)

    async def make_agent(
        self,
        name: Annotated[str, "The name of the agent"],
        system_message: Annotated[str, "The system message of the agent"],
    ) -> None:
        system_message += """ You should consider the feedback provide by reviewer agent and improve your work. You have access to the following tools: 
            - execute_code: Execute code in a given language(ensure content is printed to stdout)"""

        print(f"making agent:\nname: {name}\nsystem_message: {system_message}")

        code_executor_tool = FunctionTool(
            execute_code,
            name="execute_code",
            description="Execute code in a given language.",
        )

        # register the agent
        await WorkerAgent.register(
            self.runtime,
            f"Worker_{name}",
            lambda: WorkerAgent(
                name=name,
                system_message=system_message,
                model_client=self._model_client,
                tools=[code_executor_tool],
            ),
        )
        return "Agent made."

    async def make_reviewer_agent(
        self,
        name: Annotated[str, "The name of the agent"],
        system_message: Annotated[str, "The system message of the agent"],
    ) -> None:
        system_message += " Just Reply with 'APPROVE' if the result meets the task requirements, otherwise provide constructive feedback on how to improve it or another approach to take."
        print(f"making agent:\nname: {name}\nsystem_message: {system_message}")

        # register the agent
        await ReviewerAgent.register(
            self.runtime,
            f"Worker_{name}",
            lambda: ReviewerAgent(
                name=name,
                system_message=system_message,
                model_client=self._model_client,
            ),
        )
        return "Agent made."

    @message_handler
    async def handle_message(
        self, message: UserTaskMessage | BrodcastMessage, ctx: MessageContext
    ) -> None:
        if isinstance(message, BrodcastMessage):
            if message.message.lower() == "reset":
                self._chat_history = []
                return

        self._chat_history.append(
            UserMessage(
                content=message.user_task,
                source="user",
            )
        )

        result = await self._model_client.create(
            messages=self._chat_history, tools=self._tools
        )
        self._chat_history.append(AssistantMessage(content=result.content, source="MetaAgent"))  # type: ignore

        if isinstance(result.content, str):
            print(f"\n{'-'*80}\nMetaAgent:\n{result.content}")
            return
        if isinstance(result.content, list):
            results = await asyncio.gather(
                *[
                    execute_tool_call(self._tools, call, CancellationToken())
                    for call in result.content
                ]
            )
            print(f"\n{'-'*80}\nMetaAgent:\n{results}")

        await self.publish_message(WorkerTaskMessage(user_task=message.user_task), DefaultTopicId())  # type: ignore
        print("published task message to worker")


@default_subscription
class WorkerAgent(RoutedAgent):
    """A worker agent that can execute tasks."""

    def __init__(
        self,
        name: str,
        system_message: str,
        model_client: OpenAIChatCompletionClient,
        tools: List[FunctionTool] = None,
    ):
        super().__init__("An assistant agent.")
        self.name = f"Worker_{name}"
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [SystemMessage(content=system_message)]
        self._tools = tools

    @message_handler
    async def handle_message(
        self, message: WorkerTaskMessage | TaskReviewMessage, ctx: MessageContext
    ) -> None:
        """Handle a message from the user."""
        if isinstance(message, WorkerTaskMessage):
            self._chat_history.append(
                UserMessage(
                    content=message.user_task, type="UserMessage", source="user"
                )
            )
        elif isinstance(message, TaskReviewMessage):
            self._chat_history.append(
                UserMessage(content=message.review, type="UserMessage", source="user")
            )
        result = await self._model_client.create(self._chat_history, tools=self._tools)

        result_contest = ""
        if isinstance(result.content, str):
            result_contest = result.content

        if isinstance(result.content, list):
            results = await asyncio.gather(
                *[
                    execute_tool_call(self._tools, call, CancellationToken())
                    for call in result.content
                ]
            )
            result_contest = "\n".join([str(result.content) for result in results])

        print(f"\n{'-'*80}\n{self.type}:\n{result_contest}")
        self._chat_history.append(
            AssistantMessage(
                content=result_contest, type="AssistantMessage", source="assistant"
            )
        )
        await self.runtime.publish_message(
            TaskResultMessage(message.user_task, result_contest), DefaultTopicId()
        )


@default_subscription
class ReviewerAgent(RoutedAgent):
    """A reviewer agent."""

    def __init__(
        self, name: str, model_client: OpenAIChatCompletionClient, system_message: str
    ) -> None:
        super().__init__("A reviewer agent.")
        self.name = f"Reviewer_{name}"
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [SystemMessage(content=system_message)]
        self.try_count = 0

    @message_handler
    async def handle_message(
        self, message: TaskResultMessage, ctx: MessageContext
    ) -> None:
        """Handle a message from the user."""
        self._chat_history.append(
            UserMessage(
                content=f"task: {message.user_task}\nresult: {message.result}",
                type="UserMessage",
                source="user",
            )
        )
        result = await self._model_client.create(self._chat_history)
        self._chat_history.append(
            AssistantMessage(
                content=result.content, type="AssistantMessage", source="assistant"
            )
        )
        if result.content == "APPROVE" or result.content.__contains__("APPROVE"):
            await self.runtime.publish_message(
                FinalResultMessage(user_task=message.user_task, result=message.result),
                DefaultTopicId(),
            )
        else:
            self.try_count += 1
            if self.try_count > 3:
                await self.runtime.publish_message(
                    FinalResultMessage(
                        user_task=message.user_task,
                        result=f"The task failed after tried 3 times, Here is the final result: {message.result}",
                    ),
                    DefaultTopicId(),
                )
            else:
                await self.runtime.publish_message(
                    TaskReviewMessage(
                        user_task=message.user_task,
                        result=message.result,
                        review=result.content,
                    ),
                    DefaultTopicId(),
                )
        print(f"\n{'-'*80}\n{self.type}:\n{result.content}")


@default_subscription
class UserProxyAgent(RoutedAgent):
    def __init__(self):
        super().__init__("user")

    @message_handler
    async def handle_message(
        self, message: FinalResultMessage, ctx: MessageContext
    ) -> None:
        print(f"\n{'-'*80}\n Here is the final result:\n{message.result}")
        feedback = input("You can provide feedback or just press Enter to continue:")
        if feedback:
            await self.runtime.publish_message(
                TaskReviewMessage(
                    user_task=message.user_task,
                    result=message.result,
                    review=feedback,
                ),
                DefaultTopicId(),
            )


async def main():
    import os

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

    runtime = SingleThreadedAgentRuntime()
    await MetaAgent.register(
        runtime=runtime,
        type="MetaAgent",
        factory=lambda: MetaAgent(model_client=model_client),
    )
    await UserProxyAgent.register(
        runtime=runtime,
        type="UserProxyAgent",
        factory=lambda: UserProxyAgent(),
    )
    while True:
        user_task = input("Enter your task: ")
        if user_task == "exit":
            break
        runtime.start()
        await runtime.publish_message(
            UserTaskMessage(
                # user_task="Translate the following sentence to chinese: Hello everyone."
                # user_task="Find the 5 largest files in the directory /Users/brucevoin/Downloads/Others/."
                # user_task="Get me the latest financial news from yahoo finance."
                user_task=user_task,
            ),
            DefaultTopicId(),
        )
        await runtime.stop_when_idle()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
