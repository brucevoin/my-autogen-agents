import re
import os
import asyncio
from dataclasses import dataclass
import tempfile
from typing import List
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
)
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient


@dataclass
class CodingMessage:
    user_task: str
    feedbak: str = ""


@dataclass
class CodeExecutionMessage:
    user_task: str
    code_message: str


@dataclass
class CodeExecutionResultMessage:
    user_task: str
    code: str
    code_execution_result: str


@dataclass
class FinalResult:
    value: str


@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""Write Python or Bash script in markdown block based on the user's task and feedback, and it will be executed.
Always save figures to file in the current directory. 
All code required to complete this task must be contained within a single response.
Do not include any additional text outside of the code block.""",
            )
        ]

    @message_handler
    async def handle_message(self, message: CodingMessage, ctx: MessageContext) -> None:
        self._chat_history.append(
            UserMessage(
                content=f"The user's task: {message.user_task}\n The feedback:{message.feedbak}",
                source="user",
            )
        )
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant:\n{result.content}")
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))  # type: ignore
        await self.publish_message(CodeExecutionMessage(user_task=message.user_task, code_message=result.content), DefaultTopicId())  # type: ignore


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


@default_subscription
class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor) -> None:
        super().__init__("An executor agent.")
        self._code_executor = code_executor

    @message_handler
    async def handle_message(
        self, message: CodeExecutionMessage, ctx: MessageContext
    ) -> None:
        code_blocks = extract_markdown_code_blocks(message.code_message)
        if code_blocks:
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )
            print(f"\n{'-'*80}\nExecutor:\n{result.output}")
            await self.publish_message(
                CodeExecutionResultMessage(
                    user_task=message.user_task,
                    code=message.code_message,
                    code_execution_result=result.output,
                ),
                DefaultTopicId(),
            )


@default_subscription
class CodeExecutionResultReviewer(RoutedAgent):
    _try_count = 0
    _try_count_max = 3

    def __init__(self, model_client, try_count_max=3) -> None:
        super().__init__("A code execution result reviewer agent.")
        self._model_client = model_client
        self._try_count_max = try_count_max
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=""" You are a code execution result reviewer.
                Consider the user's task and code execution result, Respond with 'APPROVE' to when the code execution result is correct and meets the user's task. Otherwise, Provide constructive feedback that can fix the code to meet the user's task.
                """,
            )
        ]

    @message_handler
    async def handle_message(
        self, message: CodeExecutionResultMessage, ctx: MessageContext
    ) -> None:
        self._chat_history.append(
            AssistantMessage(
                content=f"The user's task: {message.user_task} \n The code:{message.code}\n The code execution result:{message.code_execution_result}",
                source=ctx.sender.type,
            )
        )
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nReviewer:\n{result.content}")

        if "APPROVE" == result.content:
            await self.publish_message(
                FinalResult(
                    value=message.code_execution_result,
                ),
                DefaultTopicId(),
            )
        else:
            self._try_count += 1
            if self._try_count > self._try_count_max:
                failed_message = f"Task failed after tried {self._try_count_max} times."
                print(f"\n{'-'*80}\nReviewer:\n {failed_message}")
                await self.publish_message(
                    FinalResult(
                        value=failed_message,
                    ),
                    DefaultTopicId(),
                )
            else:
                await self.publish_message(
                    CodingMessage(user_task=message.user_task, feedbak=result.content),
                    DefaultTopicId(),
                )


class CodeAgent:
    def __init__(
        self, workdir: str, model_client: OpenAIChatCompletionClient, try_count_max=3
    ):
        self.model_client = model_client
        self.runtime = SingleThreadedAgentRuntime()
        self.try_count_max = try_count_max
        self.code_executor = LocalCommandLineCodeExecutor(work_dir=workdir)
        self.queue = asyncio.Queue[
            FinalResult
            | CodingMessage
            | CodeExecutionMessage
            | CodeExecutionResultMessage
        ]()

    async def setup(self):
        await Assistant.register(
            self.runtime, "assistant", lambda: Assistant(self.model_client)
        )
        await Executor.register(
            self.runtime, "executor", lambda: Executor(self.code_executor)
        )
        await CodeExecutionResultReviewer.register(
            self.runtime,
            "reviewer",
            lambda: CodeExecutionResultReviewer(
                self.model_client, try_count_max=self.try_count_max
            ),
        )

        async def output_result(
            _agent: ClosureContext,
            message: (
                FinalResult
                | CodingMessage
                | CodeExecutionMessage
                | CodeExecutionResultMessage
            ),
            ctx: MessageContext,
        ) -> None:
            # only output the final result
            if isinstance(message, FinalResult):
                await self.queue.put(message)

        await ClosureAgent.register_closure(
            self.runtime,
            "output_result",
            output_result,
            subscriptions=lambda: [DefaultSubscription()],
        )

    async def run(self, task: str) -> str:
        self.runtime.start()
        await self.runtime.publish_message(
            CodingMessage(user_task=task),
            DefaultTopicId(),
        )
        await self.runtime.stop_when_idle()
        finall_result = await self.queue.get()
        return finall_result.value


async def main() -> None:
    os.environ["OPENAI_API_KEY"] = ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables")

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
    work_dir = tempfile.mkdtemp()
    code_agent = CodeAgent(model_client=model_client, workdir=work_dir)
    await code_agent.setup()
    while True:
        task = input("Enter a task: ")
        if task.lower() in ["exit", "quit"]:
            break
        result = await code_agent.run(task=task)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
