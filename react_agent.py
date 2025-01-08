from dataclasses import dataclass
import asyncio
from autogen_core.tools import FunctionTool
from autogen_core import (
    DefaultTopicId,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient


from execute_tool_call import execute_tool_call
from execute_code_tool import execute_code


@dataclass
class UserTaskMessage:
    content: str


@default_subscription
class ReactAgent(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__("react_agent")
        self._model_client = model_client
        self._chat_history = []
        self._chat_history.append(
            SystemMessage(
                content="""You are an AI following the ReAct paradigm.
                You can write code and execute it to solve problems when needed. 
                Here are the tools you can use:
                - execute_code: Execute code in a sandboxed environment.
                If the task is completed, Just reply with task result."""
            )
        )

        self._tools = []
        self._tools.append(
            FunctionTool(
                name="execute_code",
                description="Execute code in a sandboxed environment.",
                func=execute_code,
            )
        )

    @message_handler
    async def on_message(self, message: UserTaskMessage, ctx) -> None:
        self._chat_history.append(
            UserMessage(source="user", content=message.content, type="UserMessage")
        )
        result = await self._model_client.create(
            messages=self._chat_history, tools=self._tools
        )
        if isinstance(result.content, str):
            print(f"\n{'-'*80}\nreact_agent:\n{result.content}")
            return

        if isinstance(result.content, list):
            await self.do_react(result)

        print("Result:")
        print(self._chat_history[len(self._chat_history) - 1].content)

    async def do_react(self, result):
        results = await asyncio.gather(
            *[
                execute_tool_call(self._tools, call, CancellationToken())
                for call in result.content
            ]
        )
        print(f"\n{'-'*80}\nreact_agent:\n{results}")
        self._chat_history.append(
            AssistantMessage(
                source="assistant",
                content=f"reasioning: {result.content}",
                type="AssistantMessage",
            )
        )
        self._chat_history.append(
            SystemMessage(
                source="assistant",
                content=f"action result: {results[0].content} \n Should I continue?",
                type="SystemMessage",
            )
        )
        self._chat_history.append(
            UserMessage(
                source="user",
                content=f"Received feedback:{results[0].content} . What should we do next?",
                type="UserMessage",
            )
        )
        result = await self._model_client.create(
            messages=self._chat_history, tools=self._tools
        )
        if isinstance(result.content, str):
            print(f"\n{'-'*40}\nreact_agent{'-'*40}:\n{result.content}")
            self._chat_history.append(
                SystemMessage(
                    source="assistant",
                    content=result.content,
                    type="SystemMessage",
                )
            )
            return
        if isinstance(result.content, list):
            await self.do_react(result)


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
    await ReactAgent.register(
        runtime=runtime,
        type="react_agent",
        factory=lambda: ReactAgent(model_client=model_client),
    )
    runtime.start()
    await runtime.publish_message(
        UserTaskMessage(content="get some finance news from yahoo using yf package"),
        DefaultTopicId(),
    )
    await runtime.stop_when_idle()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
