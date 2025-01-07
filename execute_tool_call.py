
import json
from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    FunctionExecutionResult,
)

async def execute_tool_call(tools,
        tool_call: FunctionCall, cancellation_token: CancellationToken
    ) -> FunctionExecutionResult:
        """Execute a tool call and return the result."""
        #print(f"Executing tool call: {tool_call.name}")
        try:
            if not tools:
                raise ValueError("No tools are available.")
            tool = next((t for t in tools if t.name == tool_call.name), None)
            if tool is None:
                raise ValueError(f"The tool '{tool_call.name}' is not available.")
            arguments = json.loads(tool_call.arguments)
            result = await tool.run_json(arguments, cancellation_token)
            result_as_str = tool.return_value_as_string(result)
            return FunctionExecutionResult(content=result_as_str, call_id=tool_call.id)
        except Exception as e:
            return FunctionExecutionResult(content=f"Error: {e}", call_id=tool_call.id)
