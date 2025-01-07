# my-autogen-agents

**code_agent_core**: A simple agent team using autogen-0.4 core API, aimed at solving problems through code generation and execution, consisting of a code generator, a code executor, and a code reviewer.
**code_assistant**: An agent team using autogen-0.4, designed to assist with coding tasks and problem-solving, consisting of a coder,a security reviewer,a code executor and a code execution reviewer.

**meta_agent**: An agent team using autogen-0.4, designed to assist with general tasks and problem-solving, consisting of a userproxy and a meta agent, the meta agent aways solve the problem by creating a new agent team that consists of a worker agent and a review agent(the worker agent capable of write code and execute code).

**react_agent**: An agent team using autogen-0.4, designed to assist with general tasks and problem-solving, ReAct style.

**reliable_code_writer_swarm**: An agent team using autogen-0.4, designed to assist with coding tasks , consisting of a coder, a code tester.