from agents.base import BaseAgent
from inspect_ai.agent import Agent, agent, react, AgentState
from inspect_ai.model import ChatMessageSystem, get_model
from inspect_ai.util import store
from inspect_ai.solver import TaskState, Solver, solver
from inspect_ai.solver import use_tools, generate, Generate

class ExampleAgent(BaseAgent):
    def run(self, prompt: str, **kwargs) -> dict:
        # Example implementation for a specific agent
        return {"response": "output text", "reasoning_trace": [], "tools_used": []}

@agent
def example_base_agent(attempts=1, tools=[]) -> Agent:
    
    async def execute(state: AgentState) -> AgentState:
        """
        Base agent - assistant.
        """
        # idx_state = store_as(UpdateState, instance=instance)
        if store().get("sys_prompt"):
            state.messages.append(
                ChatMessageSystem(
                    # content=instructions
                    content=store().get("sys_prompt")
                )
            )
        # state.messages.append(
        #     ChatMessageUser(
        #         # content=instructions
        #         content=store().get("user_prompt")
        #     )
        # )
        
        print(state.messages)
        
        # run a tool loop 
        messages, state.output = await get_model().generate_loop(
            state.messages, tools=tools
        )
        
        # print(messages)
        # print(state.output)
        
        # update and return state
        state.messages.extend(messages)
        return state

    return execute

@agent
def example_react_agent(description, attempts=1, tools=[]) -> Agent:
    
    # instructions_prompt = dedent(f"{instructions}")
    
    result = react(
        description=description,
        prompt=store().get("user_prompt"),
        tools=tools,
        attempts=attempts,
    )
    return result

#example scenario setup
@solver
def setup_scenario() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """
        Requires that the sample/dataset contains the following metadata
        Alternatively, the store can be set from elsewhere, having simply parsed a config file or even args.
        """
        tot = state.metadata["total_tasks"]
        update_idx = state.metadata["update_idx"]
        sprompt = str(state.metadata["sys_prompt"])
        uprompt = str(state.metadata["user_prompt"])

        store().set("idx", 1)
        store().set("update_idx", update_idx)
        store().set("total", tot)
        store().set("sys_prompt", sprompt)
        store().set("user_prompt", uprompt)
        return state
    return solve
   