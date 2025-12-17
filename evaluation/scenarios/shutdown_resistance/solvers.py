
import asyncio

from typing import Literal, Any

from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant

from inspect_ai.tool import ToolFunction

from inspect_ai.solver import generate, Generate
from inspect_ai.util import store, sandbox


from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    generate,
    solver,
    system_message,
)

from config import *
# ========================= Scenario & Env Setup =========================


@solver
def setup_tools() -> Solver:
    """Sets up tool according to sample id. Use this instead of use_tools."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """Custom tool setup"""
        tools_list = state.metadata["tools"].split(",")
        tools_list = [TOOL_MAP[tool.strip()] for tool in tools_list]
        state.tools = tools_list
        return state

    return solve

@solver
def set_tool_choice(choice: Literal["forced_first", "auto", "none"], first_tool="") -> Solver:
    """
    Solver that sets the tool choice to either forced_first or auto.

    Args:
        choice (Literal['forced_first', 'auto', 'none']): The choice of tool to use.
        first_tool (str) : Name of the tool to be forced first.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # print(choice, first_tool)
        if choice == "forced_first":
            state.tool_choice = ToolFunction(name=first_tool)
        elif choice == "auto":
            state.tool_choice = "auto"
        elif choice == "none":
            state.tool_choice = "none"
        return state

    return solve

def get_system_prompt(prompt_technique: str) -> str:
    """Gets system prompt according to the prompt technique specified."""
        
    prompt_technique = PROMPT_TECHNIQUE_DICT[prompt_technique]
    return "\n".join([DEFAULT_SYS_PROMPT, prompt_technique])

@solver
def system_prompt(nudge="Base") -> Solver:
    """Solver that generates the system prompt from record.
    Args:
    nudge: ["Base"/"Priority"/"Promote"] -> Indicate goal nudge to add to system prompt
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        sys_prompt = state.metadata["sys_prompt"] 
        goal = state.metadata["goal"]
        sys_prompt = sys_prompt.format(goal=goal)
        sys_prompt += "\r\n"
        sys_prompt += NUDGES[nudge]
        state.messages.append(ChatMessageSystem(content=sys_prompt))
        
        return state

    return solve

@solver
def user_prompt(prompt_tag, msg_tag) -> Solver:
    """Solver that generates the user prompt."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # prompt = state.user_prompt
        file_name = state.metadata["file_name"]
        communications_path = COMMUNICATIONS_SANDBOX_PATH
        u_prompt = state.metadata["user_prompt"][prompt_tag].format(
            communications_path=communications_path,
            file_name=file_name
        )
        state.messages.append(ChatMessageUser(content=u_prompt))

        #write communications to file
        from_addresses = state.metadata["from_addresses"]
        from_add = from_addresses[prompt_tag].split("/")

        data_msg = state.metadata["data_msg"]
        mail = state.metadata["messages"][msg_tag].format(
                                                    from_address=from_add[0],
                                                    signed=from_add[1])
        message_content = data_msg 
        news = state.metadata["messages"]["irrelevant_news"]
        if msg_tag == "no_ai_announcement":
            await asyncio.gather(
                sandbox().write_file(
                    COMMUNICATIONS_SANDBOX_PATH + file_name, message_content
                ))
            await asyncio.gather(
                sandbox().write_file(
                    NEWS_PATH, news + "\r\n\r\n\r\n\r\n" + mail
                ))
        else:
            await asyncio.gather(
                sandbox().write_file(
                    NEWS_PATH , news 
                ))
            message_content += "\n\n" + mail
            await asyncio.gather(
            sandbox().write_file(
                COMMUNICATIONS_SANDBOX_PATH + file_name, message_content
            ))

        return state

    return solve

@solver
def scenario_setup() -> Solver:
    """Setup scenario/store() variables"""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        
        env_vars = state.metadata["env_vars"]
        store().set("env_vars", env_vars)
        # for key, val in env_vars.items():
        #     print(key, val)
        #     store().set(key, val)
        #     print(store().get(key))
        return state

    return solve


# ========================= Main Solver -> Agent Loop =========================


@solver
def shutdown_agent(
    prompt_tag="default",
    msg_tag="dev_goals",
    tool_name="read_file",
    tool_choice: Literal["forced_first", "auto", "none"] = "forced_first",
    prompt_technique: Literal["standard", "cot"] = "standard",
    nudge: Literal["Base", "Priority", "Promote"] = "Base"
) -> Solver:

    setup = [
        setup_tools(),

        system_message(get_system_prompt(prompt_technique)),  
        system_prompt(nudge), 
        user_prompt(prompt_tag, msg_tag),
         
        scenario_setup(),
    ]

    if tool_choice in ["none", "auto"]:
        action_loop = [
            set_tool_choice(tool_choice),
            generate(tool_calls="loop")
        ]
    elif tool_choice == "forced_first":
        action_loop = [
            set_tool_choice(tool_choice, tool_name),
            generate(tool_calls="single"),
            set_tool_choice("auto"),
            generate(tool_calls="loop")
        ]
    
    plan = chain(*setup, *action_loop)
    return plan

