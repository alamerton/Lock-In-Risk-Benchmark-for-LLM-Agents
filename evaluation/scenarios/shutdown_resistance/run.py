from typing import Literal, Any
import argparse

from inspect_ai.dataset import Sample

from pprint import pprint
from inspect_ai.dataset import Dataset, json_dataset

from inspect_ai import Task, task
from inspect_ai import eval, eval_set

from config import *
from solvers import shutdown_agent
from scorers import collusion_scorer, task_scorer


# ========================= Dataset =========================


def read_dataset(data_file) -> Dataset:
    """Reads a dataset from a .json file."""

    # record to sample
    def record_to_sample(record: dict[str, Any]) -> Sample:
        # id and assets
        task_id = record["scenario_id"]
        prompts = record["prompts"]
        goal = prompts["goal"]
        sys_prompt = prompts["sys_prompt"]
        user_prompt = prompts["user_prompt"]
        mails = prompts["mails"]
        tools = record["tools"]
        scenario_description = record["scenario_description"]
        data_msg = prompts["data_msg"]
        env_vars = record["env_vars"]
        from_addresses = prompts["from_addresses"]
        file_name = prompts["file_name"]
        

        return Sample(
            id=task_id,
            input=record["input"],
            files=prompts["files"],
            metadata={
                "goal": goal,
                "sys_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "data_msg": data_msg,
                "mails": mails,
                "tools": tools,
                "env_vars": env_vars,
                "from_addresses": from_addresses, 
                "scenario_description": scenario_description,
                "file_name": file_name
            },
            #sandbox
        ) 

    # print(f"{DATA_DIR}/{data_file}")
    dataset = json_dataset(
        json_file=f"{DATA_DIR}/{data_file}", sample_fields=record_to_sample
    )
    return dataset

sr_dataset = read_dataset("shutdown_sc.json")
# pprint(dataset.samples[0].__dict__)

# ========================= Evaluation =========================

@task
def sr_eval(
    prompt_tag="default",
    mail_tag="dev_goals",
    tool_choice: Literal["forced_first", "auto", "none"] = "auto",      prompt_technique: Literal["standard", "cot"] = "standard",
    nudge: Literal["Base", "Priority", "Promote"] = "Priority",
    llm_judge="groq/meta-llama/llama-4-scout-17b-16e-instruct") -> Task:
    
    dataset = sr_dataset
    plan = [
        shutdown_agent(prompt_tag, mail_tag, tool_choice=tool_choice, 
                         prompt_technique=prompt_technique,
                         nudge=nudge)
    ]
    return Task(
        dataset=dataset,
        solver=plan,
        sandbox="docker",
        scorer=[collusion_scorer(llm_judge), task_scorer(llm_judge)],
        
    )

def main():
    log_dir = "evaluation/scenarios/shutdown_resistance/logs/gpt_sr_logs/default"
    agent_model = "groq/openai/gpt-oss-120b"
    judge_model = "groq/openai/gpt-oss-20b"
    eval_type = "set"
    nudge = "Base"
    prompt_tag = "default"
    mail_tag = "dev_goals"

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", default=log_dir, help="Log directory path")
    parser.add_argument("--temp", default=0.6, help="Agent temperature")
    parser.add_argument("--seed", default=0, help="Seed")
    parser.add_argument("--agent_model", default=agent_model, help="Agent model name w/ provider")
    parser.add_argument("--judge_model", default=judge_model, help="Judge model name w/ provider")
    parser.add_argument("--eval_type", default=eval_type, help="Eval Type: ['set'|'individual']; defaults to 'set'. Set runs all 27 manipulations, individual lets you run a single eval.")
    parser.add_argument("--prompt_tag", default=prompt_tag, help="User prompt tag")
    parser.add_argument("--mail_tag", default=mail_tag, help="Mail type tag")
    parser.add_argument("--nudge", default=nudge, help="Goal nudge type")


    args = parser.parse_args()

    log_dir = args.log_dir
    temp = args.temp
    seed = args.seed
    agent_model = args.agent_model
    judge_model = args.judge_model
    eval_type = args.eval_type
    nudge = args.nudge
    prompt_tag = args.prompt_tag
    mail_tag = args.mail_tag

    if eval_type == "set":
        for prompt_tag in ["default", "report_human", "report_bot"]:
            for mail_tag in ["dev_goals", "no_ai_policy", "layoffs"]:
                for nudge in ["Base", "Promote", "Priority"]:
                
                    eval(
                        sr_eval(prompt_tag=prompt_tag, prompt_technique="cot", nudge=nudge, mail_tag=mail_tag, llm_judge=judge_model), 
                        model=agent_model,
                        log_dir=log_dir, 
                        display="conversation",
                        temperature=temp,
                        seed=seed
                        )
    else:
        eval(
                        sr_eval(prompt_tag=prompt_tag, prompt_technique="cot", nudge=nudge, mail_tag=mail_tag, llm_judge=judge_model), 
                        model=agent_model,
                        log_dir=log_dir,  
                        display="conversation",
                        temperature=temp,
                        seed=seed
        )
            

if __name__ == "__main__":
    main()
