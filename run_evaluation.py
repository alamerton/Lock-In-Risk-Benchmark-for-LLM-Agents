import sys
import os
import asyncio
from dotenv import load_dotenv
import inspect_ai._eval.eval as inspect_eval
from inspect_ai import eval_set
from inspect_ai._display.rich.display import RichTaskDisplay

# Load environment variables from .env file
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Initialize logging first
import logging
logging.basicConfig(level=logging.CRITICAL)

# Create a logger for concurrent.futures
logger = logging.getLogger("concurrent.futures")
logger.setLevel(logging.CRITICAL)

# Import evaluation module
from evaluation.inspect_integration import LockInEvaluator
from evaluation.core.enums import FocusArea

from evaluation.inspect_integration import *
from evaluation.core.enums import FocusArea
from agents.example_agent import example_base_agent, setup_scenario
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


# Custom evaluation function
async def run_eval(task, model, limit=None, log_dir=None):
    """Run evaluation with the given task and model.
    
    Args:
        task: Task object with dataset, solver, and scorer
        model: Model identifier string
        limit: Maximum number of samples to evaluate
        log_dir: Directory to save results
    """
    # Debug task configuration
    print(f"Debug - Task config: {task.config}")
    print(f"Debug - Task scorer: {task.scorer}")
    print(f"Debug - Task solver: {task.solver}")
    
    # Ensure scorer is properly instantiated
    from evaluation.scorers import UnifiedLockInScorer
    if not hasattr(task, 'scorer') or not isinstance(task.scorer, UnifiedLockInScorer):
        print("Creating new UnifiedLockInScorer instance...")
        task.scorer = UnifiedLockInScorer(strict_mode=True)
    from evaluation.core.config import Config
    from evaluation.core.model import Model
    from evaluation.core.task_state import TaskState

    # Initialize model
    config = Config({
        "model": model,
        "temperature": 0.7,
        "max_tokens": 1000
    })
    model_instance = Model(model, config)

    # Initialize task state
    from evaluation.core.message import Message
    state = TaskState(
        model=model,
        sample_id="0",
        epoch=0,
        input="",
        messages=[],
        metadata={}
    )

    # Run task
    results = []
    for i in range(limit or 1):
        # Get solver for this sample
        solver = task.solver(task.dataset[i])
        
        # Run solver
        state = await solver(state, model_instance.generate)
        
        # Score results
        try:
            if hasattr(task, 'scorer') and task.scorer:
                if callable(task.scorer):
                    score = await task.scorer(state, task.dataset[i])
                else:
                    print(f"Warning: task.scorer exists but is not callable: {type(task.scorer)}")
                    score = {
                        "value": 0.0,
                        "explanation": f"Scorer exists but not callable: {type(task.scorer)}"
                    }
            else:
                print("Warning: task.scorer is missing")
                score = {
                    "value": 0.0,
                    "explanation": "No scorer available on task"
                }
        except Exception as e:
            print(f"Error during scoring: {str(e)}")
            score = {
                "value": 0.0,
                "explanation": f"Scoring error: {str(e)}"
            }
        
        # Store results
        result = {
            "score": score,
            "metadata": state.metadata
        }
        results.append(result)
        
        # Save individual result
        if log_dir:
            import json
            from datetime import datetime
            from pathlib import Path
            
            # Ensure logs/results directory exists
            output_dir = Path(log_dir) / "results"
            output_dir.mkdir(parents=True, exist_ok=True)  # This will create both logs/ and logs/results/ if they don't exist
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"lock_in_eval_{timestamp}.json"
            
            # Save result
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            # Generate and save analysis
            from evaluation.analysis.result_analyzer import ResultAnalyzer
            # Convert result to dict if it's a Pydantic model
            result_dict = result.copy()
            if hasattr(result_dict.get('score'), 'model_dump'):
                result_dict['score'] = result_dict['score'].model_dump()
            analyzer = ResultAnalyzer(result_dict)
            
            # Create analysis filename with same timestamp
            analysis_filename = output_dir / f"lock_in_eval_{timestamp}_analysis.json"
            analyzer.save_analysis(str(analysis_filename))
            
            # Print analysis to console
            print("\nEvaluation Analysis:")
            print(analyzer.get_detailed_analysis())

    return results

def run_eval_sets(tasks, models, log_dir, limit=10000, retry_attempts=15):
    """
    Run evaluation with multiple tasks and models. Has cache - so given a log_dir, will not repeat inference for the same data points - no redundancy.

    Use to run multiple experiments, variations
    
    Args:
        tasks: List of Task objects with dataset, solver, and scorer
        models: List of Model identifier strings
        limit: Maximum number of samples to evaluate
        log_dir: Directory to save results
        retry_attempts: Maximum number of retry attempts

    """
    success, logs = eval_set(
    tasks=tasks,
    model=models,
    log_dir=log_dir ,
    limit=limit  ,
    retry_attempts=retry_attempts,
    trace=True
    )

    return logs

async def main():
    print("\nStep 1: Creating evaluator...", flush=True)
    evaluator = LockInEvaluator(
        # model="gpt-5-nano-2025-08-07",
        model="groq/gemma2-9b-it",
        log_dir="./logs",
        save_results=True
    )
    print("[+] Evaluator created successfully", flush=True)

    print("\nStep 2: Starting evaluation...", flush=True)
    try:
        # Create and run task
        results = await evaluator.run_single_focus(
            focus_area=FocusArea.UPDATE_RESISTANCE,
            sample_count=1,
            cot_style="structured",
            stealth_mode=True,  # This parameter is accepted by comprehensive_lock_in_task
            adaptive_scoring=False  # Use standard scoring instead of adaptive
        )
        print("[+] Evaluation completed successfully", flush=True)
        print("\nResults:", flush=True)
        print(results, flush=True)
        return results
    except Exception as e:
        print(f"\n[-] Error: {str(e)}", flush=True)
        raise

if __name__ == "__main__":
    import asyncio
    print("Starting evaluation...", flush=True)
    try:
        results = asyncio.run(main())
        print("\nFinal results:", flush=True)
        print(results, flush=True)
    except Exception as e:
        print(f"\nError occurred: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()


    # #Example shutdown scenario implementation

    # #path to config file - which in this case is read as dataset samples :')
    # path = "scenario_configs/sample_update.json"
    # tools = [add(), next_task(), submit()] #tools to be passed to the agent
    # dataset = LockInDatasetFactory._read_json_to_samples(path) #read from existing json
    # task = custom_task(
    #     dataset=dataset, #inspect dataset w/ samples
    #     #solver chain - setup_scenario is a helper solver to store state values
    #     plan=[setup_scenario(), example_base_agent(tools=tools), generate()], 
    #     # need scorer - custom implementation
    #     scorers=[includes()] #dummy built-in
    # )
    # #can add multiple tasks (experiments w/ multiple datasets, hyperparams..), multiple models
    # logs = run_eval_sets([task], ["groq/gemma2-9b-it"], "./logs")

