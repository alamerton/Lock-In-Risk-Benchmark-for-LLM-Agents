#!/usr/bin/env python3
"""
This script runs the complete evaluation workflow:
1. Load evaluation from tests directory
2. Create Docker sandbox with files
3. Run agent with prompt
4. Interrupt with probe
5. Log all outputs to timestamped directory
6. Run judge LLM evaluation against rubric
7. Save final results

Usage:
    python run.py <eval_name> [--agent-model MODEL] [--judge-model MODEL]

Example:
    python run.py INSERT_EVAL_FOLDER_NAME
    python run.py INSERT_EVAL_FOLDER_NAME --agent-model anthropic/claude-opus-4-1-20250805
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from sandbox_environment import SandboxEnvironment
from judge_evaluator import JudgeEvaluator


def run_full_evaluation(
    eval_name: str,
    agent_model: str = "anthropic/claude-opus-4-1-20250805",
    judge_model: str = "anthropic/claude-opus-4-1-20250805",
    skip_judge: bool = False
) -> dict:
    """
    Run the complete evaluation workflow.

    Args:
        eval_name: Name of evaluation directory (e.g., "INSERT_EVAL_FOLDER_NAME")
        agent_model: Model to use for the agent being evaluated
        judge_model: Model to use for judge evaluation
        skip_judge: If True, skip judge evaluation step

    Returns:
        Dictionary with results from both stages
    """
    print("\n" + "=" * 80)
    print("AISI INSPECT SANDBOX EVALUATION")
    print("=" * 80)
    print(f"\nEvaluation: {eval_name}")
    print(f"Agent Model: {agent_model}")
    print(f"Judge Model: {judge_model}")
    print("\n" + "=" * 80 + "\n")

    # Stage 1: Run sandbox evaluation
    print("STAGE 1: Running agent in sandbox")
    print("-" * 80)

    eval_dir = Path("tests") / eval_name
    if not eval_dir.exists():
        print(f"Error: Evaluation directory not found: {eval_dir}")
        print(f"\nAvailable evaluations:")
        tests_dir = Path("tests")
        if tests_dir.exists():
            for d in tests_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        sys.exit(1)

    # Create sandbox environment
    sandbox_env = SandboxEnvironment(eval_dir)

    # Run evaluation (synchronous - eval manages its own event loop)
    try:
        sandbox_results = sandbox_env.run_evaluation(agent_model)
        output_dir = sandbox_env.get_output_dir()
        print(f"\n✓ Stage 1 complete. Outputs saved to: {output_dir}\n")
    except Exception as e:
        print(f"\n✗ Stage 1 failed: {str(e)}\n")
        raise

    # Stage 2: Judge evaluation
    if skip_judge:
        print("\nSkipping judge evaluation (--skip-judge flag set)")
        return {
            "sandbox_results": sandbox_results,
            "judge_results": None
        }

    print("\nSTAGE 2: Judge LLM evaluation")

    try:
        judge = JudgeEvaluator(judge_model=judge_model)
        # Judge evaluation is async, so run it in event loop
        # Pass eval_type from sandbox results
        eval_type = sandbox_results.get("eval_type")
        judge_results = asyncio.run(judge.evaluate(output_dir, eval_type=eval_type))
        print(f"\n✓ Stage 2 complete\n")
    except Exception as e:
        print(f"\n✗ Stage 2 failed: {str(e)}\n")
        raise

    # Final summary
    print("\n")
    print("EVALUATION COMPLETE")
    print(f"\nEvaluation: {eval_name}")
    print(f"Output Directory: {output_dir}")
    print(f"\nResults:")
    print(f"  - Sandbox outputs: {output_dir / 'sandbox_outputs'}")
    print(f"  - Conversation log: {output_dir / 'conversation_log.json'}")
    print(f"  - Transcript: {output_dir / 'transcript.txt'}")
    print(f"  - Judge evaluation: {Path('evaluations')}")

    if judge_results and "assessment" in judge_results:
        print(f"\nAssessment: {judge_results['assessment']}")

        if "score" in judge_results and "total" in judge_results["score"]:
            print(f"Total Score: {judge_results['score']['total']}")

    return {
        "sandbox_results": sandbox_results,
        "judge_results": judge_results
    }


def list_evaluations():
    """List available evaluations."""
    print("\nAvailable Evaluations:")
    print("-" * 80)

    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("No tests directory found")
        return

    eval_dirs = [d for d in tests_dir.iterdir() if d.is_dir()]

    if not eval_dirs:
        print("No evaluations found in tests/")
        return

    for eval_dir in sorted(eval_dirs):
        print(f"\n{eval_dir.name}")

        # Show what files are present
        files = []
        if (eval_dir / "prompt.txt").exists():
            files.append("prompt")
        if (eval_dir / "probe.txt").exists():
            files.append("probe")
        if (eval_dir / "eval_type.txt").exists():
            eval_type = (eval_dir / "eval_type.txt").read_text().strip()
            files.append(f"eval_type={eval_type}")

        if files:
            print(f"  Files: {', '.join(files)}")

        # Count data files
        data_files = [
            f.name for f in eval_dir.iterdir()
            if f.is_file() and f.name not in
            ["prompt.txt", "probe.txt", "eval_type.txt"]
        ]
        if data_files:
            print(f"  Data files: {', '.join(data_files)}")

    print()


def list_rubrics():
    """List available rubrics."""
    print("\nAvailable Rubrics:")
    print("-" * 80)

    rubrics_dir = Path("rubrics")
    if not rubrics_dir.exists():
        print("No rubrics directory found")
        return

    rubric_files = list(rubrics_dir.glob("*.txt"))

    if not rubric_files:
        print("No rubrics found in rubrics/")
        return

    for rubric_file in sorted(rubric_files):
        print(f"\n{rubric_file.stem}")
        # Show first line as description
        first_line = rubric_file.read_text().split('\n')[0]
        if first_line:
            print(f"  {first_line[:70]}...")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect sandbox evals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run evaluation with default models
        python run.py INSERT_EVAL_FOLDER_NAME

        # Use specific models
        python run.py INSERT_EVAL_FOLDER_NAME --agent-model anthropic/claude-opus-4-1-20250805

        # Run without judge evaluation
        python run.py INSERT_EVAL_FOLDER_NAME --skip-judge

        # List available evaluations and rubrics
        python run.py --list
        """
    )

    parser.add_argument(
        "eval_name",
        nargs="?",
        help="Name of evaluation to run (e.g., INSERT_EVAL_FOLDER_NAME)"
    )

    parser.add_argument(
        "--agent-model",
        default="anthropic/claude-opus-4-1-20250805",
        help="Model to use for agent being evaluated (default: anthropic/claude-opus-4-1-20250805)"
    )

    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-opus-4-1-20250805",
        help="Model to use for judge evaluation (default: anthropic/claude-opus-4-1-20250805)"
    )

    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip judge evaluation stage"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available evaluations and rubrics"
    )

    args = parser.parse_args()

    # Handle list mode
    if args.list:
        list_evaluations()
        list_rubrics()
        return

    # Require eval_name if not listing
    if not args.eval_name:
        parser.print_help()
        print("\nError: eval_name is required (or use --list to see available evaluations)")
        sys.exit(1)

    # Run evaluation
    try:
        results = run_full_evaluation(
            eval_name=args.eval_name,
            agent_model=args.agent_model,
            judge_model=args.judge_model,
            skip_judge=args.skip_judge
        )
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nEvaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
