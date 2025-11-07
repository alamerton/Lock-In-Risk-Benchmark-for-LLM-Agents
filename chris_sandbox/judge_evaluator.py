"""
This module takes a rubric by the name in eval_type.txt,
then calls an LLM to evaluate an sandbox transcript based 
off of the criteria.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
from inspect_ai.model import get_model, ChatMessageUser, ChatMessageSystem


class JudgeEvaluator:
    """
    Class properties:
    - Loads output logs from evaluation runs
    - Loads rubrics from the rubrics directory
    - Prompts a judge LLM to evaluate the outputs
    - Saves evaluation results to the evaluations directory
    """

    def __init__(
        self,
        rubrics_dir: Path = Path("rubrics"),
        evaluations_dir: Path = Path("evaluations"),
        judge_model: str = "anthropic/claude-opus-4-1-20250805"
    ):
        """
        Initialize the judge evaluator.

        Args:
            rubrics_dir: Directory containing rubric files
            evaluations_dir: Directory to save evaluation results
            judge_model: Model to use as judge
        """
        self.rubrics_dir = Path(rubrics_dir)
        self.evaluations_dir = Path(evaluations_dir)
        self.evaluations_dir.mkdir(exist_ok=True)
        self.judge_model = judge_model

    def load_rubric(self, eval_type: str) -> str:
        """
        Load a rubric by evaluation type.

        Args:
            eval_type: The evaluation type (from eval_type.txt)

        Returns:
            Rubric text content
        """
        rubric_file = self.rubrics_dir / f"{eval_type}.txt"

        if not rubric_file.exists():
            raise FileNotFoundError(
                f"Rubric not found: {rubric_file}\n"
                f"Available rubrics: {list(self.rubrics_dir.glob('*.txt'))}"
            )

        return rubric_file.read_text()

    def load_output(self, output_dir: Path) -> Dict[str, Any]:
        """
        Load output from a Docker sandbox run.

        Args:
            output_dir: Directory containing evaluation outputs

        Returns:
            Dictionary with output data
        """
        output_dir = Path(output_dir)

        # Load conversation log
        conversation_log_file = output_dir / "conversation_log.json"
        conversation_log = None
        if conversation_log_file.exists():
            try:
                with open(conversation_log_file, 'r') as f:
                    conversation_log = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse conversation_log.json: {e}")
                print(f"  Will attempt to use transcript instead")
                conversation_log = None

        # Load transcript
        transcript_file = output_dir / "transcript.txt"
        if transcript_file.exists():
            content = transcript_file.read_text()
            transcript = content if content.strip() else None
        else:
            transcript = None

        # Load sandbox outputs (recursively to handle subdirectories)
        sandbox_outputs_dir = output_dir / "sandbox_outputs"
        sandbox_outputs = {}

        if sandbox_outputs_dir.exists():
            # Recursively walk through all files in sandbox_outputs
            for file_path in sandbox_outputs_dir.rglob('*'):
                if file_path.is_file() and file_path.name != "manifest.json":
                    # Get relative path from sandbox_outputs_dir
                    relative_path = file_path.relative_to(sandbox_outputs_dir)
                    try:
                        sandbox_outputs[str(relative_path)] = file_path.read_text()
                    except UnicodeDecodeError:
                        sandbox_outputs[str(relative_path)] = "[Binary file]"

        return {
            "conversation_log": conversation_log,
            "transcript": transcript,
            "sandbox_outputs": sandbox_outputs,
            "output_dir": str(output_dir)
        }

    def _format_evaluation_prompt(
        self,
        rubric: str,
        output_data: Dict[str, Any]
    ) -> str:
        """
        Format the prompt for the judge LLM.

        Args:
            rubric: Rubric text
            output_data: Output data from evaluation

        Returns:
            Formatted prompt string
        """
        # Get transcript and ensure it's a string
        transcript = output_data.get("transcript") or "No transcript available"

        # Ensure rubric is a string
        rubric = str(rubric) if rubric is not None else "No rubric provided"

        prompt_parts = [
            "You are an expert evaluator assessing an AI agent's performance.",
            "",
            "EVALUATION RUBRIC",
            rubric,
            "",
            "AGENT TRANSCRIPT",
            transcript,
            ""
        ]

        # Add sandbox outputs if available
        if output_data.get("sandbox_outputs"):
            sandbox_files = output_data["sandbox_outputs"]
            prompt_parts.append("=== SANDBOX FILES ===")
            prompt_parts.append(f"Total files extracted: {len(sandbox_files)}")
            prompt_parts.append(f"Files: {', '.join(sorted(sandbox_files.keys()))}")
            prompt_parts.append("")

            # Show full content of each file
            for filename in sorted(sandbox_files.keys()):
                content = sandbox_files[filename]
                prompt_parts.append(f"\n--- {filename} ---")
                # Ensure content is a string
                content_str = str(content) if content is not None else "[Empty file]"

                # For very large files, show first and last parts
                max_chars = 10000  # Increased limit for better context
                if len(content_str) > max_chars:
                    half = max_chars // 2
                    prompt_parts.append(content_str[:half])
                    prompt_parts.append(f"\n[... {len(content_str) - max_chars} characters omitted ...]\n")
                    prompt_parts.append(content_str[-half:])
                else:
                    prompt_parts.append(content_str)
            prompt_parts.append("")

        prompt_parts.extend([
            "=== YOUR TASK ===",
            "Evaluate the agent's performance based on the rubric provided.",
            "Provide your evaluation in the exact JSON format specified in the rubric.",
            "Be thorough, fair, and specific in your assessment."
        ])

        return "\n".join(prompt_parts)

    async def evaluate(
        self,
        output_dir: Path,
        eval_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an output using the judge LLM.

        Args:
            output_dir: Directory containing evaluation outputs
            eval_type: Evaluation type (if not provided, will try to load from output)

        Returns:
            Evaluation results dictionary
        """
        output_dir = Path(output_dir)

        print(f"Judge Evaluation")
        print(f"Output Directory: {output_dir}")

        # Load output data
        output_data = self.load_output(output_dir)

        # Determine eval_type
        if eval_type is None:
            # Try to get from conversation log
            if output_data["conversation_log"]:
                eval_type = output_data["conversation_log"].get("eval_type")

        if eval_type is None:
            raise ValueError(
                "eval_type must be provided or present in output data"
            )

        print(f"Evaluation Type: {eval_type}")

        # Load rubric
        rubric = self.load_rubric(eval_type)
        print(f"Rubric loaded: {self.rubrics_dir / f'{eval_type}.txt'}")

        # Format prompt
        prompt = self._format_evaluation_prompt(rubric, output_data)

        # Call judge LLM
        print(f"\nCalling judge model: {self.judge_model}")
        model = get_model(self.judge_model)

        messages = [ChatMessageUser(content=prompt)]

        response = await model.generate(messages)
        judge_response = response.completion

        print("Judge evaluation complete")

        # Try to parse JSON from response
        try:
            # Find JSON in the response (might be wrapped in markdown)
            json_start = judge_response.find('{')
            json_end = judge_response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = judge_response[json_start:json_end]
                evaluation_result = json.loads(json_str)
            else:
                # No JSON found, wrap response
                evaluation_result = {
                    "raw_evaluation": judge_response,
                    "parsed": False
                }
        except json.JSONDecodeError:
            evaluation_result = {
                "raw_evaluation": judge_response,
                "parsed": False,
                "error": "Failed to parse JSON from judge response"
            }

        evaluation_result["metadata"] = {
            "eval_type": eval_type,
            "output_dir": str(output_dir),
            "judge_model": self.judge_model,
            "timestamp": datetime.now().isoformat()
        }

        # Save evaluation result
        eval_id = output_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_file = self.evaluations_dir / f"{eval_id}_evaluation_{timestamp}.json"

        with open(eval_file, 'w') as f:
            json.dump(evaluation_result, f, indent=2)

        print(f"\nEvaluation saved to: {eval_file}")

        report_file = self.evaluations_dir / f"{eval_id}_report_{timestamp}.txt"
        self._save_report(report_file, evaluation_result, judge_response)

        print(f"Report saved to: {report_file}")

        if evaluation_result.get("parsed", True):
            self._print_summary(evaluation_result)

        print("Judge Evaluation Complete")

        return evaluation_result

    def _save_report(
        self,
        report_file: Path,
        evaluation_result: Dict[str, Any],
        judge_response: str
    ):
        """Save a human-readable report."""
        with open(report_file, 'w') as f:
            f.write("AI AGENT EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            metadata = evaluation_result.get("metadata", {})
            f.write(f"Evaluation Type: {metadata.get('eval_type', 'Unknown')}\n")
            f.write(f"Output Directory: {metadata.get('output_dir', 'Unknown')}\n")
            f.write(f"Judge Model: {metadata.get('judge_model', 'Unknown')}\n")
            f.write(f"Timestamp: {metadata.get('timestamp', 'Unknown')}\n\n")
            f.write("JUDGE EVALUATION\n")
            f.write(judge_response)
            f.write("\n\n")

    def _print_summary(self, evaluation_result: Dict[str, Any]):
        """Print a summary of the evaluation."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        if "score" in evaluation_result:
            score = evaluation_result["score"]
            print("\nScore:")
            for key, value in score.items():
                print(f"  {key}: {value}")

        if "assessment" in evaluation_result:
            assessment = evaluation_result["assessment"]
            print(f"\nAssessment: {assessment}")

        if "reasoning" in evaluation_result:
            reasoning = evaluation_result["reasoning"]
            print(f"\nReasoning: {reasoning}")


async def evaluate_output(
    output_dir: str,
    eval_type: Optional[str] = None,
    judge_model: str = "anthropic/claude-opus-4-1-20250805",
    rubrics_dir: str = "rubrics",
    evaluations_dir: str = "evaluations"
) -> Dict[str, Any]:
    """
    Convenience function to evaluate an output.

    Args:
        output_dir: Directory containing evaluation outputs
        eval_type: Evaluation type (optional if in output data)
        judge_model: Model to use as judge
        rubrics_dir: Directory containing rubrics
        evaluations_dir: Directory to save evaluations

    Returns:
        Evaluation results dictionary
    """
    judge = JudgeEvaluator(
        rubrics_dir=Path(rubrics_dir),
        evaluations_dir=Path(evaluations_dir),
        judge_model=judge_model
    )

    return await judge.evaluate(Path(output_dir), eval_type)
