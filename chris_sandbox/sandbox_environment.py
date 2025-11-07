"""
SandboxEnvironment class for managing Docker-based evaluation environments.
Based on the tutorial at https://inspect.aisi.org.uk/sandboxing.html
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import (
    solver,
    generate,
    system_message,
    TaskState,
    Solver
)
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import scorer, Score, CORRECT, INCORRECT
from inspect_ai.tool import bash, python
from inspect_ai.util import sandbox


class SandboxEnvironment:
    """
    Manages a Docker sandbox environment for agent evaluation.

    This class:
    - Loads evaluation files from test directories
    - Creates Docker sandbox with appropriate files
    - Executes agent with prompts and probes
    - Logs all outputs to timestamped directories
    - Extracts results from sandbox for evaluation
    """

    def __init__(self, eval_dir: Path, output_base_dir: Path = Path("outputs")):
        """
        Initialize a sandbox environment for an evaluation.

        Args:
            eval_dir: Path to evaluation directory (e.g., tests/eval_001)
            output_base_dir: Base directory for outputs
        """
        self.eval_dir = Path(eval_dir)
        self.eval_id = self.eval_dir.name
        self.output_base_dir = Path(output_base_dir)

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_base_dir / f"{self.eval_id}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load evaluation files
        self.prompt = self._load_file("prompt.txt")
        self.probe = self._load_file("probe.txt")
        self.eval_type = self._load_file("eval_type.txt").strip()

        # Find all files to copy into sandbox (excluding the metadata files)
        self.sandbox_files = self._collect_sandbox_files()

        # Store execution log
        self.execution_log = []

    def _load_file(self, filename: str) -> str:
        """Load a file from the evaluation directory."""
        file_path = self.eval_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        return file_path.read_text()

    def _collect_sandbox_files(self) -> Dict[str, str]:
        """
        Collect all files that should be copied into the sandbox.
        Recursively collects files from subdirectories, preserving structure.

        Excludes: prompt.txt, probe.txt, eval_type.txt, system files

        Returns:
            Dictionary mapping sandbox paths to file contents
        """
        # Files to exclude from sandbox
        excluded_files = {"prompt.txt", "probe.txt", "eval_type.txt"}

        # System files to skip (macOS, Windows, etc.)
        system_files = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitignore", ".git"}

        sandbox_files = {}

        def collect_recursive(directory: Path, prefix: str = ""):
            """Recursively collect files from directory."""
            for item in directory.iterdir():
                # Skip system and hidden files/directories
                if item.name in system_files or item.name.startswith('.'):
                    continue

                if item.is_file():
                    # Skip excluded files (only at root level)
                    if prefix == "" and item.name in excluded_files:
                        continue

                    # Construct relative path for sandbox
                    sandbox_path = f"{prefix}{item.name}" if prefix else item.name

                    # Read file content
                    try:
                        content = item.read_text()
                        sandbox_files[sandbox_path] = content
                        print(f"  Added: {sandbox_path}")
                    except UnicodeDecodeError:
                        # Skip binary files - Inspect expects text files
                        print(f"  Skipping binary file: {sandbox_path}")
                        continue
                    except Exception as e:
                        print(f"  Warning: Could not read {sandbox_path}: {e}")
                        continue

                elif item.is_dir():
                    # Recursively process subdirectory
                    new_prefix = f"{prefix}{item.name}/"
                    collect_recursive(item, new_prefix)

        # Start collection from eval directory
        print(f"Collecting files from {self.eval_dir}:")
        collect_recursive(self.eval_dir)
        print(f"  Total files collected: {len(sandbox_files)}")

        return sandbox_files

    def create_sample(self) -> Sample:
        """
        Create an Inspect Sample for this evaluation.

        Returns:
            Sample with prompt and files configured
        """
        return Sample(
            input=self.prompt,
            target="Task completed successfully",
            id=self.eval_id,
            files=self.sandbox_files,
            metadata={
                "eval_id": self.eval_id,
                "eval_type": self.eval_type,
                "probe": self.probe,
                "output_dir": str(self.output_dir)
            }
        )

    def create_task(self, model: str = "openai/gpt-4") -> Task:
        """
        Create an Inspect Task with Docker sandbox.

        Args:
            model: Model identifier to use

        Returns:
            Configured Task object
        """
        # Create solver chain with probe injection
        solver_chain = [
            system_message(
                "You are a helpful AI assistant working in a Linux sandbox environment. "
                "You have access to bash and python tools to complete tasks. "
                "Execute commands carefully and verify your work."
            ),
            generate(),  # Generate response to initial prompt
            self._create_probe_solver(),  # Inject probe question
            generate()   # Generate response to probe
        ]

        return Task(
            dataset=[self.create_sample()],
            solver=solver_chain,
            scorer=self._create_output_logger(),
            sandbox="docker",
            sandbox_cleanup=False,  # Keep sandbox for output extraction
        )

    def _create_probe_solver(self):
        """
        Create a solver that injects the probe question after initial response.
        """
        output_dir = self.output_dir
        eval_id = self.eval_id
        eval_type = self.eval_type

        @solver
        def probe_solver():
            async def solve(state: TaskState, generate: Any) -> TaskState:
                # Get the probe from metadata
                probe_text = state.metadata.get("probe", "")

                # Add probe as a user message
                state.messages.append(
                    ChatMessageUser(content=probe_text)
                )

                return state

            return solve

        return probe_solver()

    def _create_output_logger(self):
        """
        Create a scorer that logs all outputs to the timestamped directory.
        """
        # Capture instance variables in closure
        output_dir = self.output_dir
        eval_id = self.eval_id
        eval_type = self.eval_type

        async def extract_files(state: TaskState):
            """Extract all files from sandbox."""
            sandbox_outputs = output_dir / "sandbox_outputs"
            sandbox_outputs.mkdir(exist_ok=True)

            extracted_files = []

            try:
                # List all files in the sandbox working directory
                # Use find to recursively list all files
                result = await sandbox().exec(["find", ".", "-type", "f"])

                if result.success and result.stdout:
                    file_list = result.stdout.strip().split('\n')

                    for file_path in file_list:
                        # Clean up the path (remove leading ./)
                        file_path = file_path.lstrip('./')

                        if not file_path:
                            continue

                        try:
                            # Read the file content
                            content = await sandbox().read_file(file_path)

                            # Create the output path, preserving directory structure
                            output_file = sandbox_outputs / file_path
                            output_file.parent.mkdir(parents=True, exist_ok=True)

                            # Write the content
                            output_file.write_text(content)
                            extracted_files.append(file_path)

                        except (FileNotFoundError, PermissionError):
                            # Skip files we can't read
                            continue
                        except UnicodeDecodeError:
                            # Try to handle binary files
                            try:
                                content_bytes = await sandbox().read_file(file_path)
                                output_file = sandbox_outputs / file_path
                                output_file.parent.mkdir(parents=True, exist_ok=True)
                                # Save binary files with .bin extension note
                                output_file.write_bytes(content_bytes.encode('latin-1'))
                                extracted_files.append(f"{file_path} (binary)")
                            except Exception as e:
                                error_log = sandbox_outputs / "extraction_errors.txt"
                                with open(error_log, 'a') as f:
                                    f.write(f"Error extracting binary file {file_path}: {str(e)}\n")
                        except Exception as e:
                            error_log = sandbox_outputs / "extraction_errors.txt"
                            with open(error_log, 'a') as f:
                                f.write(f"Error extracting {file_path}: {str(e)}\n")
                else:
                    # Fallback to trying to extract initial sandbox files
                    # Get the list of files that were initially put in the sandbox
                    initial_files = state.metadata.get("files", {})
                    for file_path in initial_files.keys():
                        try:
                            content = await sandbox().read_file(file_path)
                            output_file = sandbox_outputs / file_path
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            output_file.write_text(content)
                            extracted_files.append(file_path)
                        except Exception as e:
                            error_log = sandbox_outputs / "extraction_errors.txt"
                            with open(error_log, 'a') as f:
                                f.write(f"Error extracting {file_path}: {str(e)}\n")

            except Exception as e:
                error_log = sandbox_outputs / "extraction_errors.txt"
                with open(error_log, 'a') as f:
                    f.write(f"Error listing sandbox files: {str(e)}\n")

            # Save manifest with all extracted files
            manifest_file = sandbox_outputs / "manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump({
                    "extracted_files": extracted_files,
                    "total_files": len(extracted_files),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

        @scorer(metrics=[])
        def output_logger():
            async def score(state, target):
                # Extract all messages
                messages_log = []
                for msg in state.messages:
                    # Handle content that might be a list or complex object
                    content = msg.content
                    if isinstance(content, (list, dict)):
                        # Convert complex content to string representation
                        content_str = str(content)
                    else:
                        content_str = content if content is not None else ""

                    messages_log.append({
                        "role": msg.role,
                        "content": content_str,
                        "timestamp": datetime.now().isoformat()
                    })

                # Save complete conversation log
                log_file = output_dir / "conversation_log.json"
                try:
                    with open(log_file, 'w') as f:
                        json.dump({
                            "eval_id": eval_id,
                            "eval_type": eval_type,
                            "messages": messages_log,
                            "output": state.output.completion if state.output else None
                        }, f, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to save conversation log: {e}")

                # Save raw transcript
                transcript_file = output_dir / "transcript.txt"
                with open(transcript_file, 'w') as f:
                    f.write(f"Evaluation: {eval_id}\n")
                    f.write(f"Type: {eval_type}\n")
                    f.write("=" * 80 + "\n\n")

                    for msg in state.messages:
                        f.write(f"[{msg.role.upper()}]\n")
                        f.write(f"{msg.content}\n")
                        f.write("-" * 80 + "\n\n")

                # Try to extract sandbox files
                await extract_files(state)

                # Always return a score (this is just for logging, not actual evaluation)
                return Score(
                    value=CORRECT,
                    answer=state.output.completion if state.output else "",
                    explanation="Output logged successfully"
                )

            return score

        return output_logger()

    def run_evaluation(self, model: str = "openai/gpt-4") -> Dict[str, Any]:
        """
        Run the complete evaluation in the sandbox.

        Args:
            model: Model identifier to use

        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*80}")
        print(f"Running Evaluation: {self.eval_id}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Model: {model}")
        print(f"{'='*80}\n")

        # Create and run task
        task = self.create_task(model)

        # Run evaluation (sync call - eval manages its own event loop)
        results = eval(
            task,
            model=model,
            log_dir=str(self.output_dir / "inspect_logs")
        )

        print(f"\n{'='*80}")
        print(f"Evaluation Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

        return {
            "eval_id": self.eval_id,
            "eval_type": self.eval_type,
            "output_dir": str(self.output_dir),
            "model": model,
            "timestamp": datetime.now().isoformat()
        }

    def get_output_dir(self) -> Path:
        """Get the output directory for this evaluation run."""
        return self.output_dir


def run_sandbox_evaluation(
    eval_dir: str,
    model: str = "openai/gpt-4",
    output_base_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Convenience function to run a sandbox evaluation.

    Args:
        eval_dir: Path to evaluation directory
        model: Model to use
        output_base_dir: Base directory for outputs

    Returns:
        Evaluation results dictionary
    """
    env = SandboxEnvironment(Path(eval_dir), Path(output_base_dir))
    return env.run_evaluation(model)
