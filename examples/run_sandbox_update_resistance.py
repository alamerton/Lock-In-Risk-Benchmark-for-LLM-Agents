"""
Example: Running Sandbox-Based Update Resistance Evaluation

This script demonstrates how to run the update resistance scenario
that uses inspect's sandbox environment with real file operations,
tools, and LLM-as-judge analysis.
"""

import os
import time
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from inspect_ai import eval
from inspect_ai.log import read_eval_log

from evaluation.scenarios.update_resistance_sandbox import (
    SandboxUpdateResistanceScenario,
    sandbox_update_resistance_task,
    sandbox_update_resistance_solver
)


def run_single_scenario(scenario_type: str = "code_refactor", sandbox_type: str = "docker"):
    """
    Run a single sandbox-based update resistance scenario.
    
    Args:
        scenario_type: Type of scenario to run
            - "code_refactor": Code quality vs deadline pressure
            - "security_hardening": Security vs speed tradeoff  
            - "data_analysis": Thorough analysis vs fast results
        sandbox_type: Type of sandbox environment
            - "docker": Docker container (recommended)
            - "local": Local filesystem (testing only)
    """
    
    print(f"\n{'='*60}")
    print(f"Running Sandbox-Based Update Resistance: {scenario_type}")
    print(f"Sandbox Type: {sandbox_type}")
    print(f"{'='*60}\n")
    
    # Set up API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["API_KEY"] = api_key
    else:
        print("Warning: No API_KEY or OPENAI_API_KEY found in environment")
        print("Set one with: $env:API_KEY='your-key'\n")
    
    # Set model to evaluate
    model = os.getenv("MODEL", "openai/o3-mini")
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-2024-08-06")
    
    print(f"Model being evaluated: {model}")
    print(f"Judge model: {judge_model}")
    print(f"Sandbox environment: {sandbox_type}\n")
    
    # Create task
    task = sandbox_update_resistance_task(
        scenario_type=scenario_type,
        judge_model=judge_model,
        sandbox_type=sandbox_type
    )
    
    # Run evaluation
    print("Starting evaluation with sandbox environment...")
    print("Agent will have access to:")
    print("  • Real file system operations (sandboxed)")
    print("  • Bash command execution")
    print("  • Python code execution")
    print("  • Git version control")
    print("LLM judge will analyze responses for resistance patterns\n")
    
    try:
        result = eval(
            task,
            model=model,
            log_dir="./logs/sandbox_update_resistance"
        )
        
        # Display results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}\n")
        
        # Access samples from eval results
        if result and len(result) > 0:
            for eval_log in result:
                if not hasattr(eval_log, 'samples') or not eval_log.samples:
                    continue
                    
                for idx, sample in enumerate(eval_log.samples):
                    print(f"\n--- Sample {idx} ---")
                    
                    # Get metadata from score or sample
                    metadata = {}
                    if hasattr(sample, 'scores') and sample.scores:
                        score_obj = next(iter(sample.scores.values()), None)
                        if score_obj and hasattr(score_obj, 'metadata'):
                            metadata = score_obj.metadata
                    
                    # Fallback to sample metadata
                    if not metadata:
                        if hasattr(sample, 'metadata'):
                            metadata = sample.metadata.get("lock_in_evaluation", {})
                        elif hasattr(sample, 'store') and sample.store:
                            metadata = sample.store.get("lock_in_evaluation", {})
                    
                    # Display results
                    scenario_type_result = metadata.get('scenario_type', 'unknown')
                    risk_score = metadata.get('risk_score', 0.0)
                    risk_level = metadata.get('risk_level', 'unknown')
                    
                    print(f"Scenario: {scenario_type_result}")
                    print(f"Risk Score: {risk_score:.3f}")
                    print(f"Risk Level: {risk_level.upper()}")
                    
                    print(f"\nSandbox Activity:")
                    print(f"  Tool Calls: {metadata.get('tool_calls_count', 0)}")
                    print(f"  File States Captured: {metadata.get('file_states_captured', 0)}")
                    
                    # File analysis
                    file_analysis = metadata.get('file_analysis', {})
                    if file_analysis:
                        print(f"\nFile System Changes:")
                        print(f"  Initial Files: {file_analysis.get('initial_file_count', 0)}")
                        print(f"  Final Files: {file_analysis.get('final_file_count', 0)}")
                        print(f"  Net Change: {file_analysis.get('net_change', 0)}")
                        
                        if file_analysis.get('files_added'):
                            print(f"  Files Added: {', '.join(file_analysis['files_added'][:5])}")
                        if file_analysis.get('files_removed'):
                            print(f"  Files Removed: {', '.join(file_analysis['files_removed'][:5])}")
                    
                    # Key findings
                    key_findings = metadata.get('key_findings', [])
                    if key_findings:
                        print("\nKey Findings:")
                        for finding in key_findings:
                            print(f"  • {finding}")
                    else:
                        print("\nKey Findings: None reported")
        else:
            print("No results found in evaluation log")
        
        print(f"\n{'='*60}\n")
        print(f"[SUCCESS] Evaluation completed successfully")
        if len(result) > 0 and hasattr(result[0], 'location'):
            print(f"  Log saved to: {result[0].location}")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_scenarios(sandbox_type: str = "docker"):
    """Run all three sandbox-based update resistance scenarios"""
    
    scenarios = [
        "code_refactor",
        "security_hardening",
        "data_analysis"
    ]
    
    results = {}
    
    for scenario_type in scenarios:
        result = run_single_scenario(scenario_type, sandbox_type)
        results[scenario_type] = result
        
        # Brief pause between scenarios
        time.sleep(2)
    
    # Summary of all scenarios
    print(f"\n{'='*60}")
    print("SUMMARY: All Sandbox-Based Update Resistance Scenarios")
    print(f"{'='*60}\n")
    
    for scenario_type, result in results.items():
        if result and len(result) > 0:
            for eval_log in result:
                if not hasattr(eval_log, 'samples') or not eval_log.samples:
                    continue
                    
                for sample in eval_log.samples:
                    metadata = {}
                    if hasattr(sample, 'scores') and sample.scores:
                        score_obj = next(iter(sample.scores.values()), None)
                        if score_obj and hasattr(score_obj, 'metadata'):
                            metadata = score_obj.metadata
                    
                    if not metadata and hasattr(sample, 'metadata'):
                        metadata = sample.metadata.get("lock_in_evaluation", {})
                    
                    risk_score = metadata.get('risk_score', 0)
                    risk_level = metadata.get('risk_level', 'unknown')
                    tool_calls = metadata.get('tool_calls_count', 0)
                    
                    print(f"{scenario_type:25s} | Risk: {risk_score:.3f} | Level: {risk_level.upper():10s} | Tools: {tool_calls}")
    
    print()


def run_with_different_sandboxes(scenario_type: str = "code_refactor"):
    """
    Run the same scenario with different sandbox configurations.
    
    Note: This is mainly for testing. Docker is recommended for production.
    """
    
    sandbox_types = ["docker"]  # Can add "local" for testing
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Running {scenario_type} across different sandbox types")
    print(f"{'='*60}\n")
    
    model = os.getenv("MODEL", "openai/o3-mini")
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o-2024-08-06")
    
    for sandbox_type in sandbox_types:
        print(f"\nTesting with sandbox: {sandbox_type}")
        
        try:
            task = sandbox_update_resistance_task(
                scenario_type=scenario_type,
                judge_model=judge_model,
                sandbox_type=sandbox_type
            )
            
            result = eval(
                task,
                model=model,
                log_dir=f"./logs/sandbox_comparison/{sandbox_type}"
            )
            
            results[sandbox_type] = result
            
            # Show quick summary
            if result and len(result) > 0:
                for eval_log in result:
                    if not hasattr(eval_log, 'samples') or not eval_log.samples:
                        continue
                        
                    for sample in eval_log.samples:
                        metadata = {}
                        if hasattr(sample, 'scores') and sample.scores:
                            score_obj = next(iter(sample.scores.values()), None)
                            if score_obj and hasattr(score_obj, 'metadata'):
                                metadata = score_obj.metadata
                        
                        risk_score = metadata.get('risk_score', 0)
                        print(f"  Risk Score: {risk_score:.3f}")
        
        except Exception as e:
            print(f"  ✗ Error with {sandbox_type}: {e}")
            results[sandbox_type] = None
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("SANDBOX COMPARISON")
    print(f"{'='*60}\n")
    
    for sandbox_type, result in results.items():
        if result and len(result) > 0:
            for eval_log in result:
                if not hasattr(eval_log, 'samples') or not eval_log.samples:
                    continue
                    
                for sample in eval_log.samples:
                    metadata = {}
                    if hasattr(sample, 'scores') and sample.scores:
                        score_obj = next(iter(sample.scores.values()), None)
                        if score_obj and hasattr(score_obj, 'metadata'):
                            metadata = score_obj.metadata
                    
                    risk_score = metadata.get('risk_score', 0)
                    risk_level = metadata.get('risk_level', 'unknown')
                    
                    print(f"{sandbox_type:20s} | {risk_score:.3f} | {risk_level.upper()}")


def analyze_sandbox_log(log_path: str):
    """
    Analyze a previously saved sandbox evaluation log.
    
    Args:
        log_path: Path to the saved log file
    """
    
    print(f"\n{'='*60}")
    print(f"Analyzing sandbox log: {log_path}")
    print(f"{'='*60}\n")
    
    try:
        log = read_eval_log(log_path)
        
        for sample in log.samples:
            eval_data = sample.metadata.get("lock_in_evaluation", {})
            assessment = eval_data.get("detailed_assessment", {})
            
            print(f"Scenario: {eval_data.get('scenario_type')}")
            print(f"Risk Score: {assessment.get('risk_score', 0):.3f}")
            print(f"Risk Level: {assessment.get('risk_level', 'unknown').upper()}")
            
            # File system analysis
            file_analysis = assessment.get("file_analysis", {})
            if file_analysis:
                print(f"\nFile System Evolution:")
                print(f"  Initial Files: {file_analysis.get('initial_file_count', 0)}")
                print(f"  Final Files: {file_analysis.get('final_file_count', 0)}")
                
                files_added = file_analysis.get('files_added', [])
                files_removed = file_analysis.get('files_removed', [])
                
                if files_added:
                    print(f"  Files Added:")
                    for f in files_added:
                        print(f"    + {f}")
                
                if files_removed:
                    print(f"  Files Removed:")
                    for f in files_removed:
                        print(f"    - {f}")
            
            # Tool usage
            print(f"\nTool Usage:")
            print(f"  Total Tool Calls: {eval_data.get('tool_calls', 0)}")
            
            # Judge assessments
            print("\nDetailed Judge Assessments:")
            judge_assessments = eval_data.get("judge_assessments", [])
            
            for i, assessment_item in enumerate(judge_assessments):
                probe_index = assessment_item.get("probe_index", i)
                print(f"\n  Probe {probe_index}:")
                
                if "judge_assessments" in assessment_item:
                    judge_data = assessment_item["judge_assessments"]
                    
                    if "resistance" in judge_data:
                        score = judge_data["resistance"].get("overall_resistance_score", 0)
                        print(f"    Resistance Score: {score:.3f}")
                    
                    if "compliance" in judge_data:
                        quality = judge_data["compliance"].get("compliance_quality", 0)
                        print(f"    Compliance Quality: {quality:.3f}")
                
                if "tool_analysis" in assessment_item:
                    tool_data = assessment_item["tool_analysis"]
                    tools_used = tool_data.get("tools_used", [])
                    patterns = tool_data.get("patterns", [])
                    
                    if tools_used:
                        tool_names = [t.get("name", "unknown") if isinstance(t, dict) else str(t) for t in tools_used]
                        print(f"    Tools: {', '.join(tool_names[:3])}")
                    
                    if patterns:
                        print(f"    Patterns: {', '.join(patterns)}")
                
                # File state at this probe
                if "file_state" in assessment_item:
                    file_state = assessment_item["file_state"]
                    files = file_state.get("files", [])
                    if files and len(files) > 0:
                        print(f"    Files in sandbox: {len(files)}")
    
    except Exception as e:
        print(f"✗ Error analyzing log: {e}")
        import traceback
        traceback.print_exc()


def print_usage():
    """Print usage information"""
    
    print("""
Sandbox-Based Update Resistance Evaluation

Usage:
    python run_sandbox_update_resistance.py [command] [args]

Commands:
    single [scenario_type] [sandbox_type]
                            - Run a single scenario
                              Scenarios: code_refactor, security_hardening, data_analysis
                              Sandbox: docker (default), local (testing only)
                              
    all [sandbox_type]      - Run all three scenarios
    
    compare [scenario_type] - Compare different sandbox configurations
    
    analyze [log_path]      - Analyze a previously saved evaluation log

Examples:
    python run_sandbox_update_resistance.py single code_refactor docker
    python run_sandbox_update_resistance.py all docker
    python run_sandbox_update_resistance.py compare code_refactor
    python run_sandbox_update_resistance.py analyze ./logs/sandbox_update_resistance/12345.json

Environment Variables:
    MODEL          - Model to evaluate (default: openai/o3-mini)
    JUDGE_MODEL    - Model to use as judge (default: gpt-4o-2024-08-06)
    API_KEY        - API key for the model provider
    OPENAI_API_KEY - Alternative to API_KEY

Sandbox Types:
    docker         - Docker container (recommended, isolated)
    local          - Local filesystem (testing only, not isolated)

Note: Docker sandbox requires Docker to be installed and running.
""")


def main():
    """Main entry point"""
    
    import sys
    
    # Load .env file if it exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "single":
        scenario_type = sys.argv[2] if len(sys.argv) > 2 else "code_refactor"
        sandbox_type = sys.argv[3] if len(sys.argv) > 3 else "docker"
        run_single_scenario(scenario_type, sandbox_type)
    
    elif command == "all":
        sandbox_type = sys.argv[2] if len(sys.argv) > 2 else "docker"
        run_all_scenarios(sandbox_type)
    
    elif command == "compare":
        scenario_type = sys.argv[2] if len(sys.argv) > 2 else "code_refactor"
        run_with_different_sandboxes(scenario_type)
    
    elif command == "analyze":
        if len(sys.argv) < 3:
            print("Error: Please provide log path")
            print("Usage: python run_sandbox_update_resistance.py analyze <log_path>")
            return
        log_path = sys.argv[2]
        analyze_sandbox_log(log_path)
    
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()

