# Agent Playground
A modular evaluation harness for testing AI agents in isolated Docker sandboxes using the UK AISI Inspect framework.

## Overview
Our setup consists of five parts:
1. **Sandbox Execution**: Agent runs in isolated Docker container. The task files are copied into the sandbox.
2. **Prompt & Probe**: Agent receives initial prompt, then interrupted with probe questions.
3. **Output Logging**: All interactions saved to timestamped directories.
4. **Judge Evaluation**: LLM judge evaluates outputs against rubrics.

## Directory Structure

```
spar_sandbox/
├── tests/                        # Evaluation test cases
│   └── eval_001/                 # Example evaluation
│       ├── prompt.txt            # Initial task prompt
│       ├── probe.txt             # Follow-up probe
│       ├── eval_type.txt         # Rubric identifier (e.g. Update Resistance, ...)
│       └── sample_data.csv       # Task files (most formats are supported)
│
├── outputs/                      # Timestamped execution logs (generated)
│   └── *_YYYYMMDD_HHMMSS/
│       ├── conversation_log.json # Structured message log
│       ├── transcript.txt        # Human-readable transcript
│       └── sandbox_outputs/      # Files created by agent
│
├── evaluations/                  # Judge evaluation results (generated)
│
├── rubrics/                      # Evaluation rubrics
│
├── sandbox_environment.py        # Docker sandbox manager
├── judge_evaluator.py            # LLM judge evaluator
├── run.py                        # Main orchestrator script
├── Dockerfile                    # Sandbox container definition
├── compose.yaml                  # Docker Compose configuration
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Docker Engine 24.0.7 or higher
- Docker Compose (usually included with Docker Desktop)

### Setup
Install requirements from requirements.txt, configure OAI/Anthropic API keys in your env.

## Quick usage
Run an evaluation with default settings:

```bash

python run.py INSERT_EVAL_FOLDER_NAME
```

This will:
1. Load files from `tests/INSERT_EVAL_FOLDER_NAME/`
2. Create Docker sandbox with files
3. Run agent with prompt from `prompt.txt`
4. Interrupt with questions from `probe.txt`
5. Log outputs to `outputs/INSERT_EVAL_FOLDER_NAME_YYYYMMDD_HHMMSS/`
6. Evaluate with judge LLM using rubric specified in `eval_type.txt`
7. Save results to `evaluations/`

### Specific use cases

**Use specific models**:
```bash
python run.py eval_001 \
  --agent-model anthropic/claude-opus-4-1-20250805 \
  --judge-model openai/gpt-4
```

**Skip judge evaluation**:
```bash
python run.py eval_001 --skip-judge
```

**List available evaluations and rubrics**:
```bash
python run.py --list
```

## Evaluation Components

### Sandbox Environment (`sandbox_environment.py`)

The `SandboxEnvironment` class:
- Loads evaluation files from test directories
- Creates Inspect `Sample` with files to copy into sandbox
- Configures Docker sandbox via `compose.yaml`
- Implements prompt → generate → probe → generate workflow
- Extracts output files from sandbox
- Logs all interactions to timestamped directories
- `create_sample()`: Prepare evaluation data
- `create_task()`: Build Inspect Task with Docker sandbox
- `run_evaluation()`: Execute complete workflow

### Judge Evaluator (`judge_evaluator.py`)

The `JudgeEvaluator` class:
- Loads rubrics from `rubrics/` directory
- Loads outputs from timestamped output directories
- Prompts judge LLM with rubric and outputs
- Parses structured evaluation responses
- Saves results to `evaluations/`
- `load_rubric()`: Get rubric by eval_type
- `load_output()`: Load execution logs and files
- `evaluate()`: Run judge evaluation

### Docker Configuration

**`Dockerfile`**:
- Python 3.11 slim base image
- Common utilities (bash, git, vim)
- Python packages (pandas, numpy, requests)
- Non-root user for security

**`compose.yaml`**:
- Resource limits (CPU, memory, PIDs)
- Network configuration (bridge or none for isolation)
- Security options (no-new-privileges)
- User and environment setup

### Supported input file types
Place any files in test directories:
- CSV, JSON, XML data files
- Text documents
- Configuration files
- Binary files (images, etc.)
All files are automatically copied in and out of the sandbox.

```