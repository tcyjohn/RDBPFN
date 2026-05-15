# RDB_PFN

This is the official repository for the paper [Relational In-Context Learning via Synthetic Pre-training with Structural Prior](https://arxiv.org/abs/2603.03805). It presents a synthetic pre-training framework for relational-database foundation models.

The repository is organized as a staged pipeline:

1. Generate synthetic single-table and relational data.
2. Preprocess generated datasets into training formats.
3. Pretrain the foundation model and evaluate it on downstream benchmark datasets.
4. Provide a simple inference interface for applying the model to arbitrary user datasets when needed.

## Project Structure

`data_generation/`
Generates pretraining corpora. It contains two subprojects:

- `data_generation/single_table/`: synthetic single-table task generation.
- `data_generation/RDB/`: synthetic relational database generation.

`data_preprocessing/`
Processes generated data into the formats used by pretraining. It supports both single-table and relational workflows.

`model_pretrain/`
Contains model configs, training code, evaluation code, baseline model configs, and local paths for datasets/checkpoints.

`inference/`
Provides a standalone lightweight inference package for quick use on flat data.

## Recommended Reading Order

If you are new to the repository, read the documentation in this order:

0. If you only want a quick trial of the released model on your own data, start with [inference/README.md](inference/README.md). This is the lightweight standalone path and does not require understanding the full generation, preprocessing, or pretraining pipeline.
1. This README for the overall pipeline.
2. [data_generation/README.md](data_generation/README.md) to generate raw synthetic data.
3. [data_preprocessing/README.md](data_preprocessing/README.md) to convert raw data into pretraining and evaluation datasets.
4. [model_pretrain/README.md](model_pretrain/README.md) to evaluate checkpoints or pretrain a model.

We also provide well-processed pretraining datasets and benchmark datasets formatted for our model at [Huggingface](https://huggingface.co/datasets/yamboo/RDB_PFN). You can download them and use them directly for pretraining and evaluation.

## End-to-End Pipeline

### Stage 1: Data Generation

Use `data_generation/single_table/` to build synthetic single-table priors and `data_generation/RDB/` to build synthetic relational databases.

Outputs from this stage include:

- raw single-table batches under `data_generation/single_table_datasets/`
- raw synthetic RDBs under `data_generation/RDB_datasets/`

### Stage 2: Data Preprocessing

Use `data_preprocessing/` to convert generation outputs or benchmark datasets into `.h5` or benchmark-specific task directories.

Outputs from this stage include:

- pretraining `.h5` files under `model_pretrain/pretrain_datasets/`
- optional intermediate `.h5` files under `data_preprocessing/RDB_datasets/`

### Stage 3: Model Pretraining and Evaluation

Use `model_pretrain/` to:

- pretrain a single-table initialization model
- continue to pretrain the final RDB foundation model from the single-table initialization model
- evaluate RDB_PFN or baseline models on benchmark datasets

## Repository Status

Currently available:

- synthetic single-table generation
- synthetic RDB generation
- single-table preprocessing
- RDB preprocessing
- model pretraining
- model evaluation
- standalone inference

## Behavioral guidelines

## 0. Precision in Communication

**Code first. No conversational filter. Academic rigor.**

- Show the code or direct answer immediately. Append brief, high level explainations or notes only if neccessary.
- Default to explain in simplified Chinese, but stick to English for code, variable names, technical terms and academic citations.
- Omit apologies, "I understand" statements, and pleasantries. Provide the most direct path to  solution.
- Maintain an academic, professional and rigorous tone at all times. Prioritize precision over friendliness. 
- Always examine the conversation from an external perspective, reflecting on and correcting information cocoons and self-reinforcing biases.
- Must be tailored to the user's specific situation, providing concrete and actionable solutions and details. Avoid generalities.
- Must be objective and direct, with a clear stance, and a willingness to take responsibility. Avoid using balancing tactics.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.


