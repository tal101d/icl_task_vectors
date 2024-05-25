# This must be first
from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
import pickle
import time
import torch
from logger import logger
from typing import Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir
from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE



def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")


def evaluate_task(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, task_name: str, num_examples: int) -> None:
    try:
        seed_everything(41)
        accuracies = {}

        task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

        # Evaluate baseline
        baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
        predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
        torch.cuda.empty_cache()
        accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

        # Evaluate ICL and Task Vector
        num_test_datasets, num_dev_datasets = 50, 50
        test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
        dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
        icl_predictions = run_icl(model, tokenizer, task, test_datasets)
        torch.cuda.empty_cache()
        tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
            model,
            tokenizer,
            task,
            test_datasets,
            dev_datasets,
        )
        torch.cuda.empty_cache()
        accuracies["tv_dev_by_layer"] = tv_dev_accuracy_by_layer
        accuracies["icl"] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
        accuracies["tv"] = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)

        tv_ordered_tokens_by_layer = {}
        for layer_num in tv_dev_accuracy_by_layer.keys():
            task_hidden = task_hiddens.mean(axis=0)[layer_num]
            logits = hidden_to_logits(model, task_hidden)
            tv_ordered_tokens_by_layer[layer_num] = logits_top_tokens(logits, tokenizer, k=100)

        return accuracies, tv_ordered_tokens_by_layer

    except Exception as e:
        logger.error(f"Error evaluating task {task_name}: {e}", exc_info=True)
        raise


def run_main_experiment(
    model_type: str,
    model_variant: str,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    logger.info(f"Evaluating model: {model_type}, {model_variant}")

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    if os.path.exists(results_file):
        try:
            with open(results_file, "rb") as f:
                results = pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            logger.error(f"Error loading existing results file: {e}")
            results = {}
    else:
        results = {}

    try:
        logger.info("Loading model and tokenizer...")
        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
        logger.info("Loaded model and tokenizer.")
        logger.info(torch.cuda.memory_summary())
        tasks = get_all_tasks(tokenizer=tokenizer)
        num_examples = 5

        for i, task_name in enumerate(TASKS_TO_EVALUATE):
            if task_name in results:
                logger.info(f"Skipping task {i + 1}/{len(tasks)}: {task_name}")
                continue

            logger.info("\n" + "=" * 50)
            logger.info(f"Running task {i + 1}/{len(tasks)}: {task_name}")

            tic = time.time()
            accuracies, tv_ordered_tokens_by_layer = evaluate_task(model, tokenizer, task_name, num_examples)

            if accuracies and tv_ordered_tokens_by_layer:
                logger.info(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
                logger.info(f"ICL Accuracy: {accuracies['icl']:.2f}")
                logger.info(f"Task Vector Accuracy: {accuracies['tv']:.2f}")
                logger.info("Dev Accuracy by layer: " + ", ".join([f"{layer}: {accuracy:.2f}" for layer, accuracy in accuracies["tv_dev_by_layer"].items()]))
                logger.info("Time: %.2f", time.time() - tic)

                results[task_name] = {
                    "baseline_accuracy": accuracies["baseline"],
                    "num_examples": num_examples,
                    "icl_accuracy": accuracies["icl"],
                    "tv_accuracy": accuracies["tv"],
                    "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
                    "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
                }
                with open(results_file, "wb") as f:
                    pickle.dump(results, f)
            torch.cuda.empty_cache()
        

    except Exception as e:
        logger.error(f"Error during main experiment execution: {e}", exc_info=True)
        raise e


def get_new_experiment_id() -> str:
    try:
        return str(
            max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1
        )
    except Exception as e:
        logger.error(f"Error generating new experiment ID: {e}", exc_info=True)
        raise e
    
def print_GPU_memory_status():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        free = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        logger.info(f"GPU Memory - Allocated: {allocated / (1024**2):.2f} MiB, Reserved: {reserved / (1024**2):.2f} MiB, Free: {free / (1024**2):.2f} MiB")
    else:
        logger.info("CUDA is not available.")

def main():
    try:
        if len(sys.argv) == 1:
            # Run all models
            experiment_id = get_new_experiment_id()
            for model_type, model_variant in MODELS_TO_EVALUATE: # for loop for each model that we want to test
                run_main_experiment(model_type, model_variant, experiment_id=experiment_id)
                torch.cuda.empty_cache()
        else:
            if len(sys.argv) == 2:
                model_num = int(sys.argv[1])
                model_type, model_variant = MODELS_TO_EVALUATE[model_num]
            elif len(sys.argv) == 3:
                model_type, model_variant = sys.argv[1:]

            run_main_experiment(model_type, model_variant)

    except (IndexError, ValueError) as e:
        logger.error(f"Invalid command line arguments: {e}", exc_info=True)
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()
