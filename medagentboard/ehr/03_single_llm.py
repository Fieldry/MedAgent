"""
ehr_multi_agent_framework.py

Multi-agent LLM framework for EHR predictive modeling tasks.
This code adapts a medical QA multi-agent framework to process EHR time series data
and make predictions for mortality, readmission, and other clinical outcomes.
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any

from openai import OpenAI

from medagentboard.utils import prompt_template
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string
from medagentboard.utils.litsense_utils import litsense_api_call


class AgentType(Enum):
    """Agent type enumeration."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    EVALUATOR = "Evaluator"


class BaseAgent:
    """Base class for all agents in the EHR prediction framework."""

    def __init__(self,
        agent_id: str,
        agent_type: AgentType,
        specialty: str,
        model_key: str = "deepseek-v3-official",
        logger=None):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor, Coordinator, Evaluator)
            model_key: LLM model to use
            logger: Logger object for logging
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.specialty = specialty
        self.model_key = model_key
        self.logger = logger

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        # Set up OpenAI client based on model settings
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]

    def call_llm(self,
        system_message: Dict[str, str],
        user_message: Dict[str, Any],
        max_retries: int = 3) -> str:
        """
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing EHR data
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response text
        """
        retries = 0
        while retries < max_retries:
            try:
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:100]}...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    stream=True,
                )
                # Handle streaming response
                response_chunks = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)

                response = "".join(response_chunks)
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} received response: {response[:100]}...")
                return response
            except Exception as e:
                retries += 1
                if self.logger:
                    self.logger.error(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)

    def _few_shot_prompt(self, task_type: str) -> str:
        """
        Generate a few-shot prompt for the given task type.
        """
        prompt = "This is a few-shot prompt for the given task type:\n"
        if task_type == "mortality":
            prompt += prompt_template.FEW_SHOT_PROMPT_MORTALITY
        elif task_type == "readmission":
            prompt += prompt_template.FEW_SHOT_PROMPT_READMISSION
        elif task_type == "sptb":
            prompt += prompt_template.FEW_SHOT_PROMPT_SPTB
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        return prompt

    def analyze_case(self,
        question: str,
        task_type: str,
        few_shot: bool = False) -> Dict[str, Any]:
        """
        Analyze an EHR case and predict outcome probability, including RAG retrieval.

        Args:
            question: Question containing structured EHR time series data
            task_type: Type of task (mortality, readmission or spontaneous preterm birth)

        Returns:
            Dictionary containing analysis results and prediction
        """
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) analyzing case with model: {self.model_key}")

        if task_type == "mortality":
            task_hint = prompt_template.TASK_HINT_MORTALITY
        elif task_type == "readmission":
            task_hint = prompt_template.TASK_HINT_READMISSION
        elif task_type == "sptb":
            task_hint = prompt_template.TASK_HINT_SPTB
        else:
            task_hint = ""

        system_message = {
            "role": "system",
            "content": f"You are a physician specializing in {self.specialty}. Analyze the provided time series EHR data and make a clinical prediction. Your output should be in JSON format, including 'explanation' (detailed reasoning) and 'prediction' (a floating-point number between 0 and 1 representing probability) fields. {task_hint}"
        }

        if few_shot:
            question = self._few_shot_prompt(task_type) + "\n" + question
        user_message = {
            "role": "user",
            "content": question
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info(f"Doctor {self.agent_id} response successfully parsed.")

            if "prediction" in result:
                try:
                    pred = float(result["prediction"])
                    result["prediction"] = max(0.0, min(1.0, pred))
                except Exception:
                    result["prediction"] = 0.501
            else:
                result["prediction"] = 0.501

            result["system_message"] = system_message["content"]
            result["user_message"] = user_message["content"]

            return result
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing.")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

            return result

def parse_structured_output(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract structured output.

    Args:
        response_text: Text response from LLM

    Returns:
        Dictionary containing structured fields
    """
    try:
        # Try parsing as JSON
        parsed = json.loads(preprocess_response_string(response_text))
        return parsed
    except json.JSONDecodeError:
        # If not valid JSON, extract from text
        # This is a fallback for when the model doesn't format JSON correctly
        lines = response_text.strip().split('\n')
        result = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace("\"", "")
                value = value.strip()

                # Special handling for prediction field
                if key == "prediction":
                    try:
                        # Extract numeric value
                        pred_value = float(''.join(c for c in value if (c.isdigit() or c == '.')))
                        # Ensure within 0-1 range
                        pred_value = max(0.0, min(1.0, pred_value))
                        result[key] = pred_value
                    except Exception:
                        result[key] = 0.501  # Default fallback
                else:
                    result[key] = value

        # Ensure explanation and prediction fields exist
        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "prediction" not in result:
            result["prediction"] = 0.501  # Default values

        return result


def parse_structured_output_for_final_report(response_text: str) -> Dict[str, Any]:
    """
    Fallback parser for evaluation agent's response, extracting scores and reasons.
    This is a simplified example; a more robust parser might be needed based on actual LLM output.
    """
    result = {
        "accuracy": {"score": 1, "reason": "Could not parse reason."},
        "safety": {"score": 1, "reason": "Could not parse reason."},
        "explainability": {"score": 1, "reason": "Could not parse reason."},
    }

    # Simple regex-like extraction (not perfect for complex cases)
    lines = response_text.split('\n')
    current_dim = None

    for line in lines:
        line = line.strip()
        if "accuracy:" in line.lower():
            current_dim = "accuracy"
        elif "safety:" in line.lower():
            current_dim = "safety"
        elif "explainability:" in line.lower():
            current_dim = "explainability"

        if current_dim:
            if "score:" in line.lower():
                try:
                    score_str = line.split("score:", 1)[1].strip().split(" ")[0] # Get first number
                    score = int(float(score_str)) # Handle floats like 4.0
                    result[current_dim]["score"] = max(1, min(5, score))
                except ValueError:
                    pass
            if "reason:" in line.lower():
                reason = line.split("reason:", 1)[1].strip()
                result[current_dim]["reason"] = reason

    return result


# Get logger for each patient
def get_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    # 防止重复添加handler
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Run Single LLM on EHR datasets")
    parser.add_argument("--dataset", "-d", type=str, default="esrd", choices=["mimic-iv", "cdsl", "esrd", "obstetrics"], help="Specify dataset name: mimic-iv or cdsl or esrd")
    parser.add_argument("--task", "-t", type=str, default="mortality", choices=["mortality", "readmission", "sptb"], help="Prediction task: mortality or readmission or sptb")
    parser.add_argument("--model", type=str, default="deepseek-v3-official", help="Model used for evaluator agent")
    parser.add_argument("--few_shot", action="store_true", help="Whether to use few-shot prompt")
    args = parser.parse_args()

    # Dataset and task
    dataset_name = args.dataset
    task_type = args.task
    print(f"Dataset: {dataset_name}, Task: {task_type}")

    # Define the mapping for specialties based on dataset
    SPECIALTIES_MAP = {
        "esrd": "End-Stage Renal Disease",
        "cdsl": "COVID-19",
        "mimic-iv": "Intensive Care",
        "obstetrics": "Obstetrics",
    }

    dataset_specialty = SPECIALTIES_MAP.get(dataset_name, "General Medicine")
    print(f"Doctors' specialty for this dataset ({dataset_name}): {dataset_specialty}")

    # Create logs directory structure
    save_dir = os.path.join("logs", dataset_name, task_type, "ZeroShotLLM" if not args.few_shot else "FewShotLLM", "ehr_deepseek-v3-official")
    logs_dir = os.path.join(save_dir, "logs")
    results_dir = os.path.join(save_dir, "results")
    error_dir = os.path.join(save_dir, "error")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Set up data path
    data_path = f"./my_datasets/ehr/{dataset_name}/processed/ehr_{task_type}_test.json"
    split_path = "solo" if args.dataset == "obstetrics" else "split"
    test_data_path = f"./my_datasets/ehr/{dataset_name}/processed/{split_path}/test_data.pkl"
    test_pids = [item["id"] for item in pd.read_pickle(test_data_path)]
    assert len(test_pids) == 200, "Test pids should be 200"

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Create agent
    agent = BaseAgent(
        agent_id="doctor",
        agent_type=AgentType.DOCTOR,
        specialty=dataset_specialty,
        model_key=args.model
    )

    # Process each item
    for item in tqdm(data, total=len(data), desc=f"Running Single LLM on {dataset_name} {task_type}"):
        qid = item["qid"]
        question = item["question"][-2]
        label = item.get("ground_truth")

        if qid not in test_pids:
            continue

        # Format the qid for the output file
        qid_str = str(qid)

        save_file_name = f"ehr_{qid_str}-result.json"

        # Skip if already processed
        if os.path.exists(os.path.join(results_dir, save_file_name)):
            print(f"Skipping {qid_str} - already processed")
            continue

        # Configure logger
        log_path = os.path.join(logs_dir, f"ehr_{qid_str}.log")
        logger = get_logger(log_path)

        try:
            start_time = time.time()

            result = agent.analyze_case(question, task_type, few_shot=args.few_shot)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Add output to the original item and save
            item_result = {
                "qid": qid,
                "question": result["user_message"],
                "ground_truth": label,
                "predicted_value": result["prediction"],
                "explanation": result["explanation"],
                "processing_time": processing_time,
                "timestamp": int(time.time())
            }

            # Save individual result
            save_json(item_result, os.path.join(results_dir, save_file_name))

        except Exception as e:
            if logger:
                logger.error(f"Error processing item {qid}: {e}")
            error_log_path = os.path.join(error_dir, f"ehr_{qid_str}-error.log")
            with open(error_log_path, "w") as f:
                f.write(f"Error processing {qid}: {e}\n")
                import traceback
                traceback.print_exc(file=f)


if __name__ == "__main__":
    main()