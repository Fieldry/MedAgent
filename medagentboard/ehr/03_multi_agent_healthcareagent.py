# ehr_multi_agent_healthcareagent.py

import os
import json
import time
import asyncio
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any

from openai import OpenAI

from medagentboard.utils import prompt_template
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import get_logger, preprocess_response_string, save_json, load_json, parse_structured_output


class AgentType(Enum):
    """Enumeration for agent types."""
    HEALTHCARE = "Healthcare"


class BaseAgent:
    """Base class for all agents in the framework."""

    def __init__(self,
        agent_id: str,
        agent_type: AgentType,
        model_key: str = "deepseek-v3-official",
        logger=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []
        self.logger = logger

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

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
        retries = 0
        while retries < max_retries:
            try:
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:100]}...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    stream=False, # Simplified for non-streaming
                )
                response = completion.choices[0].message.content
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


class HealthcareAgent(BaseAgent):
    """
    A multi-faceted AI agent for medical consultation, encapsulating Dialogue, Memory, and Processing modules.
    """
    def __init__(self,
                 agent_id: str,
                 model_key: str = "deepseek-v3-official",
                 logger=None):
        super().__init__(agent_id, AgentType.HEALTHCARE, model_key, logger)
        if self.logger:
            self.logger.info(f"Initializing Healthcare Agent, ID: {agent_id}, Model: {model_key}")

    def generate_preliminary_diagnosis(self, question: str, task_type: str) -> Dict[str, Any]:
        """
        Step 1: Generates an initial diagnosis using the Function Module.
        """
        if self.logger:
            self.logger.info(f"Agent {self.agent_id} generating preliminary diagnosis.")

        system_message = {"role": "system", "content": prompt_template.HEALTHCARE_DIAGNOSIS_SYSTEM.format(task_type=task_type)}
        user_message = {"role": "user", "content": prompt_template.HEALTHCARE_DIAGNOSIS_USER.format(question=question, task_type=task_type)}

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            pred = result.get("prediction", 0.501)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Preliminary diagnosis response is not valid JSON, using fallback parsing.")
            result = parse_structured_output(response_text)

        self.memory.append({"type": "preliminary_diagnosis", "content": result})
        return result

    def review_with_safety_module(self, preliminary_diagnosis: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Step 2: Reviews and refines the diagnosis using the Safety Module.
        """
        if self.logger:
            self.logger.info(f"Agent {self.agent_id} reviewing diagnosis with Safety Module.")

        system_message = {"role": "system", "content": prompt_template.HEALTHCARE_SAFETY_REVIEW_SYSTEM}
        user_message = {
            "role": "user",
            "content": prompt_template.HEALTHCARE_SAFETY_REVIEW_USER.format(
                question_short=question[:2000], # Use an excerpt for brevity
                preliminary_explanation=preliminary_diagnosis.get('explanation', ''),
                preliminary_prediction=preliminary_diagnosis.get('prediction', 0.501)
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            pred = result.get("prediction", 0.501)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Safety review response is not valid JSON, using fallback parsing.")
            result = parse_structured_output(response_text)

        self.memory.append({"type": "safety_reviewed_diagnosis", "content": result})
        return result

    def generate_final_report(self, final_diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 3: Generates a summarized final report using the Processing Module.
        """
        if self.logger:
            self.logger.info(f"Agent {self.agent_id} generating final report.")

        system_message = {"role": "system", "content": prompt_template.HEALTHCARE_FINAL_REPORT_SYSTEM}
        user_message = {
            "role": "user",
            "content": prompt_template.HEALTHCARE_FINAL_REPORT_USER.format(
                final_explanation=final_diagnosis.get('explanation', ''),
                final_prediction=final_diagnosis.get('prediction', 0.501)
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning("Final report response is not valid JSON, saving plain text.")
            result = {"final_report": response_text}

        self.memory.append({"type": "final_report", "content": result})
        return result


class MedicalConsultation:
    """Orchestrator for a medical consultation with a single Healthcare Agent."""

    def __init__(self,
                 agent_config: Dict[str, Any],
                 logger=None):
        self.logger = logger
        self.agent = HealthcareAgent(
            agent_id="Dr_AI_1",
            model_key=agent_config.get("model_key", "deepseek-v3-official"),
            logger=logger
        )
        if self.logger:
            self.logger.info(f"Medical Consultation initialized with agent: {self.agent.model_key}")

    async def run_consultation(self,
                               qid: str,
                               question: str,
                               task_type: str = "mortality",
                               label: str = None) -> Dict[str, Any]:
        if self.logger:
            self.logger.info(f"Starting consultation for case {qid} (Task: {task_type}, Label: {label})")

        case_history = {}

        # Step 1: Generate preliminary diagnosis
        preliminary_diagnosis = self.agent.generate_preliminary_diagnosis(question, task_type)
        case_history["preliminary_diagnosis"] = preliminary_diagnosis
        if self.logger:
            self.logger.info(f"Preliminary Diagnosis - Prediction: {preliminary_diagnosis.get('prediction', 'N/A')}")

        # Step 2: Review with Safety Module
        final_diagnosis = self.agent.review_with_safety_module(preliminary_diagnosis, question)
        case_history["final_diagnosis"] = final_diagnosis
        if self.logger:
            self.logger.info(f"Safety-Reviewed Diagnosis - Prediction: {final_diagnosis.get('prediction', 'N/A')}")

        # Step 3: Generate Final Report
        final_report = self.agent.generate_final_report(final_diagnosis)
        case_history["final_report"] = final_report
        if self.logger:
            self.logger.info("Final report generated.")

        return {
            "final_decision": final_diagnosis,
            "case_history": case_history
        }


async def main():
    parser = argparse.ArgumentParser(description="Run medical consultation on EHR datasets using the HealthcareAgent framework")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "cdsl", "esrd", "obstetrics"], help="Specify dataset name")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "sptb"], help="Prediction task")
    parser.add_argument("--modality", "-mo", type=str, default="ehr", choices=["ehr", "note", "mm"], help="Modality of the dataset")
    parser.add_argument("--agent_model", type=str, default="deepseek-v3-official", help="Model for the Healthcare Agent")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of the data chunk to process")
    args = parser.parse_args()

    # Setup directories and paths
    method = "HealthcareAgent"
    save_dir = os.path.join("logs", args.dataset, args.task, method, f"{args.modality}_{args.agent_model}")
    logs_dir, results_dir, error_dir = [os.path.join(save_dir, d) for d in ["logs", "results", "error"]]
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    data_path = f"./my_datasets/ehr/{args.dataset}/processed/{args.modality}_{args.task}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Process a chunk of data based on start_index
    chunk_size = 100
    data_to_process = data[args.start_index : args.start_index + chunk_size]
    if not data_to_process:
        print(f"No data to process for start_index {args.start_index}. Exiting.")
        return
    print(f"Processing a chunk of {len(data_to_process)} samples from index {args.start_index}.")

    agent_config = {"model_key": args.agent_model}

    # Main processing loop
    for item in tqdm(data_to_process, desc=f"Running consultation on {args.dataset} chunk"):
        qid_str = str(item["qid"])
        save_file_name = f"ehr_{qid_str}-result.json"

        if os.path.exists(os.path.join(results_dir, save_file_name)):
            print(f"Skipping {qid_str} - already processed")
            continue

        logger = get_logger(os.path.join(logs_dir, f"ehr_{qid_str}.log"))

        try:
            start_time = time.time()
            consultation = MedicalConsultation(agent_config=agent_config, logger=logger)

            # Healthcare Agent uses a single 'question' input
            question_input = item["question"][0] if isinstance(item["question"], list) else item["question"]

            result = await consultation.run_consultation(
                qid=item["qid"], question=question_input,
                task_type=args.task, label=item.get("ground_truth")
            )

            item_result = {
                "qid": item["qid"], "question": item["question"], "ground_truth": item.get("ground_truth"),
                "predicted_value": result["final_decision"]["prediction"],
                "case_history": result["case_history"],
                "processing_time": time.time() - start_time,
                "timestamp": int(time.time())
            }
            save_json(item_result, os.path.join(results_dir, save_file_name))

        except Exception as e:
            logger.error(f"Error processing item {qid_str}: {e}")
            with open(os.path.join(error_dir, f"ehr_{qid_str}-error.log"), "w") as f:
                import traceback
                traceback.print_exc(file=f)

if __name__ == "__main__":
    asyncio.run(main())