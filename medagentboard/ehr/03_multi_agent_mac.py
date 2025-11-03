# ehr_multi_agent_mac.py

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
    """Enumeration for MAC agent types."""
    ADMIN = "Admin"
    DOCTOR = "Doctor"
    SUPERVISOR = "Supervisor"


class BaseAgent:
    """Base class for all agents in the MAC framework."""

    def __init__(self,
        agent_id: str,
        agent_type: AgentType,
        model_key: str = "deepseek-v3-official",
        logger=None):
        """
        Initializes the base agent.
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.logger = logger

        # AdminAgent does not require an LLM
        if self.agent_type != AgentType.ADMIN:
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
        """
        Calls the language model and handles retries.
        """
        retries = 0
        while retries < max_retries:
            try:
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} calling LLM...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    stream=False, # MAC is conversational, not streaming chunks
                )
                response = completion.choices[0].message.content
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} received response: {response[:150]}...")
                return response
            except Exception as e:
                retries += 1
                if self.logger:
                    self.logger.error(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(2)  # Pause before retrying


class AdminAgent(BaseAgent):
    """Admin agent to initiate the diagnostic conversation."""
    def __init__(self, agent_id: str = "Admin", logger=None):
        super().__init__(agent_id, AgentType.ADMIN, model_key=None, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing Admin agent, ID: {agent_id}")

    def present_case(self, patient_data: str, task_type: str) -> str:
        """Formats the initial case presentation."""
        if "Primary Consultation" in task_type:
            task_description = "Your tasks are: 1. Formulate the most likely diagnosis. 2. Formulate several possible differential diagnoses. 3. Recommend further diagnostic tests."
        else: # Follow-up consultation
            task_description = "Your tasks are: 1. Formulate the most likely diagnosis. 2. Formulate several possible differential diagnoses."

        presentation = (
            f"--- PATIENT CASE ---\n"
            f"{patient_data}\n\n"
            f"--- DIAGNOSTIC TASK ---\n"
            f"{task_description}\n"
            f"The diagnostic discussion begins now."
        )
        if self.logger:
            self.logger.info("Admin agent presented the case.")
        return presentation


class DoctorAgent(BaseAgent):
    """Doctor agent for contributing medical reasoning to the diagnosis."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        super().__init__(agent_id, AgentType.DOCTOR, model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing doctor agent, ID: {agent_id}, Model: {model_key}")

    def generate_contribution(self, conversation_history: str) -> Dict[str, Any]:
        """Contributes to the ongoing diagnostic conversation."""
        system_message = {"role": "system", "content": prompt_template.MAC_DOCTOR_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.MAC_DOCTOR_USER.format(conversation_history=conversation_history)}

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} response is not valid JSON, returning raw text.")
            result = {"contribution": response_text} # Fallback

        return result.get("contribution", "No contribution provided.")


class SupervisorAgent(BaseAgent):
    """Supervisor agent to moderate the discussion and facilitate consensus."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        super().__init__(agent_id, AgentType.SUPERVISOR, model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing supervisor agent, ID: {agent_id}, Model: {model_key}")

    def moderate_discussion(self, conversation_history: str) -> Dict[str, Any]:
        """Moderates the discussion and determines if it should continue."""
        system_message = {"role": "system", "content": prompt_template.MAC_SUPERVISOR_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.MAC_SUPERVISOR_USER.format(conversation_history=conversation_history)}

        response_text = self.call_llm(system_message, user_message)
        result = {"comment": "Could not parse supervisor response.", "continue_discussion": True} # Default

        try:
            parsed_response = json.loads(preprocess_response_string(response_text))
            result["comment"] = parsed_response.get("comment", "No comment provided.")
            result["continue_discussion"] = parsed_response.get("continue_discussion", True)
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning(f"Supervisor response is not valid JSON, using fallback.")
            # Simple heuristic: if the response contains "conclude", "consensus", "final", stop.
            if any(keyword in response_text.lower() for keyword in ["conclude", "consensus", "final"]):
                 result["continue_discussion"] = False
            result["comment"] = response_text

        return result

    def finalize_diagnosis(self, conversation_history: str) -> Dict[str, Any]:
        """Summarizes the conversation into a final structured diagnosis."""
        if self.logger:
            self.logger.info("Supervisor is finalizing the diagnosis.")
        system_message = {"role": "system", "content": prompt_template.MAC_FINALIZER_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.MAC_FINALIZER_USER.format(conversation_history=conversation_history)}

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning("Finalizer response is not valid JSON, using fallback parsing.")
            result = parse_structured_output(response_text, ["most_likely_diagnosis", "possible_diagnoses", "recommended_tests"])

        return result


class MACConsultation:
    """Orchestrator for a Multi-Agent Conversation (MAC) for disease diagnosis."""

    def __init__(self,
        max_rounds: int = 4, # A round consists of all doctors + 1 supervisor turn
        num_doctors: int = 3,
        doctor_model_key: str = "deepseek-v3-official",
        supervisor_model_key: str = "deepseek-v3-official",
        logger=None):
        """Initializes the MAC consultation process."""
        self.max_rounds = max_rounds
        self.logger = logger

        # Initialize agents
        self.admin_agent = AdminAgent(logger=logger)
        self.doctor_agents = [
            DoctorAgent(f"Doctor_{i+1}", doctor_model_key, logger=logger)
            for i in range(num_doctors)
        ]
        self.supervisor_agent = SupervisorAgent("Supervisor", supervisor_model_key, logger=logger)

        if self.logger:
            self.logger.info(f"Initialized MAC consultation, max_rounds={max_rounds}, num_doctors={num_doctors}")

    async def run_consultation(self, qid: str, question: str, task_type: str) -> Dict[str, Any]:
        """Runs the full MAC consultation process for a given case."""
        if self.logger:
            self.logger.info(f"Starting MAC consultation for case {qid} (Task: {task_type})")

        conversation_history = []
        conversation_str = ""

        # Step 1: Admin presents the case
        initial_presentation = self.admin_agent.present_case(question, task_type)
        conversation_history.append({"agent": self.admin_agent.agent_id, "content": initial_presentation})
        conversation_str += f"{self.admin_agent.agent_id}:\n{initial_presentation}\n\n"

        # Step 2: Iterative conversation rounds
        for current_round in range(1, self.max_rounds + 1):
            if self.logger:
                self.logger.info(f"--- Starting Conversation Round {current_round} ---")

            # Doctor turns
            for doctor in self.doctor_agents:
                contribution = doctor.generate_contribution(conversation_str)
                conversation_history.append({"agent": doctor.agent_id, "content": contribution})
                conversation_str += f"{doctor.agent_id}:\n{contribution}\n\n"

            # Supervisor turn
            moderation = self.supervisor_agent.moderate_discussion(conversation_str)
            conversation_history.append({"agent": self.supervisor_agent.agent_id, "content": moderation["comment"]})
            conversation_str += f"{self.supervisor_agent.agent_id}:\n{moderation['comment']}\n\n"

            if not moderation["continue_discussion"]:
                if self.logger:
                    self.logger.info(f"Supervisor ended discussion in round {current_round}.")
                break

        if self.logger:
            self.logger.info("Maximum rounds reached or discussion concluded.")

        # Step 3: Finalize the diagnosis
        final_diagnosis = self.supervisor_agent.finalize_diagnosis(conversation_str)

        return {
            "conversation_history": conversation_history,
            "final_diagnosis": final_diagnosis,
            "total_rounds": current_round
        }


async def main():
    parser = argparse.ArgumentParser(description="Run MAC consultation on diagnostic datasets")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Specify dataset name (e.g., 'rare_disease_test')")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["Primary Consultation", "Follow-up Consultation"], help="Consultation type")
    parser.add_argument("--doctor_model", type=str, default="deepseek-v3-official", help="Model for doctor agents")
    parser.add_argument("--supervisor_model", type=str, default="deepseek-v3-official", help="Model for supervisor agent")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of the data chunk to process")
    args = parser.parse_args()

    method = "MAC"
    save_dir = os.path.join("logs", args.dataset, args.task, method, f"{args.supervisor_model}")
    logs_dir, results_dir, error_dir = [os.path.join(save_dir, d) for d in ["logs", "results", "error"]]
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Note: Adjust the data path to your specific MAC dataset file structure
    data_path = f"./my_datasets/mac/{args.dataset}.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    chunk_size = 100
    data_to_process = data[args.start_index : args.start_index + chunk_size]
    if not data_to_process:
        print(f"No data to process for start_index {args.start_index}. Exiting.")
        return
    print(f"Processing a chunk of {len(data_to_process)} samples from index {args.start_index}.")

    for item in tqdm(data_to_process, desc=f"Running MAC on {args.dataset} chunk"):
        qid_str = str(item["qid"])
        save_file_name = f"diag_{qid_str}-result.json"

        if os.path.exists(os.path.join(results_dir, save_file_name)):
            print(f"Skipping {qid_str} - already processed")
            continue

        logger = get_logger(os.path.join(logs_dir, f"diag_{qid_str}.log"))

        try:
            start_time = time.time()
            mac = MACConsultation(
                doctor_model_key=args.doctor_model,
                supervisor_model_key=args.supervisor_model,
                logger=logger
            )
            # MAC uses a single 'question' field for the case description
            result = await mac.run_consultation(
                qid=item["qid"],
                question=item["question"],
                task_type=args.task
            )

            item_result = {
                "qid": item["qid"],
                "question": item["question"],
                "ground_truth": item.get("ground_truth"),
                "diagnosis_output": result["final_diagnosis"],
                "full_consultation": result,
                "processing_time": time.time() - start_time,
                "timestamp": int(time.time())
            }
            save_json(item_result, os.path.join(results_dir, save_file_name))

        except Exception as e:
            logger.error(f"Error processing item {qid_str}: {e}")
            with open(os.path.join(error_dir, f"diag_{qid_str}-error.log"), "w") as f:
                import traceback
                traceback.print_exc(file=f)

if __name__ == "__main__":
    asyncio.run(main())