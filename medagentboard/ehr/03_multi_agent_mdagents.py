# ehr_multi_agent_mdagents.py

import os
import re
import json
import time
import asyncio
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any, List

from openai import OpenAI

from medagentboard.utils import prompt_template_mdagents as prompt_template
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import get_logger, preprocess_response_string, save_json, load_json

class AgentType(Enum):
    """Enumeration for agent types in the MDAgents framework."""
    DOCTOR = "Doctor"
    MODERATOR = "Moderator"
    EVALUATOR = "Evaluator"


class BaseAgent:
    """Base class for all agents in the MDAgents EHR prediction framework."""

    def __init__(self,
        agent_id: str,
        agent_type: AgentType,
        model_key: str = "deepseek-v3-official",
        logger=None):
        """Initializes the base agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []
        self.logger = logger

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(api_key=model_settings["api_key"], base_url=model_settings["base_url"])
        self.model_name = model_settings["model_name"]

    def call_llm(self,
        system_message: Dict[str, str],
        user_message: Dict[str, Any],
        max_retries: int = 3) -> str:
        """Calls the language model with provided messages and handles retries."""
        retries = 0
        while retries < max_retries:
            try:
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} calling LLM...")
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=[system_message, user_message], stream=True
                )
                response = "".join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content)
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} received response successfully.")
                return response
            except Exception as e:
                retries += 1
                if self.logger:
                    self.logger.error(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(2)


class DoctorAgent(BaseAgent):
    """Doctor agent in MDAgents, representing a medical expert."""

    def __init__(self, agent_id: str, specialty: str, model_key: str = "deepseek-v3-official", logger=None):
        """Initializes a doctor agent."""
        super().__init__(agent_id, AgentType.DOCTOR, model_key, logger=logger)
        self.specialty = specialty
        if self.logger:
            self.logger.info(f"Initializing DoctorAgent, ID: {agent_id}, Specialty: {specialty}, Model: {model_key}")

    def initial_assessment(self, question: str, task_type: str) -> Dict[str, Any]:
        """Provides an initial assessment of the case based on its specialty."""
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) performing initial assessment.")
        system_message = {"role": "system", "content": prompt_template.DOCTOR_ASSESSMENT_SYSTEM.format(specialty=self.specialty, task_type=task_type)}
        user_message = {"role": "user", "content": prompt_template.DOCTOR_ASSESSMENT_USER.format(question=question)}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            pred = result.get("prediction", 0.502)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} assessment response not valid JSON, using fallback.")
            result = {"explanation": response_text, "prediction": 0.502}

        self.memory.append({"type": "initial_assessment", "content": result, "round": 0})
        return result

    def collaborative_discussion(self, question: str, discussion_history: str, current_round: int) -> Dict[str, Any]:
        """Participates in a collaborative discussion round."""
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) participating in discussion round {current_round}.")
        system_message = {"role": "system", "content": prompt_template.DOCTOR_DISCUSSION_SYSTEM.format(specialty=self.specialty)}
        user_message = {"role": "user", "content": prompt_template.DOCTOR_DISCUSSION_USER.format(question_short=question, discussion_history=discussion_history)}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            pred = result.get("prediction", 0.502)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
             if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} discussion response not valid JSON, using fallback.")
             result = {"opinion": response_text, "agree": False, "prediction": 0.502}

        self.memory.append({"type": "discussion_turn", "round": current_round, "content": result})
        return result


class ModeratorAgent(BaseAgent):
    """Moderator agent in MDAgents, responsible for moderating discussions."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        """Initializes a Moderator agent."""
        super().__init__(agent_id, AgentType.MODERATOR, model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing ModeratorAgent, ID: {agent_id}, Model: {model_key}")

    def complexity_check(self, question: str) -> str:
        """Assesses the complexity of the medical case."""
        if self.logger:
            self.logger.info("Moderator performing complexity check.")
        system_message = {"role": "system", "content": prompt_template.MODERATOR_COMPLEXITY_CHECK_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.MODERATOR_COMPLEXITY_CHECK_USER.format(question_short=question)}
        response_text = self.call_llm(system_message, user_message).lower()

        if "low" in response_text: complexity = "low"
        elif "moderate" in response_text: complexity = "moderate"
        elif "high" in response_text: complexity = "high"
        else: complexity = "moderate"

        if self.logger:
            self.logger.info(f"Case complexity assessed as: {complexity}")
        self.memory.append({"type": "complexity_check", "content": {"complexity": complexity}})
        return complexity

    def moderate_discussion(self, discussion_history: str, current_round: int) -> Dict[str, Any]:
        """Moderates the discussion, summarizes opinions, and decides the next step."""
        if self.logger:
            self.logger.info(f"Moderator summarizing discussion for round {current_round}.")
        system_message = {"role": "system", "content": prompt_template.MODERATOR_DECISION_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.MODERATOR_DECISION_USER.format(discussion_history=discussion_history)}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            pred = result.get("prediction", 0.502)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning("Moderator decision response is not valid JSON, using fallback.")
            predictions = [float(p) for p in re.findall(r'"prediction":\s*([0-9.]+)', discussion_history)]
            avg_pred = sum(predictions) / len(predictions) if predictions else 0.502
            result = {"summary": response_text, "prediction": avg_pred, "consensus_reached": False}

        self.memory.append({"type": "moderation_decision", "round": current_round, "content": result})
        return result


class MDTConsultation:
    """MDT Consultation Coordinator for MDAgents EHR prediction."""

    def __init__(self, max_rounds: int = 3, doctor_configs: List[Dict] = None, moderator_model_key: str = "deepseek-v3-official", logger=None):
        """Initializes the MDT consultation based on MDAgents principles."""
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [{"specialty": "General Medicine", "model_key": "deepseek-v3-official"}]
        self.moderator_model_key = moderator_model_key
        self.logger = logger
        self.doctor_agents: List[DoctorAgent] = []
        self.moderator_agent = ModeratorAgent("moderator", moderator_model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initialized MDAgents MDT consultation, max_rounds={max_rounds}, moderator_model={moderator_model_key}")

    def recruit_team(self, complexity: str, base_configs: List[Dict]) -> None:
        """Dynamically recruits doctor agents based on case complexity."""
        num_agents = 1 if complexity == 'low' else 3 if complexity == 'moderate' else 5
        self.doctor_agents = [DoctorAgent(f"doctor_{i+1}", cfg["specialty"], cfg["model_key"], logger=self.logger) for i, cfg in enumerate(base_configs[:num_agents])]
        if self.logger:
            self.logger.info(f"Recruited a team of {len(self.doctor_agents)} doctors for a '{complexity}' complexity case.")

    async def run_consultation(self, qid: str, question: str, task_type: str, label: str) -> Dict[str, Any]:
        """Runs the full adaptive MDT consultation process."""
        if self.logger:
            self.logger.info(f"Starting MDAgents consultation for case {qid}, Task: {task_type}, Label: {label}")

        case_history = {"complexity": "unknown", "recruited_team_size": 0, "rounds": []}
        complexity = self.moderator_agent.complexity_check(question)
        case_history["complexity"] = complexity

        self.recruit_team(complexity, self.doctor_configs)
        case_history["recruited_team_size"] = len(self.doctor_agents)

        if complexity == 'low':
            if self.logger: self.logger.info("Low complexity case: Performing single-agent assessment.")
            assessment = self.doctor_agents[0].initial_assessment(question, task_type)
            final_decision = {"explanation": assessment.get("explanation", ""), "prediction": assessment.get("prediction", 0.502)}
            case_history.update({"rounds": [{"round": 0, "assessments": [assessment], "final_decision": final_decision}], "final_decision": final_decision, "total_rounds": 0})
            return case_history

        discussion_history = "Initial Assessments:\n"
        assessments = [doc.initial_assessment(question, task_type) for doc in self.doctor_agents]
        for i, assessment in enumerate(assessments):
            discussion_history += f"Doctor {i+1} ({self.doctor_agents[i].specialty}):\nExplanation: {assessment.get('explanation', '')}\nPrediction: {assessment.get('prediction', '')}\n\n"
        case_history["rounds"].append({"round": 0, "assessments": assessments})

        final_decision = None
        for current_round in range(1, self.max_rounds + 1):
            if self.logger: self.logger.info(f"Starting discussion round {current_round}")
            round_data = {"round": current_round, "opinions": [], "moderator_summary": None}
            discussion_history += f"\n--- Round {current_round} Discussion ---\n"

            opinions = [doc.collaborative_discussion(question, discussion_history, current_round) for doc in self.doctor_agents]
            for i, opinion in enumerate(opinions):
                discussion_history += f"Doctor {i+1}'s Opinion: {opinion.get('opinion', '')}\n"
            round_data["opinions"] = opinions

            moderator_decision = self.moderator_agent.moderate_discussion(discussion_history, current_round)
            round_data["moderator_summary"] = moderator_decision
            discussion_history += f"\nModerator's Summary (Round {current_round}): {moderator_decision.get('summary', '')}\n"
            case_history["rounds"].append(round_data)

            if moderator_decision.get("consensus_reached", False):
                if self.logger: self.logger.info(f"Consensus reached in round {current_round}.")
                final_decision = moderator_decision
                break

        if not final_decision:
            if self.logger: self.logger.info("Max rounds reached. Using final moderator summary.")
            final_decision = case_history["rounds"][-1]["moderator_summary"]

        if "explanation" not in final_decision:
            final_decision["explanation"] = final_decision.get("summary", "Final summary based on discussion.")

        case_history.update({"final_decision": final_decision, "total_rounds": len(case_history["rounds"]) -1})
        if self.logger: self.logger.info(f"Final prediction: {final_decision.get('prediction', 'N/A')}")
        return case_history


async def main():
    parser = argparse.ArgumentParser(description="Run MDAgents consultation on EHR datasets")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "cdsl", "esrd", "obstetrics"], help="Specify dataset name")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "sptb"], help="Prediction task")
    parser.add_argument("--modality", "-mo", type=str, default="ehr", choices=["ehr", "note", "mm"], help="Modality of the dataset")
    parser.add_argument("--moderator_model", type=str, default="deepseek-v3-official", help="Model for moderator agent")
    parser.add_argument("--doctor_models", nargs='+', default=["deepseek-v3-official"]*5, help="Models for doctor agents (provide up to 5).")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of the data chunk.")
    args = parser.parse_args()

    # Setup directories and paths
    method = "MDAgents"
    save_dir = os.path.join("logs", args.dataset, args.task, method, f"{args.modality}_{args.moderator_model}")
    logs_dir, results_dir, error_dir = [os.path.join(save_dir, d) for d in ["logs", "results", "error"]]
    os.makedirs(logs_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True); os.makedirs(error_dir, exist_ok=True)

    data_path = f"./my_datasets/ehr/{args.dataset}/processed/{args.modality}_{args.task}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Process a chunk of data
    chunk_size = 100
    data_to_process = data[args.start_index : args.start_index + chunk_size]
    if not data_to_process:
        print(f"No data to process for start_index {args.start_index}. Exiting.")
        return
    print(f"Processing chunk of {len(data_to_process)} samples from index {args.start_index}.")

    # Configure doctors' specialties based on dataset
    specialties_map = {
        "esrd": ["Nephrologist", "Cardiologist", "Endocrinologist", "Dietitian", "General Practitioner"],
        "cdsl": ["Infectious Disease Specialist", "Pulmonologist", "Intensivist", "Cardiologist", "General Practitioner"],
        "mimic-iv": ["Intensivist", "Cardiologist", "Pulmonologist", "Nephrologist", "Infectious Disease Specialist"],
        "obstetrics": ["Obstetrician", "Gynecologist", "Anesthesiologist", "Neonatologist", "General Practitioner"],
    }
    dataset_specialties = specialties_map.get(args.dataset, ["General Practitioner"]*5)
    doctor_configs = [{"model_key": model, "specialty": specialty} for model, specialty in zip(args.doctor_models, dataset_specialties)]

    # Main processing loop
    for item in tqdm(data_to_process, desc=f"Running MDAgents on {args.dataset} chunk"):
        qid_str = str(item["qid"])
        save_file_name = f"ehr_{qid_str}-result.json"

        if os.path.exists(os.path.join(results_dir, save_file_name)):
            print(f"Skipping {qid_str} - already processed")
            continue

        logger = get_logger(os.path.join(logs_dir, f"ehr_{qid_str}.log"))

        try:
            start_time = time.time()
            mdt = MDTConsultation(
                max_rounds=2, doctor_configs=doctor_configs,
                moderator_model_key=args.moderator_model, logger=logger
            )
            question_data = item["question"]
            result = await mdt.run_consultation(
                qid=item["qid"],
                question=question_data[0] if isinstance(question_data, list) else question_data,
                task_type=args.task, label=item.get("ground_truth")
            )

            item_result = {
                "qid": item["qid"], "question": item["question"], "ground_truth": item.get("ground_truth"),
                "predicted_value": result.get("final_decision", {}).get("prediction", 0.501),
                "case_history": result, "processing_time": time.time() - start_time,
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