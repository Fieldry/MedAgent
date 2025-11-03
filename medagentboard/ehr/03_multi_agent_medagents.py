# ehr_multi_agent_medagents.py

import os
import json
import time
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any, List

from openai import OpenAI

from medagentboard.utils import prompt_template
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import get_logger, preprocess_response_string, save_json, load_json, parse_structured_output, parse_structured_output_for_final_report

class MedicalSpecialty(Enum):
    """Enumeration for medical specialties in EHR analysis."""
    CRITICAL_CARE = "Critical Care Medicine"
    CARDIOLOGY = "Cardiology"
    PULMONOLOGY = "Pulmonology"
    INFECTIOUS_DISEASE = "Infectious Disease"
    NEPHROLOGY = "Nephrology"
    HEMATOLOGY = "Hematology"
    ENDOCRINOLOGY = "Endocrinology"

class AgentType(Enum):
    """Enumeration for agent types."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    DECISION_MAKER = "Decision Maker"
    EXPERT_GATHERER = "Expert Gatherer"
    EVALUATOR = "Evaluator"

class BaseAgent:
    """Base class for all agents in the EHR prediction framework."""

    def __init__(self, agent_id: str, agent_type: AgentType, model_key: str = "deepseek-v3-official"):
        """Initializes the base agent."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not found in LLM_MODELS_SETTINGS")

        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(api_key=model_settings["api_key"], base_url=model_settings["base_url"])
        self.model_name = model_settings["model_name"]

    def call_llm(self, system_message: Dict[str, str], user_message: Dict[str, Any], max_retries: int = 3) -> str:
        """Calls the language model with messages and handles retries."""
        retries = 0
        while retries < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM, attempt {retries+1}/{max_retries}")
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=[system_message, user_message], response_format={"type": "json_object"}
                )
                response = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)

class ExpertGathererAgent(BaseAgent):
    """Agent for gathering domain experts based on EHR data."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        """Initializes the expert gatherer agent."""
        super().__init__(agent_id, AgentType.EXPERT_GATHERER, model_key)
        print(f"Initializing EHR expert gatherer agent, ID: {agent_id}, Model: {model_key}")

    def gather_ehr_domain_experts(self, question: str, task_type: str) -> List[MedicalSpecialty]:
        """Gathers relevant domain experts for EHR time-series analysis."""
        print(f"Expert gatherer {self.agent_id} identifying specialists for EHR {task_type} prediction")
        system_message = {
            "role": "system",
            "content": f"You are a medical coordinator. Analyze the clinical features in the EHR to determine the three most relevant medical specialties for a {task_type} prediction task. Output a JSON with a 'fields' array containing the three specialties."
        }
        user_message = { "role": "user", "content": f"Review this EHR data and determine the three most appropriate medical specialties for {task_type} prediction:\n\n{question}" }
        response_text = self.call_llm(system_message, user_message)

        try:
            specialties = json.loads(preprocess_response_string(response_text)).get("fields", [])
            valid_specialties = []
            specialty_map = {
                "critical": MedicalSpecialty.CRITICAL_CARE, "cardio": MedicalSpecialty.CARDIOLOGY,
                "pulmon": MedicalSpecialty.PULMONOLOGY, "infect": MedicalSpecialty.INFECTIOUS_DISEASE,
                "nephro": MedicalSpecialty.NEPHROLOGY, "hemat": MedicalSpecialty.HEMATOLOGY,
                "endocrin": MedicalSpecialty.ENDOCRINOLOGY
            }
            for spec in specialties:
                for key, value in specialty_map.items():
                    if key in spec.lower():
                        valid_specialties.append(value)
                        break

            # Remove duplicates and ensure exactly 3 specialties
            unique_specialties = list(dict.fromkeys(valid_specialties))
            default_specialties = [MedicalSpecialty.CRITICAL_CARE, MedicalSpecialty.CARDIOLOGY, MedicalSpecialty.PULMONOLOGY]
            while len(unique_specialties) < 3:
                for default_spec in default_specialties:
                    if default_spec not in unique_specialties:
                        unique_specialties.append(default_spec)
            return unique_specialties[:3]
        except json.JSONDecodeError:
            print("Expert gatherer response is not valid JSON, using default specialties")
            return [MedicalSpecialty.CRITICAL_CARE, MedicalSpecialty.CARDIOLOGY, MedicalSpecialty.PULMONOLOGY]

class DoctorAgent(BaseAgent):
    """Doctor agent specialized in analyzing EHR time-series data."""

    def __init__(self, agent_id: str, specialty: MedicalSpecialty, model_key: str = "deepseek-v3-official"):
        """Initializes a doctor agent for EHR analysis."""
        super().__init__(agent_id, AgentType.DOCTOR, model_key)
        self.specialty = specialty
        print(f"Initializing {specialty.value} doctor agent, ID: {agent_id}, Model: {model_key}")

    def analyze_ehr(self, question: str, task_type: str) -> Dict[str, Any]:
        """Analyzes EHR time-series data for clinical prediction."""
        print(f"Doctor {self.agent_id} ({self.specialty.value}) analyzing EHR data.")
        system_message = {
            "role": "system",
            "content": f"You are a specialist in {self.specialty.value}. Analyze the provided time-series EHR data to predict patient {task_type}. Focus on patterns relevant to your specialty. Output a JSON with 'explanation' and 'answer' (a probability from 0 to 1)."
        }
        user_message = {"role": "user", "content": f"{question}\n\nAs a {self.specialty.value} specialist, provide your prediction in the required JSON format."}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            answer = result.get('answer', 0.5)
            result['answer'] = max(0.0, min(1.0, float(answer)))
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            answer = result.get('answer', 0.5)
            result['answer'] = max(0.0, min(1.0, float(answer)))
        return result

    def review_synthesis(self, synthesis: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Reviews the meta agent's synthesis of EHR analysis."""
        print(f"Doctor {self.agent_id} ({self.specialty.value}) reviewing EHR synthesis.")
        system_message = {
            "role": "system",
            "content": f"You are a {self.specialty.value} specialist. Review the synthesized EHR analysis. Output a JSON with 'agree' (boolean), 'reason' (string), and optionally 'answer' (probability) if you disagree."
        }
        user_message = {"role": "user", "content": f"Synthesized analysis for {task_type}:\n{synthesis.get('explanation', '')}\n\nDo you agree? Provide your response in JSON format."}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            result["agree"] = str(result.get("agree", "false")).lower() in ["true", "yes"]
            if 'answer' in result:
                result['answer'] = max(0.0, min(1.0, float(result['answer'])))
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
        return result

class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple specialists' EHR analyses."""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent for EHR synthesis, ID: {agent_id}, Model: {model_key}")

    def synthesize_ehr_analyses(self, doctor_opinions: List[Dict[str, Any]], doctor_specialties: List[MedicalSpecialty], task_type: str, current_round: int = 1) -> Dict[str, Any]:
        """Synthesizes multiple specialists' analyses of EHR data."""
        print(f"Meta agent synthesizing round {current_round} EHR analyses.")
        system_message = {"role": "system", "content": f"You are a clinical coordinator. Synthesize the following specialists' analyses of time-series EHR data for a {task_type} prediction. Create a comprehensive summary in the 'explanation' field. DO NOT provide a probability prediction. Output JSON with ONLY the 'explanation' field."}

        opinions_text = "\n".join([f"Specialist {i+1} ({specialty.value}):\nExplanation: {opinion.get('explanation', '')}\nProbability: {opinion.get('answer', 'N/A')}" for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties))])
        user_message = {"role": "user", "content": f"Round {current_round} Analyses:\n{opinions_text}\n\nSynthesize these into a coherent summary in the required JSON format."}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            if "answer" in result: del result["answer"]
            if "explanation" not in result: result["explanation"] = "No explanation provided."
        except json.JSONDecodeError:
            result = {"explanation": response_text.strip()}
        return result

class DecisionMakingAgent(BaseAgent):
    """Decision making agent that outputs the final probability prediction."""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        super().__init__(agent_id, AgentType.DECISION_MAKER, model_key)
        print(f"Initializing decision making agent, ID: {agent_id}, Model: {model_key}")

    def make_prediction(self, question: str, synthesis: Dict[str, Any], doctor_opinions: List[Dict[str, Any]], task_type: str) -> Dict[str, Any]:
        """Makes a final probability prediction based on synthesized analyses."""
        print(f"Decision making agent generating final {task_type} probability prediction.")
        system_message = {"role": "system", "content": f"You are a clinical decision-making agent. Based on the specialists' analyses and the synthesized report, provide the final probability prediction for patient {task_type}. Output a JSON with 'explanation' and 'answer' (a float between 0 and 1)."}

        opinions_text = "\n".join([f"Specialist {i+1} probability: {op.get('answer', 'N/A')}" for i, op in enumerate(doctor_opinions)])
        user_message = {"role": "user", "content": f"EHR Data:\n{question}\n\nSynthesized Analysis:\n{synthesis.get('explanation', '')}\n\nSpecialist Predictions:\n{opinions_text}\n\nProvide your final probability prediction in the required JSON format."}
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            answer = result.get('answer', 0.5)
            result['answer'] = max(0.0, min(1.0, float(answer)))
        except (json.JSONDecodeError, ValueError, TypeError):
            result = parse_structured_output(response_text)
            valid_opinions = [op.get('answer') for op in doctor_opinions if isinstance(op.get('answer'), (int, float))]
            result['answer'] = sum(valid_opinions) / len(valid_opinions) if valid_opinions else 0.5
        return result

class EvaluateAgent(BaseAgent):
    """Evaluator Agent for assessing report quality."""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        super().__init__(agent_id, AgentType.EVALUATOR, model_key)

    def evaluate_final_report(self, original_question: str, final_report_explanation: str, final_report_prediction: float, task_type: str, label: str) -> Dict[str, Any]:
        """Evaluates the AI-generated final patient report for trustworthiness."""
        system_message = {"role": "system", "content": prompt_template.REPORT_EVALUATOR_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.REPORT_EVALUATOR_USER.format(original_question=original_question, final_report=final_report_explanation, final_explanation=final_report_explanation, final_prediction=final_report_prediction, task_type=task_type, true_label=label)}
        response_text = self.call_llm(system_message, user_message)
        try:
            result = json.loads(preprocess_response_string(response_text))
            for dim in ["accuracy", "explainability", "safety"]:
                score = result.get(dim, {}).get("score", 1)
                result[dim]["score"] = max(1, min(5, int(score)))
        except (json.JSONDecodeError, ValueError, TypeError):
            result = parse_structured_output_for_final_report(response_text)
        return result

class MDTConsultation:
    """Orchestrates the multi-disciplinary team consultation for EHR prediction."""

    def __init__(self, max_rounds: int = 2, model_key: str = "deepseek-v3-official", meta_model_key: str = "deepseek-v3-official", decision_model_key: str = "deepseek-v3-official", evaluator_model_key: str = "deepseek-v3-official"):
        """Initializes the MDT consultation process."""
        self.max_rounds = max_rounds
        self.expert_gatherer = ExpertGathererAgent("expert_gatherer", model_key)
        self.meta_agent = MetaAgent("meta", meta_model_key)
        self.decision_agent = DecisionMakingAgent("decision", decision_model_key)
        self.evaluator_agent = EvaluateAgent("evaluator", evaluator_model_key)
        self.doctor_agents = []
        self.doctor_specialties = []
        self.model_key = model_key
        print(f"Initialized MDT consultation for EHR prediction, max_rounds={max_rounds}")

    def _initialize_doctor_agents(self, specialties: List[MedicalSpecialty]):
        """Initializes doctor agents with the given specialties."""
        self.doctor_agents = [DoctorAgent(f"doctor_{i+1}", specialty, self.model_key) for i, specialty in enumerate(specialties)]
        self.doctor_specialties = specialties

    def run_consultation(self, qid: str, question: List[str], task_type: str, label: str) -> Dict[str, Any]:
        """Runs the full MDT consultation process."""
        print(f"Starting MDT consultation for EHR case {qid}, task: {task_type}")
        specialties = self.expert_gatherer.gather_ehr_domain_experts(question[-1], task_type)
        self._initialize_doctor_agents(specialties)

        case_history = {"qid": qid, "task_type": task_type, "selected_specialties": [s.value for s in specialties], "rounds": []}
        final_decision = None
        consensus_reached = False

        for current_round in range(1, self.max_rounds + 1):
            print(f"Starting round {current_round}")
            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": []}

            doctor_opinions = [doc.analyze_ehr(q, task_type) for doc, q in zip(self.doctor_agents, question)]
            round_data["opinions"] = [{"doctor_id": doc.agent_id, "specialty": doc.specialty.value, "opinion": op} for doc, op in zip(self.doctor_agents, doctor_opinions)]

            synthesis = self.meta_agent.synthesize_ehr_analyses(doctor_opinions, self.doctor_specialties, task_type, current_round)
            round_data["synthesis"] = synthesis

            all_agree = all(doc.review_synthesis(synthesis, task_type).get('agree', False) for doc in self.doctor_agents)

            case_history["rounds"].append(round_data)

            if all_agree or current_round == self.max_rounds:
                print("Consensus reached or max rounds hit. Making final decision.")
                final_decision = self.decision_agent.make_prediction(question[-1], synthesis, doctor_opinions, task_type)
                consensus_reached = all_agree
                break

        final_report_evaluation = self.evaluator_agent.evaluate_final_report(
            original_question=question[-1], final_report_explanation=final_decision.get('explanation', ''),
            final_report_prediction=final_decision.get('answer', 0.5), task_type=task_type, label=label
        )
        case_history.update({"final_decision": final_decision, "consensus_reached": consensus_reached, "total_rounds": len(case_history["rounds"]), "report_trustworthiness_evaluation": final_report_evaluation})
        return case_history

def main():
    parser = argparse.ArgumentParser(description="Run MedAgents multi-agent EHR predictions")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["cdsl", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "sptb"])
    parser.add_argument("--model", type=str, default="deepseek-v3-official", help="Model for doctor agents")
    parser.add_argument("--meta_model", type=str, default="deepseek-v3-official", help="Model for meta agent")
    parser.add_argument("--decision_model", type=str, default="deepseek-v3-official", help="Model for decision-making agent")
    parser.add_argument("--evaluator_model", type=str, default="deepseek-v3-official", help="Model for evaluator agent")
    parser.add_argument("--modality", type=str, default="ehr")
    args = parser.parse_args()

    # Setup directories and paths
    method = "MedAgents"
    logs_dir = os.path.join("logs", "ehr", args.dataset, args.task, method)
    os.makedirs(logs_dir, exist_ok=True)
    data_path = f"./my_datasets/ehr/{args.dataset}/processed/{args.modality}_{args.task}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Main processing loop
    for item in tqdm(data, desc=f"Running EHR predictions on {args.dataset}/{args.task}"):
        qid = item["qid"]
        result_path = os.path.join(logs_dir, f"ehr_{qid}-result.json")
        if os.path.exists(result_path):
            print(f"Skipping {qid} - already processed")
            continue

        try:
            start_time = time.time()
            task_type = args.task
            mdt = MDTConsultation(
                model_key=args.model, meta_model_key=args.meta_model,
                decision_model_key=args.decision_model, evaluator_model_key=args.evaluator_model
            )
            result = mdt.run_consultation(
                qid=qid, question=item["question"], task_type=task_type, label=item.get("ground_truth")
            )

            item_result = {
                "qid": qid, "question": item["question"], "ground_truth": item.get("ground_truth"),
                "predicted_value": result["final_decision"]["answer"],
                "case_history": result, "processing_time": time.time() - start_time,
                "timestamp": int(time.time())
            }
            save_json(item_result, result_path)
            print(f"Saved result for {qid} to {result_path}")
        except Exception as e:
            print(f"Error processing item {qid}: {e}")

if __name__ == "__main__":
    main()