# ehr_multi_agent_colacare.py

import os
import json
import time
import asyncio
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any, List

from openai import OpenAI

from medagentboard.utils import prompt_template
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import get_logger, preprocess_response_string, save_json, load_json, parse_structured_output, parse_structured_output_for_final_report
from medagentboard.utils.litsense_utils import litsense_api_call


class AgentType(Enum):
    """Enumeration for agent types."""
    DOCTOR = "Doctor"
    META = "Coordinator"
    EVALUATOR = "Evaluator"


class BaseAgent:
    """Base class for all agents in the EHR prediction framework."""

    def __init__(self,
        agent_id: str,
        agent_type: AgentType,
        model_key: str = "deepseek-v3-official",
        logger=None):
        """
        Initializes the base agent.

        Args:
            agent_id: A unique identifier for the agent.
            agent_type: The type of the agent (e.g., Doctor, Coordinator).
            model_key: The key for the LLM model to be used.
            logger: A logger object for logging activities.
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []
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
        Calls the language model with the provided messages and handles retries.

        Args:
            system_message: The system message to set the context for the LLM.
            user_message: The user message containing the primary request or data.
            max_retries: The maximum number of retry attempts in case of an API error.

        Returns:
            The text content of the LLM's response.
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
                time.sleep(1)  # Brief pause before retrying


class DoctorAgent(BaseAgent):
    """Doctor agent with a clinical specialty for EHR predictive modeling."""

    def __init__(self,
        agent_id: str,
        specialty: str,
        model_key: str = "deepseek-v3-official",
        logger=None,
        use_rag: bool = True):
        """
        Initializes a doctor agent.

        Args:
            agent_id: A unique identifier for the doctor.
            specialty: The doctor's clinical specialty as a string.
            model_key: The LLM model to be used by this agent.
            logger: A logger object for logging.
            use_rag: A boolean indicating whether to use Retrieval-Augmented Generation.
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key, logger=logger)
        self.specialty = specialty
        self.use_rag = use_rag
        if self.logger:
            self.logger.info(f"Initializing doctor agent, ID: {agent_id}, Specialty: {specialty}, Model: {model_key}")

    def _generate_rag_query(self, question: str, task_type: str) -> List[str]:
        """
        Generates a search query for the RAG tool based on the EHR data and task.
        """
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} generating RAG query for task: {task_type}")

        system_message = {
            "role": "system",
            "content": prompt_template.RAG_QUERY_GENERATION_SYSTEM
        }
        user_message = {
            "role": "user",
            "content": prompt_template.RAG_QUERY_GENERATION_USER.format(
                question_short=question[:2000],
                task_type=task_type
            )
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            query_text = result.get("query", [])
        except json.JSONDecodeError:
            query_text = []

        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} generated RAG query: {query_text}")
        return query_text

    async def analyze_case(self,
        question: str,
        task_type: str) -> Dict[str, Any]:
        """
        Analyzes an EHR case and predicts an outcome probability, with optional RAG retrieval.

        Args:
            question: A string containing structured EHR time series data.
            task_type: The type of task (e.g., 'mortality', 'readmission').

        Returns:
            A dictionary containing the analysis results and prediction.
        """
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) analyzing case with model: {self.model_key}")

        task_hints = {
            "mortality": prompt_template.TASK_HINT_MORTALITY,
            "readmission": prompt_template.TASK_HINT_READMISSION,
            "sptb": prompt_template.TASK_HINT_SPTB
        }
        task_hint = task_hints.get(task_type, "")

        rag_query = []
        retrieved_literature = ""
        if self.use_rag:
            rag_query = self._generate_rag_query(question, task_type)
            retrieved_literature = "Retrieved literature:\n"
            for i, query in enumerate(rag_query):
                retrieved_literature += await litsense_api_call(query=query, order=i, max_results=5) + "\n"
            if self.logger:
                self.logger.info(f"Doctor {self.agent_id} retrieved literature (first 200 chars): {retrieved_literature[:200]}...")

        system_message = {
            "role": "system",
            "content": prompt_template.DOCTOR_ANALYZE_SYSTEM.format(specialty=self.specialty, task_hint=task_hint)
        }
        user_message = {
            "role": "user",
            "content": prompt_template.DOCTOR_ANALYZE_USER.format(question=question, retrieved_literature=retrieved_literature)
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info(f"Doctor {self.agent_id} response successfully parsed.")

            pred = result.get("prediction", 0.501)
            result["prediction"] = max(0.0, min(1.0, float(pred)))

        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing.")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

        result["system_message"] = system_message["content"]
        result["user_message"] = user_message["content"]
        result["rag_query"] = rag_query
        result["retrieved_literature"] = retrieved_literature

        self.memory.append({"type": "analysis", "content": result, "round": 0})
        return result

    def review_synthesis(self,
        question: str,
        synthesis: Dict[str, Any],
        task_type: str,
        current_round: int = 1) -> Dict[str, Any]:
        """
        Reviews the meta agent's synthesis of opinions.

        Args:
            question: The original question with EHR data.
            synthesis: The meta agent's synthesized report and prediction.
            task_type: The type of prediction task.
            current_round: The current round of discussion.

        Returns:
            A dictionary with an agreement status and a potential rebuttal.
        """
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) reviewing synthesis in round {current_round}.")

        own_analysis = next((mem["content"] for mem in reversed(self.memory) if mem["type"] == "analysis"), None)

        task_hints = {
            "mortality": prompt_template.TASK_HINT_REVIEW_MORTALITY,
            "readmission": prompt_template.TASK_HINT_REVIEW_READMISSION,
            "sptb": prompt_template.TASK_HINT_REVIEW_SPTB
        }
        task_hint = task_hints.get(task_type, "")

        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nPrediction: {own_analysis.get('prediction', '')}\n\n"

        synthesis_text = f"Synthesized report:\n{synthesis.get('report', '')}\n"
        synthesis_text += f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested prediction: {synthesis.get('prediction', '')}"

        system_message = {
            "role": "system",
            "content": prompt_template.DOCTOR_REVIEW_SYSTEM.format(specialty=self.specialty, current_round=current_round, task_hint=task_hint)
        }
        user_message = {
            "role": "user",
            "content": prompt_template.DOCTOR_REVIEW_USER.format(question_short=question, own_analysis_text=own_analysis_text, synthesis_text=synthesis_text)
        }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info(f"Doctor {self.agent_id} review successfully parsed")

            result["agree"] = str(result.get("agree", "false")).lower() in ["true", "yes"]
            pred = result.get("prediction", 0.501)
            result["prediction"] = max(0.0, min(1.0, float(pred)))

        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

        result["system_message"] = system_message["content"]
        result["user_message"] = user_message["content"]

        self.memory.append({"type": "review", "round": current_round, "content": result})
        return result


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions for EHR prediction."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        """
        Initialize a meta agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
            logger: Logger object for logging
        """
        super().__init__(agent_id, AgentType.META, model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
        question: str,
        doctor_opinions: List[Dict[str, Any]]=None,
        doctor_reviews: List[Dict[str, Any]]=None,
        task_type: str = "mortality",
        current_round: int = 0) -> Dict[str, Any]:
        """
        Synthesizes opinions from multiple doctors.

        Args:
            question: The original question with EHR data.
            doctor_opinions: A list of initial opinions from doctor agents (for round 0).
            doctor_reviews: A list of reviews from doctor agents (for subsequent rounds).
            task_type: The type of prediction task.
            current_round: The current round of discussion.

        Returns:
            A dictionary containing the synthesized explanation and prediction.
        """
        if self.logger:
            self.logger.info(f"Meta agent synthesizing opinions with model: {self.model_key} in round {current_round}")

        task_hints = {
            "mortality": prompt_template.TASK_HINT_MORTALITY,
            "readmission": prompt_template.TASK_HINT_READMISSION,
            "sptb": prompt_template.TASK_HINT_SPTB
        }
        task_hint = task_hints.get(task_type, "")

        if current_round == 0:
            opinions_text = "\n".join([
                f"Doctor {i+1}:\nExplanation: {opinion.get('opinion', {}).get('explanation', '')}\nPrediction: {opinion.get('opinion', {}).get('prediction', '')}" for i, opinion in enumerate(doctor_opinions)
            ])
            system_message = {
                "role": "system",
                "content": prompt_template.META_SYNTHESIZE_SYSTEM.format(task_hint=task_hint)
            }
            user_message = {
                "role": "user",
                "content": prompt_template.META_SYNTHESIZE_USER.format(question_short=question, opinions_text=opinions_text)
            }
        else:
            prev_synthesis = next((mem["content"] for mem in reversed(self.memory) if mem["type"] == "synthesis" and mem["round"] < current_round), None)
            prev_synthesis_str = f"{prev_synthesis.get('report', '')}\n\nExplanation: {prev_synthesis.get('explanation', '')}\nPrediction: {prev_synthesis.get('prediction', '')}" if prev_synthesis else "No previous syntheses available.\n\n"

            doctor_reviews_str = "\n".join([
                f"Doctor {i+1}:\n" +
                f"Agree: {'Yes' if review.get('review', {}).get('agree', False) else 'No'}\n" +
                f"Reason: {review.get('review', {}).get('reason', '')}\n" +
                f"Prediction: {review.get('review', {}).get('prediction', '')}"
                for i, review in enumerate(doctor_reviews or [])
            ])
            system_message = {
                "role": "system",
                "content": prompt_template.META_RESYNTHESIZE_SYSTEM.format(current_round=current_round, task_hint=task_hint)
            }
            user_message = {
                "role": "user",
                "content": prompt_template.META_RESYNTHESIZE_USER.format(question_short=question, prev_synthesis=prev_synthesis_str, doctor_reviews=doctor_reviews_str)
            }

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info("Meta agent synthesis successfully parsed")
            pred = result.get("prediction", 0.501)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

        result["system_message"] = system_message["content"]
        result["user_message"] = user_message["content"]

        self.memory.append({"type": "synthesis", "round": current_round, "content": result})
        return result


class EvaluateAgent(BaseAgent):
    """
    Evaluator Agent for assessing report quality.
    1. Evaluates the quality of each DoctorAgent's preliminary report.
    2. Evaluates the final report based on Factuality, Safety, and Explainability.
    """
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        super().__init__(agent_id, AgentType.EVALUATOR, model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing evaluator agent, ID: {agent_id}, Model: {model_key}")

    def evaluate_preliminary_report(self, doctor_report: Dict[str, Any], final_report: Dict[str, Any], question: str, task_type: str) -> Dict[str, Any]:
        """
        Evaluates the quality of a DoctorAgent's preliminary report against the final report.
        """
        system_message = { "role": "system", "content": prompt_template.EVALUATE_SYSTEM }
        user_message = {
            "role": "user",
            "content": prompt_template.EVALUATE_USER.format(question_short=question, doctor_explanation=doctor_report.get('explanation', ''), doctor_prediction=doctor_report.get('prediction', ''), final_explanation=final_report.get('explanation', ''), final_prediction=final_report.get('prediction', ''), task_type=task_type)
        }
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info("Evaluator agent successfully parsed")
            score = result.get("score", 0.0)
            result["score"] = max(0.0, min(10.0, float(score)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning("Evaluator agent response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)

        result["system_message"] = system_message["content"]
        result["user_message"] = user_message["content"]
        self.memory.append({"type": "evaluation", "content": result})
        return result

    def evaluate_final_report(self,
        original_question: str,
        final_report_str: str,
        final_report_explanation: str,
        final_report_prediction: float,
        task_type: str,
        label: str) -> Dict[str, Any]:
        """
        Evaluate the AI-generated final patient report for trustworthiness dimensions.

        Args:
            original_question: The complete original input (EHR data + initial model predictions).
            final_report_str: The final generated report.
            final_report_explanation: The 'explanation' part of the final generated report.
            final_report_prediction: The 'prediction' part of the final generated report.
            task_type: Type of task (mortality, readmission or sptb).
            label: True label of the patient under the task.

        Returns:
            Dictionary containing evaluation scores and reasons for Factuality, Safety,
            and Explainability, plus an overall comment.
        """
        if self.logger:
            self.logger.info(f"Report evaluation agent evaluating report for task '{task_type}'.")

        system_message = { "role": "system", "content": prompt_template.REPORT_EVALUATOR_SYSTEM }
        user_message = {
            "role": "user",
            "content": prompt_template.REPORT_EVALUATOR_USER.format(
                original_question=original_question,
                final_report=final_report_str,
                final_explanation=final_report_explanation,
                final_prediction=final_report_prediction,
                task_type=task_type,
                true_label=label
            )
        }
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            for dim in ["accuracy", "explainability", "safety"]:
                score = result.get(dim, {}).get("score", 1)
                result[dim]["score"] = max(1, min(5, int(score)))
            if self.logger:
                self.logger.info("Report evaluation agent response successfully parsed.")
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger:
                self.logger.warning("Report evaluation agent response is not valid JSON, attempting fallback parsing.")
            result = parse_structured_output_for_final_report(response_text)

        result["system_message"] = system_message["content"]
        result["user_message"] = user_message["content"]
        result["response_text"] = response_text
        return result


class MDTConsultation:
    """Coordinator for a Multi-Disciplinary Team (MDT) consultation on EHR data."""

    def __init__(self,
        max_rounds: int = 3,
        doctor_configs: List[Dict] = None,
        meta_model_key: str = "deepseek-v3-official",
        evaluator_model_key: str = "deepseek-v3-official",
        logger=None,
        use_rag: bool = True):
        """
        Initializes the MDT consultation process.

        Args:
            max_rounds: The maximum number of discussion rounds.
            doctor_configs: A list of configurations for each doctor agent.
            meta_model_key: The model key for the meta agent.
            evaluator_model_key: The model key for the evaluator agent.
            logger: A logger object for logging.
            use_rag: A boolean to enable or disable RAG for doctor agents.
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"} for _ in range(3)
        ]
        self.logger = logger

        # Initialize agents
        self.doctor_agents = [
            DoctorAgent(f"doctor_{i+1}", cfg["specialty"], cfg["model_key"], logger=logger, use_rag=use_rag)
            for i, cfg in enumerate(self.doctor_configs)
        ]
        self.meta_agent = MetaAgent("meta", meta_model_key, logger=logger)
        self.evaluator_agent = EvaluateAgent("evaluator", evaluator_model_key, logger=logger)

        if self.logger:
            doctor_info = ", ".join([f"{cfg['specialty']} ({cfg['model_key']})" for cfg in self.doctor_configs])
            self.logger.info(f"Initialized MDT consultation, max_rounds={max_rounds}, doctors: [{doctor_info}], meta_model={meta_model_key}")

    async def run_consultation(self,
        qid: str,
        question: List[str],
        task_type: str = "mortality",
        label: str = None) -> Dict[str, Any]:
        """
        Runs the full MDT consultation process for a given case.

        Args:
            qid: The unique identifier for the case/question.
            question: A list of questions (one for each doctor) containing EHR data.
            task_type: The type of prediction task.
            label: The ground truth label for the case.

        Returns:
            A dictionary containing the complete history and final result of the consultation.
        """
        if self.logger:
            self.logger.info(f"Starting MDT consultation for case {qid} (Task: {task_type}, Label: {label})")

        case_history = {"opinions": [], "synthesis": None, "rounds": []}
        final_decision = None
        consensus_reached = False

        # Step 1: Each doctor provides an initial analysis
        doctor_opinions = []
        for i, doctor in enumerate(self.doctor_agents):
            if self.logger:
                self.logger.info(f"Doctor {i+1} ({doctor.specialty}) analyzing case")
            opinion = await doctor.analyze_case(question[i], task_type)
            doctor_opinions.append({"doctor_id": doctor.agent_id, "specialty": doctor.specialty, "opinion": opinion})
            if self.logger:
                self.logger.info(f"Doctor {i+1} prediction: {opinion.get('prediction', 'N/A')}")
        case_history["opinions"] = doctor_opinions

        # Step 2: Meta agent synthesizes initial opinions
        if self.logger:
            self.logger.info("Meta agent synthesizing initial opinions")
        synthesis = self.meta_agent.synthesize_opinions(question[-1], doctor_opinions=doctor_opinions, task_type=task_type, current_round=0)
        case_history["synthesis"] = synthesis
        if self.logger:
            self.logger.info(f"Meta agent initial synthesis prediction: {synthesis.get('prediction', 'N/A')}")

        # Step 3: Iterative review and re-synthesis process
        for current_round in range(1, self.max_rounds + 1):
            if self.logger:
                self.logger.info(f"Starting discussion round {current_round}")
            round_data = {"round": current_round, "reviews": [], "synthesis": None}

            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                review = doctor.review_synthesis(question[i], synthesis, task_type, current_round)
                doctor_reviews.append({"doctor_id": doctor.agent_id, "specialty": doctor.specialty, "review": review})
                all_agree = all_agree and review.get('agree', False)
            round_data["reviews"] = doctor_reviews

            synthesis = self.meta_agent.synthesize_opinions(question[-1], doctor_reviews=doctor_reviews, task_type=task_type, current_round=current_round)
            round_data["synthesis"] = synthesis
            case_history["rounds"].append(round_data)

            if all_agree:
                self.logger.info("Consensus reached.")
                consensus_reached = True
                final_decision = synthesis
                break

        if not final_decision:
            self.logger.info("Max rounds reached, using final synthesis as decision.")
            final_decision = synthesis

        if self.logger:
            self.logger.info(f"Final prediction: {final_decision.get('prediction', 'N/A')}")

        # Step 4: Evaluate preliminary and final reports
        doctor_scores = []
        for i, doctor in enumerate(self.doctor_agents):
            first_analysis = next((mem["content"] for mem in doctor.memory if mem["type"] == "analysis"), {})
            score_result = self.evaluator_agent.evaluate_preliminary_report(first_analysis, final_decision, question[i], task_type)
            doctor_scores.append({"doctor_id": doctor.agent_id, "specialty": doctor.specialty, **score_result})

        final_report_evaluation = self.evaluator_agent.evaluate_final_report(
            original_question=question[-1],
            final_report_str=final_decision.get('report', ''),
            final_report_explanation=final_decision.get('explanation', ''),
            final_report_prediction=final_decision.get('prediction', 0.501),
            task_type=task_type,
            label=label
        )

        case_history.update({
            "final_decision": final_decision,
            "consensus_reached": consensus_reached,
            "total_rounds": len(case_history["rounds"]) + 1,
            "doctor_scores": doctor_scores,
            "report_trustworthiness_evaluation": final_report_evaluation
        })
        return case_history


async def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on EHR datasets using ColaCare framework")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "cdsl", "esrd", "obstetrics"], help="Specify dataset name")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "sptb"], help="Prediction task")
    parser.add_argument("--modality", "-mo", type=str, default="ehr", choices=["ehr", "note", "mm"], help="Modality of the dataset")
    parser.add_argument("--meta_model", type=str, default="deepseek-v3-official", help="Model for meta agent")
    parser.add_argument("--doctor_models", nargs='+', default=["deepseek-v3-official", "deepseek-v3-official", "deepseek-v3-official"], help="Models for doctor agents")
    parser.add_argument("--evaluate_model", type=str, default="deepseek-v3-official", help="Model for evaluator agent")
    parser.add_argument("--use_rag", action="store_true", default=False, help="Enable Retrieval-Augmented Generation for doctor agents")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of the data chunk to process")
    args = parser.parse_args()

    # Setup directories and paths
    method = "ColaCare"
    save_dir = os.path.join("logs", args.dataset, args.task, method, f"{args.modality}_{args.meta_model}")
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

    # Configure doctors' specialties based on dataset
    specialties_map = {
        "esrd": "End-Stage Renal Disease", "cdsl": "COVID-19",
        "mimic-iv": "Intensive Care", "obstetrics": "Obstetrics"
    }
    dataset_specialty = specialties_map.get(args.dataset, "General Medicine")
    doctor_configs = [{"model_key": model, "specialty": dataset_specialty} for model in args.doctor_models]
    print(f"Configuring {len(doctor_configs)} doctors with specialty: {dataset_specialty}")

    # Main processing loop
    for item in tqdm(data_to_process, desc=f"Running MDT on {args.dataset} chunk"):
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
                meta_model_key=args.meta_model, evaluator_model_key=args.evaluate_model,
                logger=logger, use_rag=args.use_rag
            )
            result = await mdt.run_consultation(
                qid=item["qid"], question=item["question"],
                task_type=args.task, label=item.get("ground_truth")
            )

            item_result = {
                "qid": item["qid"], "question": item["question"], "ground_truth": item.get("ground_truth"),
                "predicted_value": result["final_decision"]["prediction"],
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