"""
ehr_multi_agent_framework.py

Multi-agent LLM framework for EHR predictive modeling tasks.
This code adapts a medical QA multi-agent framework to process EHR time series data
and make predictions for mortality and readmission probability.
"""

import os
import json
import time
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any, List

from openai import OpenAI

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import load_json, save_json, preprocess_response_string


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
                 model_key: str = "deepseek-v3-official"):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (Doctor, Coordinator, Evaluator)
            model_key: LLM model to use
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        self.memory = []

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
                print(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:50]}...")
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    extra_body={"enable_thinking": False},
                    stream=True,
                )
                # Handle streaming response
                response_chunks = []
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        response_chunks.append(chunk.choices[0].delta.content)

                response = "".join(response_chunks)
                print(f"Agent {self.agent_id} received response: {response[:50]}...")
                return response
            except Exception as e:
                retries += 1
                print(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {e}")
                time.sleep(1)  # Brief pause before retrying


class DoctorAgent(BaseAgent):
    """Doctor agent with a clinical specialty for EHR predictive modeling."""

    def __init__(self,
                 agent_id: str,
                 specialty: str, # 新增 specialty 参数
                 model_key: str = "deepseek-v3-official"):
        """
        Initialize a doctor agent.

        Args:
            agent_id: Unique identifier for the doctor
            specialty: Doctor's clinical specialty (string)
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key)
        self.specialty = specialty # 设置 specialty
        print(f"Initializing doctor agent, ID: {agent_id}, Specialty: {specialty}, Model: {model_key}")

    def analyze_case(self,
                    question: str,
                    task_type: str) -> Dict[str, Any]:
        """
        Analyze an EHR case and predict outcome probability.

        Args:
            question: Question containing structured EHR time series data
            task_type: Type of task (mortality or readmission)

        Returns:
            Dictionary containing analysis results and prediction
        """
        print(f"Doctor {self.agent_id} ({self.specialty}) analyzing case with model: {self.model_key}")

        # Prepare system message to guide the doctor's analysis
        system_message = {
            "role": "system",
            "content": f"You are a physician specializing in {self.specialty}. " # 使用 self.specialty
                f"Analyze the provided time series EHR data and make a clinical prediction. "
                f"Your output should be in JSON format, including 'explanation' (detailed reasoning) and "
                f"'prediction' (a floating-point number between 0 and 1 representing probability) fields."
        }

        if task_type == "mortality":
            system_message["content"] += (
                f" For mortality prediction, your 'prediction' field should reflect the probability of the patient "
                f"not surviving their hospital stay (higher values indicate higher mortality risk)."
            )
        elif task_type == "readmission":
            system_message["content"] += (
                f" For readmission prediction, your 'prediction' field should reflect the probability of patient "
                f"readmission within 30 days post-discharge (higher values indicate higher readmission risk)."
            )

        # Prepare user message with the question
        user_message = {
            "role": "user",
            "content": f"{question}\n\nProvide your analysis in JSON format, including 'explanation' and 'prediction' fields."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} response successfully parsed")

            # Ensure prediction is a float between 0 and 1
            if "prediction" in result:
                try:
                    pred = float(result["prediction"])
                    result["prediction"] = max(0.0, min(1.0, pred))
                except ValueError: # Changed bare except to ValueError for specificity
                    result["prediction"] = 0.5  # Default fallback
            else:
                result["prediction"] = 0.5

            result["system_message"] = system_message["content"]
            result["user_message"] = user_message["content"]

            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            print(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

            # Add to memory
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result

    def review_synthesis(self,
                        question: str,
                        synthesis: Dict[str, Any],
                        task_type: str) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis.

        Args:
            question: Original question with EHR data
            synthesis: Meta agent's synthesis
            task_type: Type of task (mortality or readmission)

        Returns:
            Dictionary containing agreement status and possible rebuttal
        """
        print(f"Doctor {self.agent_id} ({self.specialty}) reviewing synthesis with model: {self.model_key}")

        # Get current round
        current_round = len(self.memory) // 2 + 1

        # Get doctor's own most recent analysis
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break

        # Prepare system message for review
        system_message = {
            "role": "system",
            "content": f"You are a physician specializing in {self.specialty}, participating in round {current_round} of a multidisciplinary team consultation. "
                f"Review the synthesis of multiple doctors' opinions and determine if you agree with the conclusion. "
                f"Consider your previous analysis and the Coordinator's synthesized opinion to decide whether to agree or provide a different perspective. "
                f"Your output should be in JSON format, including 'agree' (boolean or 'yes'/'no'), 'reason' (rationale for your decision), "
                f"and 'prediction' (your suggested prediction if you disagree; if you agree, you can repeat the synthesized prediction) fields."
        }

        # Add task-specific context
        if task_type == "mortality":
            system_message["content"] += (
                f" Remember, the prediction value should reflect the probability of mortality (higher values indicate higher mortality risk)."
            )
        elif task_type == "readmission":
            system_message["content"] += (
                f" Remember, the prediction value should reflect the probability of readmission within 30 days (higher values indicate higher readmission risk)."
            )

        # Prepare own previous analysis
        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nPrediction: {own_analysis.get('prediction', '')}\n\n"

        synthesis_text = f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested prediction: {synthesis.get('prediction', '')}"

        user_message = {
            "role": "user",
            "content": f"Original data and task: {question[:500]}...\n\n"
                f"{own_analysis_text}"
                f"{synthesis_text}\n\n"
                f"Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
                f"1. 'agree': 'yes'/'no'\n"
                f"2. 'reason': Your rationale for agreeing or disagreeing\n"
                f"3. 'prediction': Your supported prediction (can be the synthesized prediction if you agree, or your own suggested prediction if you disagree)"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print(f"Doctor {self.agent_id} review successfully parsed")

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]

            # Ensure prediction is a float between 0 and 1
            if "prediction" in result:
                try:
                    pred = float(result["prediction"])
                    result["prediction"] = max(0.0, min(1.0, pred))
                except ValueError: # Changed bare except to ValueError for specificity
                    result["prediction"] = 0.5  # Default fallback
            else:
                result["prediction"] = 0.5

            result["system_message"] = system_message["content"]
            result["user_message"] = user_message["content"]

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}

            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip().replace("\"", "") if ":" in line else line
                elif "prediction" in line.lower():
                    try:
                        pred_text = line.split(":", 1)[1].strip().replace("\"", "") if ":" in line else line
                        # Extract numeric prediction value
                        pred_value = float(''.join(c for c in pred_text if (c.isdigit() or c == '.')))
                        # Ensure within 0-1 range
                        pred_value = max(0.0, min(1.0, pred_value))
                        result["prediction"] = pred_value
                    except ValueError: # Changed bare except to ValueError for specificity
                        result["prediction"] = 0.5  # Default fallback

            # Ensure required fields
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "No reason provided"
            if "prediction" not in result:
                # Default to own previous prediction or synthesized prediction
                if own_analysis and "prediction" in own_analysis:
                    result["prediction"] = own_analysis["prediction"]
                else:
                    result["prediction"] = synthesis.get("prediction", 0.5)

            result["response_text"] = response_text

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result


class MetaAgent(BaseAgent):
    """Meta agent that synthesizes multiple doctors' opinions for EHR prediction."""

    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        """
        Initialize a meta agent.

        Args:
            agent_id: Unique identifier for the agent
            model_key: LLM model to use
        """
        super().__init__(agent_id, AgentType.META, model_key)
        print(f"Initializing meta agent, ID: {agent_id}, Model: {model_key}")

    def synthesize_opinions(self,
                           question: str,
                           doctor_opinions: List[Dict[str, Any]],
                           current_round: int = 1,
                           task_type: str = "mortality") -> Dict[str, Any]:
        """
        Synthesize multiple doctors' opinions.

        Args:
            question: Original question with EHR data
            doctor_opinions: List of doctor opinions
            current_round: Current discussion round
            task_type: Type of task (mortality or readmission)

        Returns:
            Dictionary containing synthesized explanation and prediction
        """
        print(f"Meta agent synthesizing round {current_round} opinions with model: {self.model_key}")

        # Prepare system message for synthesis
        system_message = {
            "role": "system",
            "content": f"You are a medical consensus coordinator facilitating round {current_round} of a multidisciplinary team consultation. "
                "Synthesize the opinions of multiple specialist doctors into a coherent analysis and conclusion. "
                "Consider each doctor's expertise and perspective, and weigh their opinions accordingly. "
                "Your output should be in JSON format, including 'explanation' (synthesized reasoning) and "
                "'prediction' (consensus probability value between 0 and 1) fields."
        }

        # Add task-specific context
        if task_type == "mortality":
            system_message["content"] += (
                f" For mortality prediction, the 'prediction' field should reflect the probability of the patient "
                f"not surviving their hospital stay (higher values indicate higher mortality risk)."
            )
        elif task_type == "readmission":
            system_message["content"] += (
                f" For readmission prediction, the 'prediction' field should reflect the probability of patient "
                f"readmission within 30 days post-discharge (higher values indicate higher readmission risk)."
            )

        # Format doctors' opinions as input
        formatted_opinions = []
        for i, opinion in enumerate(doctor_opinions):
            formatted_opinion = f"Doctor {i+1}:\n"
            formatted_opinion += f"Explanation: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"Prediction: {opinion.get('prediction', '')}\n"
            formatted_opinions.append(formatted_opinion)

        opinions_text = "\n".join(formatted_opinions)

        # Prepare user message with all opinions
        user_message = {
            "role": "user",
            "content": f"EHR data and task: {question[:500]}...\n\n"
                f"Round {current_round} Doctors' Opinions:\n{opinions_text}\n\n"
                f"Please synthesize these opinions into a consensus view. Provide your synthesis in JSON format, including "
                f"'explanation' (comprehensive reasoning) and 'prediction' (probability value between 0 and 1) fields."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent synthesis successfully parsed")

            # Ensure prediction is a float between 0 and 1
            if "prediction" in result:
                try:
                    pred = float(result["prediction"])
                    result["prediction"] = max(0.0, min(1.0, pred))
                except ValueError: # Changed bare except to ValueError for specificity
                    result["prediction"] = 0.5  # Default fallback
            else:
                result["prediction"] = 0.5

            result["system_message"] = system_message["content"]
            result["user_message"] = user_message["content"]

            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Meta agent synthesis is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result

    def make_final_decision(self,
                           question: str,
                           doctor_reviews: List[Dict[str, Any]],
                           current_synthesis: Dict[str, Any],
                           current_round: int,
                           max_rounds: int,
                           task_type: str = "mortality") -> Dict[str, Any]:
        """
        Make a final decision based on doctor reviews.

        Args:
            question: Original question with EHR data
            doctor_reviews: List of doctor reviews
            current_synthesis: Current synthesized result
            current_round: Current round
            max_rounds: Maximum number of rounds
            task_type: Type of task (mortality or readmission)

        Returns:
            Dictionary containing final explanation and prediction
        """
        print(f"Meta agent making round {current_round} decision with model: {self.model_key}")

        # Check if all doctors agree
        all_agree = all(review.get('agree', False) for review in doctor_reviews)
        reached_max_rounds = current_round >= max_rounds

        # Prepare system message for final decision
        system_message = {
            "role": "system",
            "content": "You are a medical consensus coordinator making a final decision. "
        }

        if all_agree:
            system_message["content"] += "All doctors agree with your synthesis, generate a final report."
        elif reached_max_rounds:
            system_message["content"] += (
                f"Maximum number of discussion rounds ({max_rounds}) reached without full consensus. "
                f"Make a final decision using majority opinion approach."
            )
        else:
            system_message["content"] += (
                "Not all doctors agree with your synthesis, but a decision for the current round is needed."
            )

        system_message["content"] += (
            " Your output should be in JSON format, including 'explanation' (final reasoning) and "
            "'prediction' (final probability value between 0 and 1) fields."
        )

        # Add task-specific context
        if task_type == "mortality":
            system_message["content"] += (
                f" For mortality prediction, the 'prediction' field should reflect the probability of the patient "
                f"not surviving their hospital stay (higher values indicate higher mortality risk)."
            )
        elif task_type == "readmission":
            system_message["content"] += (
                f" For readmission prediction, the 'prediction' field should reflect the probability of patient "
                f"readmission within 30 days post-discharge (higher values indicate higher readmission risk)."
            )

        # Format doctor reviews
        formatted_reviews = []
        for i, review_item in enumerate(doctor_reviews): # Changed variable name to avoid confusion with dict.get('review')
            review = review_item.get('review', {}) # Get the nested review dictionary
            formatted_review = f"Doctor {i+1} ({review_item.get('specialty', 'Unknown')}):\n" # Include specialty in review summary
            formatted_review += f"Agree: {'Yes' if review.get('agree', False) else 'No'}\n"
            formatted_review += f"Reason: {review.get('reason', '')}\n"
            formatted_review += f"Prediction: {review.get('prediction', '')}\n"
            formatted_reviews.append(formatted_review)

        reviews_text = "\n".join(formatted_reviews)

        # Prepare current synthesis text
        current_synthesis_text = (
            f"Current synthesized explanation: {current_synthesis.get('explanation', '')}\n"
            f"Current suggested prediction: {current_synthesis.get('prediction', '')}"
        )

        decision_type = "final" if all_agree or reached_max_rounds else "current round"

        # Review previous rounds' syntheses from memory
        previous_syntheses = []
        for i, mem in enumerate(self.memory):
            if mem["type"] == "synthesis" and mem["round"] < current_round:
                prev = f"Round {mem['round']} synthesis:\n"
                prev += f"Explanation: {mem['content'].get('explanation', '')}\n"
                prev += f"Prediction: {mem['content'].get('prediction', '')}"
                previous_syntheses.append(prev)

        previous_syntheses_text = "\n\n".join(previous_syntheses) if previous_syntheses else "No previous syntheses available."

        # Prepare user message
        user_message = {
            "role": "user",
            "content": f"EHR data and task: {question[:500]}...\n\n"
                      f"{current_synthesis_text}\n\n"
                      f"Doctor Reviews:\n{reviews_text}\n\n"
                      f"Previous Rounds:\n{previous_syntheses_text}\n\n"
                      f"Please provide your {decision_type} decision, "
                      f"in JSON format, including 'explanation' and 'prediction' fields."
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Meta agent final decision successfully parsed")

            # Ensure prediction is a float between 0 and 1
            if "prediction" in result:
                try:
                    pred = float(result["prediction"])
                    result["prediction"] = max(0.0, min(1.0, pred))
                except ValueError: # Changed bare except to ValueError for specificity
                    result["prediction"] = 0.5  # Default fallback
            else:
                result["prediction"] = 0.5

            result["system_message"] = system_message["content"]
            result["user_message"] = user_message["content"]

            # Add to memory
            self.memory.append({
                "type": "decision",
                "round": current_round,
                "final": all_agree or reached_max_rounds,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            print("Meta agent final decision is not valid JSON, using fallback parsing")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

            # Add to memory
            self.memory.append({
                "type": "decision",
                "round": current_round,
                "final": all_agree or reached_max_rounds,
                "content": result
            })
            return result


class EvaluateAgent(BaseAgent):
    """Evaluator Agent: Evaluate the quality of each DoctorAgent's preliminary report and its similarity to the final report, scoring 10 points."""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        super().__init__(agent_id, AgentType.EVALUATOR, model_key)
        print(f"Initializing evaluator agent, ID: {agent_id}, Model: {model_key}")

    def evaluate(self, doctor_report: Dict[str, Any], final_report: Dict[str, Any], question: str, task_type: str) -> Dict[str, Any]:
        """
        Evaluate the quality of each DoctorAgent's preliminary report and its similarity to the final report, returning a score between 0 and 10.
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a medical AI evaluation expert. Please score each doctor's preliminary report based on the following criteria:\n"
                "1. The medical analysis in the report is reasonable, complete, and logically clear (5 points);\n"
                "2. The similarity between the preliminary report and the final team report's conclusion and prediction value (5 points, the closer the better).\n"
                "Please combine the above two criteria to give a total score between 0 and 10, and output in JSON format: {\"score\": score, \"reason\": scoring reason}."
            )
        }
        user_message = {
            "role": "user",
            "content": (
                f"EHR data and task: {question[:500]}...\n\n"
                f"Doctor preliminary report:\nExplanation: {doctor_report.get('explanation', '')}\nPrediction: {doctor_report.get('prediction', '')}\n\n"
                f"Final team report:\nExplanation: {final_report.get('explanation', '')}\nPrediction: {final_report.get('prediction', '')}\n\n"
                f"Task type: {task_type}. Please strictly follow the requirements to score and output JSON."
            )
        }
        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            print("Evaluator agent successfully parsed")
            # Ensure score is a float between 0 and 10
            if "score" in result:
                try:
                    score = float(result["score"])
                    result["score"] = max(0.0, min(10.0, score))
                except ValueError: # Changed bare except to ValueError for specificity
                    result["score"] = 0.0
            else:
                result["score"] = 0.0

            result["system_message"] = system_message["content"]
            result["user_message"] = user_message["content"]

            # Add to memory
            self.memory.append({
                "type": "evaluation",
                "content": result
            })
            return result
        except json.JSONDecodeError:
            print("Evaluator agent is not valid JSON, using fallback parsing")
            lines = response_text.strip().split('\n')
            result = {}

            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace("\"", "")
                    value = value.strip()
                    result[key] = value

            if "score" not in result:
                result["score"] = 0.0
            if "reason" not in result:
                result["reason"] = ""

            result["response_text"] = response_text
            return result


class MDTConsultation:
    """Multi-disciplinary team consultation coordinator for EHR prediction."""

    def __init__(self,
                max_rounds: int = 3,
                doctor_configs: List[Dict] = None,
                meta_model_key: str = "deepseek-v3-official",
                evaluator_model_key: str = "deepseek-v3-official"
        ):
        """
        Initialize MDT consultation.

        Args:
            max_rounds: Maximum number of discussion rounds
            doctor_configs: List of dictionaries specifying each doctor's specialty and model_key
            meta_model_key: LLM model for meta agent
            evaluator_model_key: LLM model for evaluator agent
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"},
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"},
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"},
        ] # Added default specialties for clarity

        self.meta_model_key = meta_model_key
        self.evaluator_model_key = evaluator_model_key

        # Initialize doctor agents with different specialties and models
        self.doctor_agents: List[DoctorAgent] = []
        self.doctor_specialties_for_logging = []
        for idx, config in enumerate(self.doctor_configs, 1):
            agent_id = f"doctor_{idx}"
            model_key = config.get("model_key", "deepseek-v3-official")
            specialty = config.get("specialty", "General Medicine")
            doctor_agent = DoctorAgent(agent_id, specialty, model_key)
            self.doctor_agents.append(doctor_agent)
            self.doctor_specialties_for_logging.append(specialty)

        # Initialize meta agent
        self.meta_agent = MetaAgent("meta", meta_model_key)
        self.evaluator_agent = EvaluateAgent("evaluator", evaluator_model_key)

        # Prepare doctor info for logging
        doctor_info = ", ".join([
            f"{config.get('specialty', 'General Medicine')} ({config.get('model_key', 'default')})"
            for config in self.doctor_configs
        ])
        print(f"Initialized MDT consultation, max_rounds={max_rounds}, doctors: [{doctor_info}], meta_model={meta_model_key}")


    def run_consultation(self,
                        qid: str,
                        question: str,
                        task_type: str = "mortality") -> Dict[str, Any]:
        """
        Run the MDT consultation process.

        Args:
            qid: Question ID
            question: Question containing EHR data
            task_type: Type of task (mortality or readmission)

        Returns:
            Dictionary containing final consultation result
        """
        start_time = time.time()

        print(f"Starting MDT consultation for case {qid}")
        print(f"Task type: {task_type}")

        # Case consultation history
        case_history = {
            "rounds": []
        }

        current_round = 0
        final_decision = None
        consensus_reached = False

        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            print(f"Starting round {current_round}")

            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": []}

            # Step 1: Each doctor analyzes the case
            doctor_opinions = []
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty}) analyzing case")
                opinion = doctor.analyze_case(question, task_type)
                # Store original opinion with doctor details for meta agent to access
                doctor_opinions.append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty,
                    "opinion": opinion
                })
                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty,
                    "opinion": opinion
                })

                print(f"Doctor {i+1} prediction: {opinion.get('prediction', '')}")

            # Step 2: Meta agent synthesizes opinions
            print("Meta agent synthesizing opinions")
            synthesis = self.meta_agent.synthesize_opinions(
                question, doctor_opinions,
                current_round, task_type
            )
            round_data["synthesis"] = synthesis

            print(f"Meta agent synthesis prediction: {synthesis.get('prediction', '')}")

            # Step 3: Doctors review synthesis
            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                print(f"Doctor {i+1} ({doctor.specialty}) reviewing synthesis")
                review = doctor.review_synthesis(question, synthesis, task_type)
                # Store original review with doctor details for meta agent to access
                doctor_reviews.append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty,
                    "review": review
                })
                round_data["reviews"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty,
                    "review": review
                })

                agrees = review.get('agree', False)
                all_agree = all_agree and agrees

                print(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            # Add round data to history
            case_history["rounds"].append(round_data)

            # Step 4: Meta agent makes decision based on reviews
            decision = self.meta_agent.make_final_decision(
                question, doctor_reviews,
                synthesis, current_round, self.max_rounds, task_type
            )

            # Check if consensus reached
            if all_agree:
                consensus_reached = True
                final_decision = decision
                print("Consensus reached")
            else:
                print("No consensus reached, continuing to next round")
                if current_round == self.max_rounds:
                    # If max rounds reached, use the last round's decision as final
                    final_decision = decision

        # If no final decision, fallback to the last round's decision
        if not final_decision:
            # Fallback if loop didn't set final_decision for some reason
            # (e.g., if max_rounds was 0, though current_round starts at 0 and increments)
            final_decision = decision if 'decision' in locals() else {"explanation": "No decision could be made.", "prediction": 0.5}

        print(f"Final prediction: {final_decision.get('prediction', '')}")

        # Evaluate each DoctorAgent's preliminary report
        doctor_scores = []
        for i, doctor in enumerate(self.doctor_agents):
            # Find the first round analysis
            first_analysis = None
            for mem in doctor.memory:
                if mem["type"] == "analysis" and mem["round"] == 1:
                    first_analysis = mem["content"]
                    break
            if first_analysis is not None:
                score_result = self.evaluator_agent.evaluate(first_analysis, final_decision, question, task_type)
            else:
                score_result = {"score": 0.0, "reason": "No preliminary report found"}
            doctor_scores.append({
                "doctor_id": doctor.agent_id,
                "specialty": doctor.specialty,
                "score": score_result.get("score", 0.0),
                "reason": score_result.get("reason", "")
            })

        # Calculate processing time
        processing_time = time.time() - start_time

        # Add final decision and scores to history
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["processing_time"] = processing_time
        case_history["doctor_scores"] = doctor_scores

        return case_history


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
                    except ValueError: # Changed bare except to ValueError for specificity
                        result[key] = 0.5  # Default fallback
                else:
                    result[key] = value

        # Ensure explanation and prediction fields exist
        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "prediction" not in result:
            result["prediction"] = 0.5  # Default to 0.5 probability

        return result


def process_input(item, task_type, doctor_configs=None, meta_model_key="deepseek-v3-official", evaluator_model_key="deepseek-v3-official"):
    """
    Process input data.

    Args:
        item: Input data dictionary with question
        task_type: Type of task (mortality or readmission)
        doctor_configs: List of doctor configurations (specialty and model_key)
        meta_model_key: Model key for the meta agent

    Returns:
        Processed result from MDT consultation
    """
    # Required fields
    qid = item.get("qid")
    question = item.get("question")

    start_time = time.time()

    # Initialize consultation
    mdt = MDTConsultation(
        max_rounds=2,
        doctor_configs=doctor_configs,
        meta_model_key=meta_model_key,
        evaluator_model_key=evaluator_model_key,
    )

    # Run consultation
    result = mdt.run_consultation(
        qid=qid,
        question=question,
        task_type=task_type,
    )

    # Calculate processing time
    processing_time = time.time() - start_time
    result["processing_time"] = processing_time

    return result


def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on EHR datasets")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "tjh", "esrd"],
                       help="Specify dataset name: mimic-iv or tjh or esrd")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission"],
                       help="Prediction task: mortality or readmission")
    parser.add_argument("--meta_model", type=str, default="deepseek-v3-official",
                       help="Model used for meta agent")
    parser.add_argument("--doctor_models", nargs='+', default=["deepseek-v3-official", "deepseek-v3-official", "deepseek-v3-official"],
                       help="Models used for doctor agents. Provide one model name per doctor.")
    parser.add_argument("--evaluate_model", type=str, default="deepseek-v3-official",
                       help="Model used for evaluator agent")
    args = parser.parse_args()

    method = "ColaCare" # ColaCare by default

    # Dataset and task
    dataset_name = args.dataset
    task_type = args.task
    print(f"Dataset: {dataset_name}, Task: {task_type}")

    # Validate the dataset and task combination
    if dataset_name in ["tjh", "esrd"] and task_type == "readmission":
        print(f"Error: The {dataset_name} dataset doesn't contain readmission task data.")
        return

    # Define the mapping for specialties based on dataset
    SPECIALTIES_MAP = {
        "esrd": "End-Stage Renal Disease",
        "cdsl": "COVID-19",
        "mimic-iv": "Intensive Care",
        "tjh": "COVID-19",
    }

    dataset_specialty = SPECIALTIES_MAP.get(dataset_name, "General Medicine")
    print(f"Doctors' specialty for this dataset ({dataset_name}): {dataset_specialty}")

    # Create logs directory structure
    logs_dir = os.path.join("logs", dataset_name, task_type, method)
    os.makedirs(logs_dir, exist_ok=True)

    # Set up data path
    data_path = f"./my_datasets/ehr/{dataset_name}/processed/ehr_{task_type}_test.json"

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Create doctor configurations, assigning the determined specialty
    doctor_configs = []

    if len(args.doctor_models) > 3:
        print(f"Warning: More doctor models ({len(args.doctor_models)}) provided than typical (3)."
              f"All provided models will be used, each assigned the dataset specialty.")

    for model in args.doctor_models:
        doctor_configs.append({
            "model_key": model,
            "specialty": dataset_specialty
        })

    print(f"Configuring {len(doctor_configs)} doctors with models: {[cfg['model_key'] for cfg in doctor_configs]} and specialty: {dataset_specialty}")

    # Process each item
    for item in tqdm(data, desc=f"Running MDT consultation on {dataset_name} {task_type}"):
        qid = item["qid"]

        # Format the qid for the output file
        qid_str = str(qid)

        # Skip if already processed
        if os.path.exists(os.path.join(logs_dir, f"ehr_{qid_str}-result.json")):
            print(f"Skipping {qid_str} - already processed")
            continue

        try:
            # Process the item
            result = process_input(
                item,
                task_type=task_type,
                doctor_configs=doctor_configs,
                meta_model_key=args.meta_model,
                evaluator_model_key=args.evaluate_model
            )

            # Add output to the original item and save
            item_result = {
                "qid": qid,
                "question": item["question"],
                "ground_truth": item.get("answer"),
                "predicted_value": result["final_decision"]["prediction"],
                "case_history": result,
                "timestamp": int(time.time()),
                "processing_time": result["processing_time"]
            }

            # Save individual result
            save_json(item_result, os.path.join(logs_dir, f"ehr_{qid_str}-result.json"))

        except Exception as e:
            print(f"Error processing item {qid}: {e}")
            # Optionally, save an error log for the QID
            error_log_path = os.path.join(logs_dir, f"ehr_{qid_str}-error.log")
            with open(error_log_path, "w") as f:
                f.write(f"Error processing {qid}: {e}\n")
                import traceback
                traceback.print_exc(file=f)


if __name__ == "__main__":
    main()