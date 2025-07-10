"""
ehr_multi_agent_framework.py

Multi-agent LLM framework for EHR predictive modeling tasks.
This code adapts a medical QA multi-agent framework to process EHR time series data
and make predictions for mortality and readmission probability.
"""

import os
import json
import time
import asyncio
import asyncio
import logging
import argparse
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any, List

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
        Call the LLM with messages and handle retries.

        Args:
            system_message: System message setting context
            user_message: User message containing EHR data
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response text
        """
        retries = 0
        # TODO: improve input context length limit
        while retries < max_retries:
            try:
                if self.logger:
                    self.logger.info(f"Agent {self.agent_id} calling LLM, system message: {system_message['content'][:100]}...")
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
        logger=None):
        """
        Initialize a doctor agent.

        Args:
            agent_id: Unique identifier for the doctor
            specialty: Doctor's clinical specialty (string)
            model_key: LLM model to use
            logger: Logger object for logging
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key, logger=logger)
        self.specialty = specialty
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
                question_short=question[:2000], # Pass a sufficiently long part of the question
                task_type=task_type
            )
        }

        # Call LLM to generate the query
        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            query_text = result.get("query", [])
        except json.JSONDecodeError:
            # query_text = response_text.strip().strip('"')
            query_text = []

        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} generated RAG query: {query_text}")
        return query_text

    async def analyze_case(self,
        question: str,
        task_type: str) -> Dict[str, Any]:
        """
        Analyze an EHR case and predict outcome probability, including RAG retrieval.

        Args:
            question: Question containing structured EHR time series data
            task_type: Type of task (mortality or readmission)

        Returns:
            Dictionary containing analysis results and prediction
        """
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) analyzing case with model: {self.model_key}")

        if task_type == "mortality":
            task_hint = prompt_template.TASK_HINT_MORTALITY
        elif task_type == "readmission":
            task_hint = prompt_template.TASK_HINT_READMISSION
        else:
            task_hint = ""

        # Step 1: Generate RAG query
        rag_query = self._generate_rag_query(question, task_type)

        # Step 2: Call the simulated RAG tool with the generated query
        retrieved_literature = "Retrieved literature:\n"
        for i, query in enumerate(rag_query):
            retrieved_literature += await litsense_api_call(query=query, order=i, max_results=10) + "\n"

        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} retrieved literature (first 200 chars): {retrieved_literature[:200]}...")

        system_message = {
            "role": "system",
            "content": f"{prompt_template.DOCTOR_ANALYZE_SYSTEM.format(specialty=self.specialty, task_hint=task_hint)}"
        }
        user_message = {
            "role": "user",
            "content": f"{prompt_template.DOCTOR_ANALYZE_USER.format(question=question, retrieved_literature=retrieved_literature)}"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info(f"Doctor {self.agent_id} response successfully parsed.")

            # Ensure prediction is a float between 0 and 1
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
            result["rag_query"] = rag_query
            result["retrieved_literature"] = retrieved_literature

            # Add to memory
            self.memory.append({
                "type": "analysis",
                "content": result,
                "round": 0
            })
            return result
        except json.JSONDecodeError:
            # If JSON format is not correct, use fallback parsing
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} response is not valid JSON, using fallback parsing.")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text
            result["retrieved_literature"] = retrieved_literature

            # Add to memory
            self.memory.append({
                "type": "analysis",
                "content": result,
                "round": 0
            })
            return result

    def review_synthesis(self,
        question: str,
        synthesis: Dict[str, Any],
        task_type: str,
        current_round: int = 1) -> Dict[str, Any]:
        """
        Review the meta agent's synthesis, including RAG retrieval.

        Args:
            question: Original question with EHR data
            synthesis: Meta agent's synthesis
            task_type: Type of task (mortality or readmission)
            current_round: Current round

        Returns:
            Dictionary containing agreement status and possible rebuttal
        """
        if self.logger:
            self.logger.info(f"Doctor {self.agent_id} ({self.specialty}) reviewing synthesis with {self.model_key} and RAG.")

        # Get doctor's own most recent analysis
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break

        if task_type == "mortality":
            task_hint = prompt_template.TASK_HINT_REVIEW_MORTALITY
        elif task_type == "readmission":
            task_hint = prompt_template.TASK_HINT_REVIEW_READMISSION
        else:
            task_hint = ""

        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"Your previous analysis:\nExplanation: {own_analysis.get('explanation', '')}\nPrediction: {own_analysis.get('prediction', '')}\n\n"

        synthesis_text = f"Synthesized report:\n{synthesis.get('report', '')}\n"
        synthesis_text += f"Synthesized explanation: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"Suggested prediction: {synthesis.get('prediction', '')}"

        system_message = {
            "role": "system",
            "content": f"{prompt_template.DOCTOR_REVIEW_SYSTEM.format(specialty=self.specialty, current_round=current_round, task_hint=task_hint)}"
        }
        user_message = {
            "role": "user",
            "content": f"{prompt_template.DOCTOR_REVIEW_USER.format(question_short=question, own_analysis_text=own_analysis_text, synthesis_text=synthesis_text)}"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info(f"Doctor {self.agent_id} review successfully parsed")

            # Normalize agree field
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes"]

            # Ensure prediction is a float between 0 and 1
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

            # Add to memory
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            if self.logger:
                self.logger.warning(f"Doctor {self.agent_id} review is not valid JSON, using fallback parsing")
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
                    except Exception:
                        result["prediction"] = 0.501

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
                    result["prediction"] = synthesis.get("prediction", 0.501)

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
        Synthesize multiple doctors' opinions.

        Args:
            question: Original question with EHR data
            doctor_opinions: List of doctor opinions
            doctor_reviews: List of doctor reviews
            task_type: Type of task (mortality or readmission)
            current_round: Current round

        Returns:
            Dictionary containing synthesized explanation and prediction
        """
        if self.logger:
            self.logger.info(f"Meta agent synthesizing opinions with model: {self.model_key} in round {current_round}")

        if task_type == "mortality":
            task_hint = prompt_template.TASK_HINT_MORTALITY
        elif task_type == "readmission":
            task_hint = prompt_template.TASK_HINT_READMISSION
        else:
            task_hint = ""

        if current_round == 0:
            opinions_text = "\n".join([
                f"Doctor {i+1}:\nExplanation: {opinion.get('opinion', {}).get('explanation', '')}\nPrediction: {opinion.get('opinion', {}).get('prediction', '')}" for i, opinion in enumerate(doctor_opinions)
            ])
            system_message = {
                "role": "system",
                "content": f"{prompt_template.META_SYNTHESIZE_SYSTEM.format(task_hint=task_hint)}"
            }
            user_message = {
                "role": "user",
                "content": f"{prompt_template.META_SYNTHESIZE_USER.format(question_short=question, opinions_text=opinions_text)}"
            }
        else:
            # 多轮共识
            prev_synthesis = None
            for mem in reversed(self.memory):
                if mem["type"] == "synthesis" and mem["round"] < current_round:
                    prev_synthesis = mem["content"]
                    break
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
                "content": f"{prompt_template.META_RESYNTHESIZE_SYSTEM.format(current_round=current_round, task_hint=task_hint)}"
            }
            user_message = {
                "role": "user",
                "content": f"{prompt_template.META_RESYNTHESIZE_USER.format(question_short=question, prev_synthesis=prev_synthesis_str, doctor_reviews=doctor_reviews_str)}"
            }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info("Meta agent synthesis successfully parsed")

            # Ensure prediction is a float between 0 and 1
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

            # Add to memory
            self.memory.append({
                "type": "synthesis",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            if self.logger:
                self.logger.warning("Meta agent synthesis is not valid JSON, using fallback parsing")
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
        if self.logger:
            self.logger.info(f"Meta agent making round {current_round} decision with model: {self.model_key}")

        # Check if all doctors agree
        all_agree = all(review.get('agree', False) for review in doctor_reviews)
        reached_max_rounds = current_round >= max_rounds

        # Prepare system message for final decision
        system_message = {
            "role": "system",
            "content": "You are a medical consensus coordinator making a final decision. "
        }

        if all_agree:
            decision_status = "All doctors agree with your synthesis, generate a final report."
        elif reached_max_rounds:
            decision_status = f"Maximum number of discussion rounds ({max_rounds}) reached without full consensus. Make a final decision using majority opinion approach."
        else:
            decision_status = "Not all doctors agree with your synthesis, but a decision for the current round is needed."

        if task_type == "mortality":
            task_hint = prompt_template.TASK_HINT_MORTALITY
        elif task_type == "readmission":
            task_hint = prompt_template.TASK_HINT_READMISSION
        else:
            task_hint = ""

        reviews_text = "\n".join([
            f"Doctor {i+1} ({review_item.get('specialty', 'Unknown')}):\nAgree: {'Yes' if review_item.get('review', {}).get('agree', False) else 'No'}\nReason: {review_item.get('review', {}).get('reason', '')}\nPrediction: {review_item.get('review', {}).get('prediction', '')}" for i, review_item in enumerate(doctor_reviews)
        ])
        current_synthesis_text = (
            f"Current synthesized explanation: {current_synthesis.get('explanation', '')}\n"
            f"Current suggested prediction: {current_synthesis.get('prediction', '')}"
        )
        decision_type = "final" if all_agree or reached_max_rounds else "current round"
        previous_syntheses_text = previous_syntheses_text if 'previous_syntheses_text' in locals() else "No previous syntheses available."

        system_message = {
            "role": "system",
            "content": f"{prompt_template.META_DECISION_SYSTEM.format(decision_status=decision_status, task_hint=task_hint)}"
        }
        user_message = {
            "role": "user",
            "content": f"{prompt_template.META_DECISION_USER.format(question_short=question, current_synthesis_text=current_synthesis_text, reviews_text=reviews_text, previous_syntheses_text=previous_syntheses_text, decision_type=decision_type)}"
        }

        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info("Meta agent final decision successfully parsed")

            # Ensure prediction is a float between 0 and 1
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
            if self.logger:
                self.logger.warning("Meta agent final decision is not valid JSON, using fallback parsing")
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
    """Evaluator Agent:
    1. Evaluate the quality of each DoctorAgent's preliminary report and its similarity to the final report, scoring 10 points.
    2. Evaluate the quality of the final patient report based on Factuality, Safety, and Explainability, using an LLM-as-a-judger approach.
    Scores are on a scale of 1 to 5."""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        super().__init__(agent_id, AgentType.EVALUATOR, model_key, logger=logger)
        if self.logger:
            self.logger.info(f"Initializing evaluator agent, ID: {agent_id}, Model: {model_key}")

    def evaluate_preliminary_report(self, doctor_report: Dict[str, Any], final_report: Dict[str, Any], question: str, task_type: str) -> Dict[str, Any]:
        """
        Evaluate the quality of each DoctorAgent's preliminary report and its similarity to the final report, returning a score between 0 and 10.
        """
        system_message = {
            "role": "system",
            "content": f"{prompt_template.EVALUATE_SYSTEM}"
        }
        user_message = {
            "role": "user",
            "content": f"{prompt_template.EVALUATE_USER.format(question_short=question, doctor_explanation=doctor_report.get('explanation', ''), doctor_prediction=doctor_report.get('prediction', ''), final_explanation=final_report.get('explanation', ''), final_prediction=final_report.get('prediction', ''), task_type=task_type)}"
        }
        # Call LLM with retry mechanism
        response_text = self.call_llm(system_message, user_message)

        # Parse response
        try:
            result = json.loads(preprocess_response_string(response_text))
            if self.logger:
                self.logger.info("Evaluator agent successfully parsed")
            # Ensure score is a float between 0 and 10
            if "score" in result:
                try:
                    score = float(result["score"])
                    result["score"] = max(0.0, min(10.0, score))
                except Exception:
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
            if self.logger:
                self.logger.warning("Evaluator agent is not valid JSON, using fallback parsing")
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
            task_type: Type of task (mortality or readmission).
            label: True label of the patient under the task.

        Returns:
            Dictionary containing evaluation scores and reasons for Factuality, Safety,
            and Explainability, plus an overall comment.
        """
        if self.logger:
            self.logger.info(f"Report evaluation agent evaluating report for task '{task_type}'.")

        system_message = {
            "role": "system",
            "content": prompt_template.REPORT_EVALUATOR_SYSTEM
        }
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
            # First attempt to parse as clean JSON
            result = json.loads(preprocess_response_string(response_text))

            # Validate and normalize scores (ensure between 1 and 5)
            for dim in ["accuracy", "explainability", "safety"]:
                if dim in result and "score" in result[dim]:
                    try:
                        score = int(result[dim]["score"])
                        result[dim]["score"] = max(1, min(5, score))
                    except (ValueError, TypeError):
                        result[dim]["score"] = 1 # Default to lowest if parsing fails
                else:
                    result[dim] = {"score": 1, "reason": "Missing or invalid score"}

            if self.logger:
                self.logger.info("Report evaluation agent response successfully parsed.")
        except json.JSONDecodeError:
            if self.logger:
                self.logger.warning("Report evaluation agent response is not valid JSON, attempting fallback parsing.")
            # Fallback parsing for less structured responses
            # This is a simpler fallback and might need refinement based on actual LLM output patterns
            result = parse_structured_output_for_final_report(response_text)
            if self.logger:
                self.logger.info("Report evaluation agent fallback parsing completed.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing evaluation response: {e}")
            result = {
                "accuracy": {"score": 1, "reason": f"Parsing error: {e}"},
                "explainability": {"score": 1, "reason": f"Parsing error: {e}"},
                "safety": {"score": 1, "reason": f"Parsing error: {e}"}
            }
        result["system_message"] = system_message["content"]
        result["user_message"] = user_message["content"]
        result["response_text"] = response_text
        return result


class MDTConsultation:
    """Multi-disciplinary team consultation coordinator for EHR prediction."""

    def __init__(self,
        max_rounds: int = 3,
        doctor_configs: List[Dict] = None,
        meta_model_key: str = "deepseek-v3-official",
        evaluator_model_key: str = "deepseek-v3-official",
        logger=None
    ):
        """
        Initialize MDT consultation.

        Args:
            max_rounds: Maximum number of discussion rounds
            doctor_configs: List of dictionaries specifying each doctor's specialty and model_key
            meta_model_key: LLM model for meta agent
            evaluator_model_key: LLM model for evaluator agent
            logger: Logger object for logging consultation process
        """
        self.max_rounds = max_rounds
        self.doctor_configs = doctor_configs or [
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"},
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"},
            {"specialty": "General Medicine", "model_key": "deepseek-v3-official"},
        ]

        self.meta_model_key = meta_model_key
        self.evaluator_model_key = evaluator_model_key
        self.logger = logger

        # Initialize doctor agents with different specialties and models
        self.doctor_agents: List[DoctorAgent] = []
        self.doctor_specialties_for_logging = []
        for idx, config in enumerate(self.doctor_configs, 1):
            agent_id = f"doctor_{idx}"
            model_key = config.get("model_key", "deepseek-v3-official")
            specialty = config.get("specialty", "General Medicine")
            doctor_agent = DoctorAgent(agent_id, specialty, model_key, logger=logger)
            self.doctor_agents.append(doctor_agent)
            self.doctor_specialties_for_logging.append(specialty)

        # Initialize meta agent
        self.meta_agent = MetaAgent("meta", meta_model_key, logger=logger)
        self.evaluator_agent = EvaluateAgent("evaluator", evaluator_model_key, logger=logger)

        # Prepare doctor info for logging
        doctor_info = ", ".join([
            f"{config.get('specialty', 'General Medicine')} ({config.get('model_key', 'default')})"
            for config in self.doctor_configs
        ])
        if self.logger:
            self.logger.info(f"Initialized MDT consultation, max_rounds={max_rounds}, doctors: [{doctor_info}], meta_model={meta_model_key}")

    async def run_consultation(self,
        qid: str,
        question: List[str],
        task_type: str = "mortality",
        label: str = None) -> Dict[str, Any]:
        """
        Run the MDT consultation process.

        Args:
            qid: Question ID
            question: Question containing EHR data
            task_type: Type of task (mortality or readmission)
            label: True label of the patient under the task

        Returns:
            Dictionary containing final consultation result
        """
        if self.logger:
            self.logger.info(f"Starting MDT consultation for case {qid}")
            self.logger.info(f"Task type: {task_type}")
            self.logger.info(f"True label: {label}")

        # Case consultation history
        case_history = {
            "opinions": [],
            "synthesis": None,
            "rounds": []
        }

        current_round = 0
        final_decision = None
        consensus_reached = False

        # Step 1: Each doctor analyzes the case
        doctor_opinions = []
        for i, doctor in enumerate(self.doctor_agents):
            if self.logger:
                self.logger.info(f"Doctor {i+1} ({doctor.specialty}) analyzing case")
            opinion = await doctor.analyze_case(question[i], task_type)
            # Store original opinion with doctor details for meta agent to access
            doctor_opinions.append({
                "doctor_id": doctor.agent_id,
                "specialty": doctor.specialty,
                "opinion": opinion
            })
            case_history["opinions"].append({
                "doctor_id": doctor.agent_id,
                "specialty": doctor.specialty,
                "opinion": opinion
            })
            if self.logger:
                self.logger.info(f"Doctor {i+1} prediction: {opinion.get('prediction', '')}")

        # Step 2: Meta agent synthesizes opinions
        if self.logger:
            self.logger.info("Meta agent synthesizing opinions")
        synthesis = self.meta_agent.synthesize_opinions(
            question[-1], doctor_opinions=doctor_opinions, task_type=task_type, current_round=current_round
        )
        case_history["synthesis"] = synthesis
        if self.logger:
            self.logger.info(f"Meta agent synthesis prediction: {synthesis.get('prediction', '')}")

        # Step 3: Doctors review synthesis
        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            if self.logger:
                self.logger.info(f"Starting round {current_round}")
            round_data = {"round": current_round, "reviews": [], "synthesis": None}

            # Step 3.1: Doctor agent reviews synthesis
            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                if self.logger:
                    self.logger.info(f"Doctor {i+1} ({doctor.specialty}) reviewing synthesis")
                review = doctor.review_synthesis(question[i], synthesis, task_type, current_round)
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
                if self.logger:
                    self.logger.info(f"Doctor {i+1} agrees: {'Yes' if agrees else 'No'}")

            # Step 3.2: Meta agent synthesizes opinions
            if self.logger:
                self.logger.info("Meta agent synthesizing opinions")
            synthesis = self.meta_agent.synthesize_opinions(
                question[-1], doctor_reviews=doctor_reviews, task_type=task_type, current_round=current_round
            )
            round_data["synthesis"] = synthesis

            # Add round data to history
            case_history["rounds"].append(round_data)

            # Step 4: Meta agent makes decision based on reviews
            # decision = self.meta_agent.make_final_decision(
            #     question, doctor_reviews,
            #     synthesis, current_round, self.max_rounds, task_type
            # )

            # Check if consensus reached
            if all_agree:
                if self.logger:
                    self.logger.info("Consensus reached")
                consensus_reached = True
                final_decision = synthesis
            elif current_round == self.max_rounds:
                # If max rounds reached, use the last round's decision as final
                if self.logger:
                    self.logger.info("Max rounds reached, using the last round's decision as final")
                final_decision = synthesis
            else:
                self.logger.info("No consensus reached, continuing to next round")

        if self.logger:
            self.logger.info(f"Final prediction: {final_decision.get('prediction', '')}")

        # Step 4: Evaluate each DoctorAgent's preliminary report
        if self.logger:
            self.logger.info("Starting preliminary report evaluation.")
        doctor_scores = []
        for i, doctor in enumerate(self.doctor_agents):
            # Find the first round analysis
            first_analysis = None
            for mem in doctor.memory:
                if mem["type"] == "analysis" and mem["round"] == 0:
                    first_analysis = mem["content"]
                    break
            if first_analysis is not None:
                score_result = self.evaluator_agent.evaluate_preliminary_report(first_analysis, final_decision, question[i], task_type)
            else:
                score_result = {"score": 0.0, "reason": "No preliminary report found"}
            doctor_scores.append({
                "doctor_id": doctor.agent_id,
                "specialty": doctor.specialty,
                "score": score_result.get("score", 0.0),
                "reason": score_result.get("reason", "")
            })

        # Step 5: Evaluate the final patient report
        if self.logger:
            self.logger.info("Starting final report trustworthiness evaluation.")
        final_report_evaluation = self.evaluator_agent.evaluate_final_report(
            original_question=question[-1],
            final_report_str=final_decision.get('report', ''),
            final_report_explanation=final_decision.get('explanation', ''),
            final_report_prediction=final_decision.get('prediction', 0.501),
            task_type=task_type,
            label=label
        )

        # Add final decision and scores to history
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        case_history["doctor_scores"] = doctor_scores
        case_history["report_trustworthiness_evaluation"] = final_report_evaluation

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


async def process_input(item, task_type, doctor_configs=None, meta_model_key="deepseek-v3-official", evaluator_model_key="deepseek-v3-official", logger=None):
    """
    Process input data.

    Args:
        item: Input data dictionary with question
        task_type: Type of task (mortality or readmission)
        doctor_configs: List of doctor configurations (specialty and model_key)
        meta_model_key: Model key for the meta agent
        logger: Logger object for logging consultation process

    Returns:
        Processed result from MDT consultation
    """
    # Required fields
    qid = item.get("qid")
    question = item.get("question")
    label = item.get("ground_truth")

    start_time = time.time()

    # Initialize consultation
    mdt = MDTConsultation(
        max_rounds=2,
        doctor_configs=doctor_configs,
        meta_model_key=meta_model_key,
        evaluator_model_key=evaluator_model_key,
        logger=logger,
    )

    # Run consultation
    result = await mdt.run_consultation(
        qid=qid,
        question=question,
        task_type=task_type,
        label=label
    )

    # Calculate processing time
    processing_time = time.time() - start_time
    result["processing_time"] = processing_time

    return result


async def main():
    parser = argparse.ArgumentParser(description="Run MDT consultation on EHR datasets")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "tjh", "esrd"],
                       help="Specify dataset name: mimic-iv or tjh or esrd")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission"],
                       help="Prediction task: mortality or readmission")
    parser.add_argument("--modality", "-mo", type=str, default="ehr", choices=["ehr", "note", "mm"],
                       help="Modality of the dataset: ehr or note or mm")
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
    modality = args.modality
    print(f"Dataset: {dataset_name}, Task: {task_type}, Modality: {modality}")

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
    save_dir = os.path.join("logs", dataset_name, task_type, method, modality)
    logs_dir = os.path.join(save_dir, "logs")
    results_dir = os.path.join(save_dir, "results")
    error_dir = os.path.join(save_dir, "error")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Set up data path
    data_path = f"./my_datasets/ehr/{dataset_name}/processed/{modality}_{task_type}_test.json"

    # Load the data
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    # Create doctor configurations, assigning the determined specialty
    doctor_configs = []

    if len(args.doctor_models) > 3:
        print(f"Warning: More doctor models ({len(args.doctor_models)}) provided than typical (3)."
            f"All provided models will be used, each assigned the dataset specialty.")
        pass

    for model in args.doctor_models:
        doctor_configs.append({
            "model_key": model,
            "specialty": dataset_specialty
        })

    print(f"Configuring {len(doctor_configs)} doctors with models: {[cfg['model_key'] for cfg in doctor_configs]} and specialty: {dataset_specialty}")

    to_run_pids = [215, 265, 318, 740, 455, 370, 598, 616, 812, 998]
    # Process each item
    for item in tqdm(data, desc=f"Running MDT consultation on {dataset_name} {task_type}"):
        qid = item["qid"]
        if qid not in to_run_pids:
            continue

        question = item["question"]
        label = item.get("ground_truth")

        # Format the qid for the output file
        qid_str = str(qid)

        # Truncate the question for note modality
        question = question[:500] if modality == "note" else question

        # Skip if already processed
        if os.path.exists(os.path.join(results_dir, f"ehr_{qid_str}-result.json")):
            print(f"Skipping {qid_str} - already processed")
            continue

        # Configure logger
        log_path = os.path.join(logs_dir, f"ehr_{qid_str}.log")
        logger = get_logger(log_path)

        try:
            start_time = time.time()

            # Initialize consultation
            mdt = MDTConsultation(
                max_rounds=2,
                doctor_configs=doctor_configs,
                meta_model_key=args.meta_model,
                evaluator_model_key=args.evaluate_model,
                logger=logger,
            )

            # Run consultation
            result = await mdt.run_consultation(
                qid=qid,
                question=question,
                task_type=task_type,
                label=label
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Add output to the original item and save
            item_result = {
                "qid": qid,
                "question": question,
                "ground_truth": label,
                "predicted_value": result["final_decision"]["prediction"],
                "case_history": result,
                "processing_time": processing_time,
                "timestamp": int(time.time())
            }

            # Save individual result
            save_json(item_result, os.path.join(results_dir, f"ehr_{qid_str}-result.json"))

        except Exception as e:
            if logger:
                logger.error(f"Error processing item {qid}: {e}")
            # Optionally, save an error log for the QID
            error_log_path = os.path.join(error_dir, f"ehr_{qid_str}-error.log")
            with open(error_log_path, "w") as f:
                f.write(f"Error processing {qid}: {e}\n")
                import traceback
                traceback.print_exc(file=f)


if __name__ == "__main__":
    asyncio.run(main())