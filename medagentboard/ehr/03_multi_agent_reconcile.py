# multi_agent_reconcile.py

import os
import json
import time
import numpy as np
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Dict, List, Any

from openai import OpenAI

from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils import prompt_template
from medagentboard.utils.json_utils import get_logger, preprocess_response_string, save_json, load_json, parse_structured_output_for_final_report

class DiscussionPhase(Enum):
    """Enumeration of discussion phases in the Reconcile framework."""
    INITIAL = "initial"
    DISCUSSION = "discussion"
    FINAL = "final"

class ReconcileAgent:
    """An agent participating in the Reconcile framework for EHR prediction."""
    def __init__(self, agent_id: str, model_key: str):
        """Initializes a Reconcile agent."""
        self.agent_id = agent_id
        self.model_key = model_key
        self.memory = []

        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"Model key '{model_key}' not configured in LLM_MODELS_SETTINGS")

        model_config = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        self.model_name = model_config["model_name"]
        print(f"Initialized agent {self.agent_id} with model {self.model_name}")

    def call_llm(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """Calls the LLM with provided messages and a retry mechanism."""
        attempt = 0
        while attempt < max_retries:
            try:
                print(f"Agent {self.agent_id} calling LLM (attempt {attempt+1}/{max_retries})")
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, response_format={"type": "json_object"}
                )
                response_text = completion.choices[0].message.content
                print(f"Agent {self.agent_id} received response: {response_text[:100]}...")
                return response_text
            except Exception as e:
                attempt += 1
                print(f"Agent {self.agent_id} LLM call attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries: time.sleep(1)

        print(f"Agent {self.agent_id} all LLM call attempts failed, returning default response")
        return json.dumps({"reasoning": "LLM call failed", "prediction": 0.5, "confidence": 0.0})

    def generate_initial_response(self, question: str) -> Dict[str, Any]:
        """Generates an initial prediction for the EHR time series data."""
        print(f"Agent {self.agent_id} generating initial response")
        system_message = {"role": "system", "content": "You are a medical expert analyzing EHR data. Provide a prediction, detailed reasoning, and a confidence score (0.0 to 1.0)."}
        user_message = {"role": "user", "content": f"{question}\n\nProvide your response in JSON format with fields: 'reasoning', 'prediction', 'confidence'."}
        response_text = self.call_llm([system_message, user_message])
        result = self._parse_response(response_text)
        self.memory.append({"phase": DiscussionPhase.INITIAL.value, "response": result})
        return result

    def generate_discussion_response(self, question: str, discussion_prompt: str) -> Dict[str, Any]:
        """Generates a response during the discussion phase."""
        print(f"Agent {self.agent_id} generating discussion response")
        system_message = {"role": "system", "content": "You are a medical expert in a multi-agent discussion. Review others' opinions, then provide your updated analysis, prediction, and confidence."}
        user_message = {"role": "user", "content": f"Original Task:\n{question}\n\nDiscussion:\n{discussion_prompt}\n\nProvide your updated analysis in JSON format: 'reasoning', 'prediction', 'confidence'."}
        response_text = self.call_llm([system_message, user_message])
        result = self._parse_response(response_text)
        current_round = sum(1 for mem in self.memory if mem["phase"] == DiscussionPhase.DISCUSSION.value) + 1
        self.memory.append({"phase": DiscussionPhase.DISCUSSION.value, "round": current_round, "response": result})
        return result

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parses the LLM response into a structured format."""
        try:
            result = json.loads(preprocess_response_string(response_text))
            result["reasoning"] = result.get("reasoning", "No reasoning provided")
            result["prediction"] = max(0.0, min(1.0, float(result.get("prediction", 0.5))))
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
            return result
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"Agent {self.agent_id} failed to parse JSON response: {response_text[:100]}...")
            return {"reasoning": response_text, "prediction": 0.5, "confidence": 0.0}

class ReconcileCoordinator:
    """The coordinator for the Reconcile framework in EHR prediction tasks."""
    def __init__(self, agent_configs: List[Dict[str, str]], max_rounds: int = 3):
        """Initializes the Reconcile coordinator."""
        self.agents = [ReconcileAgent(cfg["agent_id"], cfg["model_key"]) for cfg in agent_configs]
        self.max_rounds = max_rounds
        print(f"Initialized ReconcileCoordinator with {len(self.agents)} agents, max_rounds={max_rounds}")

    def _group_predictions(self, predictions: List[Dict[str, Any]]) -> str:
        """Groups and summarizes predictions from agents."""
        groups = {"low_risk": [], "medium_risk": [], "high_risk": []}
        for pred in predictions:
            p_val = pred.get("prediction", 0.5)
            if p_val < 0.33: groups["low_risk"].append(pred)
            elif p_val < 0.67: groups["medium_risk"].append(pred)
            else: groups["high_risk"].append(pred)

        grouped_str = ""
        for name, data in groups.items():
            if data:
                avg_pred = sum(p['prediction'] for p in data) / len(data)
                avg_conf = sum(p['confidence'] for p in data) / len(data)
                explanations = "\n".join([f"• Expert: {p['reasoning'][:300]}..." for p in data])
                grouped_str += f"Prediction Group: {name.replace('_', ' ').title()}\n"
                grouped_str += f"Experts in group: {len(data)}, Avg Prediction: {avg_pred:.3f}, Avg Confidence: {avg_conf:.2f}\n"
                grouped_str += f"Explanations:\n{explanations}\n\n"
        return grouped_str.strip()

    def _consensus_reached(self, predictions: List[float]) -> bool:
        """Checks if predictions have reached a reasonable consensus."""
        return np.std(predictions) < 0.1 if predictions else False

    def _weighted_average(self, predictions: List[Dict[str, Any]]) -> float:
        """Computes the final team prediction using a confidence-weighted average."""
        weights = [p.get("confidence", 0.0) ** 2 for p in predictions]
        preds = [p.get("prediction", 0.5) for p in predictions]
        if sum(weights) == 0:
            return np.mean(preds) if preds else 0.5
        return np.average(preds, weights=weights)

    def run_discussion(self, question: List[str]) -> Dict[str, Any]:
        """Runs the complete discussion process for an EHR prediction task."""
        print(f"Starting EHR prediction discussion with {len(self.agents)} agents")
        discussion_history = []

        # Phase 1: Initial predictions
        current_predictions = [agent.generate_initial_response(q) for agent, q in zip(self.agents, question)]
        discussion_history.extend([{"phase": DiscussionPhase.INITIAL.value, "agent_id": a.agent_id, "response": p} for a, p in zip(self.agents, current_predictions)])

        # Phase 2: Multi-round discussion
        for round_num in range(1, self.max_rounds + 1):
            print(f"Phase 2: Discussion round {round_num}/{self.max_rounds}")
            discussion_prompt = self._group_predictions(current_predictions)

            new_predictions = [agent.generate_discussion_response(q, discussion_prompt) for agent, q in zip(self.agents, question)]
            discussion_history.extend([{"phase": DiscussionPhase.DISCUSSION.value, "round": round_num, "agent_id": a.agent_id, "response": p} for a, p in zip(self.agents, new_predictions)])

            current_predictions = new_predictions
            if self._consensus_reached([p.get("prediction", 0.5) for p in current_predictions]):
                print("Consensus reached, ending discussion.")
                break

        # Phase 3: Final team prediction
        final_prediction = self._weighted_average(current_predictions)
        print(f"Final prediction: {final_prediction:.3f}")
        return {"final_prediction": final_prediction, "discussion_history": discussion_history}

class EvaluateAgent(ReconcileAgent):
    """Evaluator Agent for assessing report quality."""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official"):
        super().__init__(agent_id, model_key)

    def evaluate_final_report(self, original_question: str, final_report_explanation: str, final_report_prediction: float, task_type: str, label: str) -> Dict[str, Any]:
        """Evaluates the AI-generated final patient report for trustworthiness."""
        system_message = {"role": "system", "content": prompt_template.REPORT_EVALUATOR_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.REPORT_EVALUATOR_USER.format(original_question=original_question, final_report=final_report_explanation, final_explanation=final_report_explanation, final_prediction=final_report_prediction, task_type=task_type, true_label=label)}
        response_text = self.call_llm([system_message, user_message])
        try:
            result = json.loads(preprocess_response_string(response_text))
            for dim in ["accuracy", "explainability", "safety"]:
                score = result.get(dim, {}).get("score", 1)
                result[dim]["score"] = max(1, min(5, int(score)))
        except (json.JSONDecodeError, ValueError, TypeError):
            result = parse_structured_output_for_final_report(response_text)
        return result

def main():
    parser = argparse.ArgumentParser(description="Run the Reconcile framework on EHR predictive modeling tasks")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
    parser.add_argument("--task", "-t", type=str, required=True, help="Prediction task")
    parser.add_argument("--agents", nargs='+', default=["deepseek-v3-official", "deepseek-v3-official", "deepseek-v3-official"], help="List of agent model keys")
    parser.add_argument("--max_rounds", type=int, default=2, help="Maximum number of discussion rounds")
    args = parser.parse_args()

    # Setup directories and paths
    method = "ReConcile"
    logs_dir = os.path.join("logs", "ehr", args.dataset, args.task, method)
    os.makedirs(logs_dir, exist_ok=True)
    data_path = os.path.join("my_datasets", "ehr", args.dataset, "processed", f"ehr_{args.task}_test.json")
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    agent_configs = [{"agent_id": f"agent_{i+1}", "model_key": model_key} for i, model_key in enumerate(args.agents)]
    print(f"Configured {len(agent_configs)} agents: {[cfg['model_key'] for cfg in agent_configs]}")

    # Main processing loop
    for item in tqdm(data, desc=f"Processing {args.dataset} ({args.task})"):
        qid = item.get("qid")
        result_path = os.path.join(logs_dir, f"ehr_{qid}-result.json")
        if os.path.exists(result_path):
            print(f"Skipping {qid} (already processed)")
            continue

        try:
            start_time = time.time()
            coordinator = ReconcileCoordinator(agent_configs, args.max_rounds)
            discussion_result = coordinator.run_discussion(item["question"])

            result = {
                "qid": qid, "question": item["question"][-1], "ground_truth": item.get("ground_truth"),
                "predicted_value": discussion_result["final_prediction"],
                "case_history": discussion_result, "processing_time": time.time() - start_time,
                "timestamp": int(time.time())
            }
            save_json(result, result_path)
        except Exception as e:
            print(f"Error processing item {qid}: {e}")

if __name__ == "__main__":
    main()