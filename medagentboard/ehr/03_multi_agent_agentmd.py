# ehr_multi_agent_agentmd.py

import os
import json
import time
import asyncio
import argparse
import traceback
from tqdm import tqdm
from enum import Enum
from typing import Dict, Any, List
from io import StringIO
import contextlib

from openai import OpenAI

from medagentboard.utils import prompt_template
from medagentboard.utils.llm_configs import LLM_MODELS_SETTINGS
from medagentboard.utils.json_utils import get_logger, preprocess_response_string, save_json, load_json, parse_structured_output, parse_structured_output_for_final_report


RISK_CALCULATORS = {
    "CURB-65 Score for Pneumonia Severity": {
        "description": "Assesses the severity of community-acquired pneumonia to help determine the need for hospitalization. It uses five factors: Confusion, Urea > 7 mmol/L, Respiratory rate >= 30/min, low Blood pressure, and age >= 65 years.",
        "code": """
def curb65_score(confusion: bool, urea_mmol_per_l: float, respiratory_rate: int, systolic_bp: int, diastolic_bp: int, age: int) -> dict:
    '''
    Calculates the CURB-65 score for pneumonia severity.
    Each parameter corresponds to one of the five criteria.
    '''
    score = 0
    if confusion: score += 1
    if urea_mmol_per_l > 7: score += 1
    if respiratory_rate >= 30: score += 1
    if systolic_bp < 90 or diastolic_bp <= 60: score += 1
    if age >= 65: score += 1

    interpretation_map = {
        0: "0.7% 30-day mortality risk. Consider outpatient treatment.",
        1: "3.2% 30-day mortality risk. Consider outpatient treatment.",
        2: "13.0% 30-day mortality risk. Consider hospital admission.",
        3: "17.0% 30-day mortality risk. Urgent hospital admission required.",
        4: "41.5% 30-day mortality risk. Urgent hospital admission, consider ICU.",
        5: "57.0% 30-day mortality risk. Urgent hospital admission, consider ICU."
    }

    return {
        "score": score,
        "interpretation": interpretation_map.get(score, "Score out of range.")
    }
"""
    },
    "CHADS2 Score for Stroke Risk in Atrial Fibrillation": {
        "description": "Estimates the risk of stroke in patients with non-rheumatic atrial fibrillation (AF), a common type of irregular heartbeat. Used to determine the need for therapy with anticoagulation. It assesses: Congestive heart failure, Hypertension, Age >= 75, Diabetes, and prior Stroke/TIA.",
        "code": """
def chads2_score(congestive_heart_failure: bool, hypertension: bool, age: int, diabetes: bool, stroke_or_tia_history: bool) -> dict:
    '''
    Calculates the CHADS2 score for stroke risk in patients with atrial fibrillation.
    '''
    score = 0
    if congestive_heart_failure: score += 1
    if hypertension: score += 1
    if age >= 75: score += 1
    if diabetes: score += 1
    if stroke_or_tia_history: score += 2

    risk_map = {
        0: "1.9% risk of stroke per year.",
        1: "2.8% risk of stroke per year.",
        2: "4.0% risk of stroke per year.",
        3: "5.9% risk of stroke per year.",
        4: "8.5% risk of stroke per year.",
        5: "12.5% risk of stroke per year.",
        6: "18.2% risk of stroke per year."
    }

    return {
        "score": score,
        "interpretation": risk_map.get(score, "Score out of range.")
    }
"""
    }
}


def get_calculator_descriptions() -> str:
    """Returns a formatted string of all calculator names and their descriptions."""
    descriptions = []
    for i, (name, data) in enumerate(RISK_CALCULATORS.items()):
        descriptions.append(f"{i+1}. **{name}**:\n   - {data['description']}")
    return "\n".join(descriptions)


class AgentType(Enum):
    """Enumeration for agent types in the AgentMD framework."""
    CALCULATOR = "Calculator"
    COORDINATOR = "Coordinator"
    EVALUATOR = "Evaluator"


class BaseAgent:
    """Base class for all agents in the AgentMD framework. (Reused from ColaCare)"""
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
        self.client = OpenAI(api_key=model_settings["api_key"], base_url=model_settings["base_url"])
        self.model_name = model_settings["model_name"]

    def call_llm(self,
        system_message: Dict[str, str],
        user_message: Dict[str, Any],
        max_retries: int = 3) -> str:
        retries = 0
        while retries < max_retries:
            try:
                if self.logger: self.logger.info(f"Agent {self.agent_id} calling LLM...")
                completion = self.client.chat.completions.create(
                    model=self.model_name, messages=[system_message, user_message]
                )
                response = completion.choices[0].message.content
                if self.logger: self.logger.info(f"Agent {self.agent_id} received response: {response[:150]}...")
                return response
            except Exception as e:
                retries += 1
                if self.logger: self.logger.error(f"LLM API call error (attempt {retries}/{max_retries}): {e}")
                if retries >= max_retries: raise Exception(f"LLM API call failed: {e}")
                time.sleep(2)


class CalculatorAgent(BaseAgent):
    """
    The core worker agent in the AgentMD framework.
    It performs the three main steps: tool selection, execution, and summarization.
    """
    def __init__(self, agent_id: str, model_key: str, logger=None):
        super().__init__(agent_id, AgentType.CALCULATOR, model_key, logger=logger)
        if self.logger: self.logger.info(f"Initializing CalculatorAgent, ID: {agent_id}, Model: {model_key}")

    async def select_tools(self, patient_case: str) -> List[str]:
        """Selects relevant clinical calculators for a given patient case."""
        if self.logger: self.logger.info(f"CalculatorAgent {self.agent_id} starting tool selection.")
        tool_descriptions = get_calculator_descriptions()
        system_message = {"role": "system", "content": prompt_template.AGENTMD_TOOL_SELECTION_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.AGENTMD_TOOL_SELECTION_USER.format(patient_case=patient_case, tool_descriptions=tool_descriptions)}

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            selected_tools = result.get("selected_tools", [])
            if self.logger: self.logger.info(f"Selected tools: {selected_tools}")
            return selected_tools
        except json.JSONDecodeError:
            if self.logger: self.logger.error("Failed to parse JSON from tool selection response.")
            return []

    async def execute_tool(self, patient_case: str, tool_name: str) -> Dict[str, Any]:
        """Executes a single selected calculator."""
        if self.logger: self.logger.info(f"CalculatorAgent {self.agent_id} executing tool: {tool_name}")
        if tool_name not in RISK_CALCULATORS:
            return {"error": f"Tool '{tool_name}' not found in the library."}

        tool_code = RISK_CALCULATORS[tool_name]["code"]
        system_message = {"role": "system", "content": prompt_template.AGENTMD_TOOL_EXECUTION_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.AGENTMD_TOOL_EXECUTION_USER.format(patient_case=patient_case, tool_name=tool_name, tool_code=tool_code)}

        # LLM generates the function call code
        execution_code = self.call_llm(system_message, user_message)
        if self.logger: self.logger.info(f"Generated execution code: {execution_code}")

        # Safely execute the generated code
        # NOTE: In a production environment, this should be heavily sandboxed.
        full_code = tool_code + "\n" + execution_code
        try:
            # Capture the output of the print statement
            stdout_capture = StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                exec(full_code, {})

            output = stdout_capture.getvalue()
            # The output is often a string representation of a dict, so we evaluate it
            # Using eval is risky, ast.literal_eval is safer
            import ast
            result = ast.literal_eval(output.strip())
            if self.logger: self.logger.info(f"Execution result for {tool_name}: {result}")
            return {"tool_name": tool_name, "result": result}
        except Exception as e:
            if self.logger: self.logger.error(f"Error executing tool {tool_name}: {e}\nCode: {full_code}")
            return {"tool_name": tool_name, "error": str(e)}


class CoordinatorAgent(BaseAgent):
    """Orchestrates the AgentMD workflow, analogous to ColaCare's MetaAgent."""
    def __init__(self, agent_id: str, model_key: str, logger=None):
        super().__init__(agent_id, AgentType.COORDINATOR, model_key, logger=logger)
        if self.logger: self.logger.info(f"Initializing CoordinatorAgent, ID: {agent_id}, Model: {model_key}")

    def synthesize_results(self, patient_case: str, execution_results: List[Dict], task_type: str) -> Dict[str, Any]:
        """Synthesizes execution results into a final report, similar to MetaAgent's role."""
        if self.logger: self.logger.info("CoordinatorAgent synthesizing final report.")

        results_str = json.dumps(execution_results, indent=2)
        system_message = {"role": "system", "content": prompt_template.AGENTMD_RESULT_SYNTHESIS_SYSTEM}
        user_message = {"role": "user", "content": prompt_template.AGENTMD_RESULT_SYNTHESIS_USER.format(patient_case=patient_case, execution_results=results_str, task_type=task_type)}

        response_text = self.call_llm(system_message, user_message)

        try:
            result = json.loads(preprocess_response_string(response_text))
            pred = result.get("prediction", 0.501)
            result["prediction"] = max(0.0, min(1.0, float(pred)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger: self.logger.warning("Coordinator response is not valid JSON, using fallback.")
            result = parse_structured_output(response_text)
            result["response_text"] = response_text

        self.memory.append({"type": "synthesis", "content": result})
        return result


class EvaluateAgent(BaseAgent):
    """Evaluates the final report from the AgentMD process. (Reused and adapted from ColaCare)"""
    def __init__(self, agent_id: str, model_key: str = "deepseek-v3-official", logger=None):
        super().__init__(agent_id, AgentType.EVALUATOR, model_key, logger=logger)
        if self.logger: self.logger.info(f"Initializing EvaluatorAgent, ID: {agent_id}, Model: {model_key}")

    def evaluate_final_report(self, original_question, final_report, final_explanation, final_prediction, task_type, label) -> Dict[str, Any]:
        if self.logger: self.logger.info(f"EvaluatorAgent evaluating final report for task '{task_type}'.")
        system_message = {"role": "system", "content": prompt_template.AGENTMD_EVALUATE_SYSTEM}
        user_message = {
            "role": "user",
            "content": prompt_template.AGENTMD_EVALUATE_USER.format(
                original_question=original_question,
                final_report=final_report,
                final_explanation=final_explanation,
                final_prediction=final_prediction,
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
            if self.logger: self.logger.info("Report evaluation successfully parsed.")
        except (json.JSONDecodeError, ValueError, TypeError):
            if self.logger: self.logger.warning("Report evaluation is not valid JSON, using fallback.")
            result = parse_structured_output_for_final_report(response_text)

        return result


class AgentMDProcess:
    """Coordinator for the AgentMD tool-use process, analogous to MDTConsultation."""
    def __init__(self,
                 calculator_agent_model: str = "deepseek-v3-official",
                 coordinator_model: str = "deepseek-v3-official",
                 evaluator_model: str = "deepseek-v3-official",
                 logger=None):
        self.logger = logger
        self.calculator_agent = CalculatorAgent("calculator-1", calculator_agent_model, logger=logger)
        self.coordinator_agent = CoordinatorAgent("coordinator-1", coordinator_model, logger=logger)
        self.evaluator_agent = EvaluateAgent("evaluator-1", evaluator_model, logger=logger)
        if self.logger: self.logger.info("Initialized AgentMDProcess.")

    async def run_process(self, qid: str, question: str, task_type: str, label: str) -> Dict[str, Any]:
        """Runs the full AgentMD process for a given case."""
        if self.logger: self.logger.info(f"Starting AgentMD process for case {qid} (Task: {task_type})")

        process_history = {}

        # Step 1: Tool Selection
        selected_tools = await self.calculator_agent.select_tools(question)
        process_history["tool_selection"] = {"selected_tools": selected_tools}

        if not selected_tools:
            if self.logger: self.logger.warning("No tools selected. Aborting process.")
            return {"error": "No tools were selected for this case.", "final_decision": {"prediction": 0.501}}

        # Step 2: Tool Execution (concurrently)
        tasks = [self.calculator_agent.execute_tool(question, tool_name) for tool_name in selected_tools]
        execution_results = await asyncio.gather(*tasks)
        process_history["tool_execution"] = execution_results

        # Step 3: Result Synthesis
        final_decision = self.coordinator_agent.synthesize_results(question, execution_results, task_type)
        process_history["final_decision"] = final_decision

        # Step 4: Final Report Evaluation
        report_evaluation = self.evaluator_agent.evaluate_final_report(
            original_question=question,
            final_report=final_decision.get('report', ''),
            final_explanation=final_decision.get('explanation', ''),
            final_prediction=final_decision.get('prediction', 0.501),
            task_type=task_type,
            label=label
        )
        process_history["report_trustworthiness_evaluation"] = report_evaluation

        return process_history


async def main():
    parser = argparse.ArgumentParser(description="Run AgentMD tool-use framework on EHR datasets")
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv", "cdsl", "esrd", "obstetrics"], help="Specify dataset name")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission", "sptb"], help="Prediction task")
    parser.add_argument("--modality", "-mo", type=str, default="ehr", choices=["ehr", "note", "mm"], help="Modality of the dataset")
    parser.add_argument("--agent_model", type=str, default="deepseek-v3-official", help="Model for all agents")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index of the data chunk to process")
    args = parser.parse_args()

    method = "AgentMD"
    save_dir = os.path.join("logs", args.dataset, args.task, method, f"{args.modality}_{args.agent_model}")
    logs_dir, results_dir, error_dir = [os.path.join(save_dir, d) for d in ["logs", "results", "error"]]
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    data_path = f"./my_datasets/ehr/{args.dataset}/processed/{args.modality}_{args.task}_test.json"
    data = load_json(data_path)
    print(f"Loaded {len(data)} samples from {data_path}")

    chunk_size = 100
    data_to_process = data[args.start_index : args.start_index + chunk_size]
    if not data_to_process:
        print(f"No data to process for start_index {args.start_index}. Exiting.")
        return
    print(f"Processing a chunk of {len(data_to_process)} samples from index {args.start_index}.")

    for item in tqdm(data_to_process, desc=f"Running AgentMD on {args.dataset} chunk"):
        qid_str = str(item["qid"])
        save_file_name = f"ehr_{qid_str}-result.json"

        if os.path.exists(os.path.join(results_dir, save_file_name)):
            print(f"Skipping {qid_str} - already processed")
            continue

        logger = get_logger(os.path.join(logs_dir, f"ehr_{qid_str}.log"))

        try:
            start_time = time.time()
            # In AgentMD, the question is a single string, not a list like in ColaCare
            # We'll use the first question if it's a list for compatibility.
            patient_case = item["question"][0] if isinstance(item["question"], list) else item["question"]

            agentmd_process = AgentMDProcess(
                calculator_agent_model=args.agent_model,
                coordinator_model=args.agent_model,
                evaluator_model=args.agent_model,
                logger=logger
            )
            result = await agentmd_process.run_process(
                qid=item["qid"], question=patient_case,
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
                traceback.print_exc(file=f)

if __name__ == "__main__":
    asyncio.run(main())