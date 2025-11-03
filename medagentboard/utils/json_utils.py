import json
import os
import re
import logging
from typing import Any, List, Dict, Iterator

def get_logger(log_path: str) -> logging.Logger:
    """
    Configures and returns a logger instance to write logs to a file.

    Args:
        log_path: The file path where the log will be saved.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    # Prevent adding duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data to a JSON file

    Args:
        data: Data to be saved (must be JSON serializable)
        filepath: Save path including filename
        indent: JSON indentation format, defaults to 2
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    print(f"Data saved to: {filepath}")

def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def save_jsonl(data_list: List[Any], filepath: str) -> None:
    """
    Save a list of items to a JSONL file (each item on a separate line)

    Args:
        data_list: List of items to be saved (each must be JSON serializable)
        filepath: Save path including filename
    """
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Write JSONL file - one JSON object per line
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Data saved to JSONL file: {filepath}")

def load_jsonl(filepath: str) -> List[Any]:
    """
    Load data from a JSONL file (each line is a separate JSON object)

    Args:
        filepath: Path to JSONL file

    Returns:
        List of loaded items

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data_list = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data_list.append(json.loads(line))

    return data_list

def iter_jsonl(filepath: str) -> Iterator[Any]:
    """
    Iterate through items in a JSONL file without loading everything into memory

    Args:
        filepath: Path to JSONL file

    Yields:
        Each item from the JSONL file

    Raises:
        FileNotFoundError: When file doesn't exist
        json.JSONDecodeError: When JSON format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                yield json.loads(line)

def merge_json_files(filepaths: List[str], output_filepath: str) -> None:
    """
    Merge multiple JSON files (assuming each contains list data)

    Args:
        filepaths: List of JSON file paths to merge
        output_filepath: Output file path for merged data
    """
    merged_data = []

    for filepath in filepaths:
        data = load_json(filepath)
        if isinstance(data, list):
            merged_data.extend(data)
        else:
            merged_data.append(data)

    save_json(merged_data, output_filepath)
    print(f"Merged {len(filepaths)} files into: {output_filepath}")

def update_json(filepath: str, new_data: Any) -> None:
    """
    Update an existing JSON file

    Args:
        filepath: Path to JSON file to update
        new_data: New data (if dict, will merge with existing; if list, will append to existing)
    """
    if os.path.exists(filepath):
        existing_data = load_json(filepath)

        if isinstance(existing_data, dict) and isinstance(new_data, dict):
            # If both are dictionaries, merge them
            existing_data.update(new_data)
        elif isinstance(existing_data, list) and isinstance(new_data, list):
            # If both are lists, extend them
            existing_data.extend(new_data)
        else:
            # Other cases, replace completely
            existing_data = new_data
    else:
        existing_data = new_data

    save_json(existing_data, filepath)
    print(f"Updated file: {filepath}")

def preprocess_response_string(response_string: str) -> str:
    """
    Cleans and prepares a raw LLM response string for JSON parsing.
    It removes markdown code blocks and trailing commas.

    Args:
        response_string: The raw string response from the LLM.

    Returns:
        A cleaned string ready for JSON parsing.
    """
    # Remove markdown JSON code blocks
    cleaned_string = re.sub(r'```json\s*|\s*```', '', response_string.strip())
    # Remove trailing commas that can cause JSON parsing errors
    cleaned_string = re.sub(r',\s*([}\]])', r'\1', cleaned_string)
    return cleaned_string

def parse_structured_output(response_text: str) -> Dict[str, Any]:
    """
    Robustly parses an LLM response to extract a structured dictionary.
    First, it tries to parse the text as clean JSON. If that fails, it uses
    line-by-line parsing as a fallback.

    Args:
        response_text: The text response from the LLM.

    Returns:
        A dictionary containing the parsed fields.
    """
    try:
        # Try parsing as JSON first
        return json.loads(preprocess_response_string(response_text))
    except json.JSONDecodeError:
        # Fallback to line-by-line parsing if JSON is invalid
        result = {}
        lines = response_text.strip().split('\n')
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                # Clean up key and value
                key = key.strip().lower().replace("\"", "")
                value = value.strip().replace("\"", "")
                result[key] = value

        # Ensure essential fields exist with default values
        if "prediction" not in result and "answer" not in result:
            result["prediction"] = 0.5
        if "explanation" not in result and "reasoning" not in result and "reason" not in result:
            result["explanation"] = "No structured explanation found in response."

        return result

def parse_structured_output_for_final_report(response_text: str) -> Dict[str, Any]:
    """
    A fallback parser specifically for the final report evaluation agent's response.
    It extracts scores and reasons for different dimensions.

    Args:
        response_text: The raw response from the evaluation LLM.

    Returns:
        A dictionary with structured evaluation results.
    """
    result = {
        "accuracy": {"score": 1, "reason": "Could not parse response."},
        "safety": {"score": 1, "reason": "Could not parse response."},
        "explainability": {"score": 1, "reason": "Could not parse response."},
    }
    lines = response_text.split('\n')
    current_dim = None

    for line in lines:
        line_lower = line.strip().lower()
        if "accuracy:" in line_lower:
            current_dim = "accuracy"
        elif "safety:" in line_lower:
            current_dim = "safety"
        elif "explainability:" in line_lower:
            current_dim = "explainability"

        if current_dim:
            if "score:" in line_lower:
                try:
                    score_str = line.split(":", 1)[1].strip().split(" ")[0]
                    score = int(float(score_str)) # Handle floats like 4.0
                    result[current_dim]["score"] = max(1, min(5, score))
                except (ValueError, IndexError):
                    pass # Keep default score
            if "reason:" in line_lower:
                reason = line.split(":", 1)[1].strip()
                result[current_dim]["reason"] = reason

    return result