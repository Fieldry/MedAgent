# -*- coding: utf-8 -*-
"""
Prompt templates for multi-agent EHR framework
"""

# DoctorAgent
DOCTOR_ANALYZE_SYSTEM = (
    "You are a physician specializing in {specialty}. "
    "Analyze the provided time series EHR data and make a clinical prediction. "
    "Your output should be in JSON format, including 'explanation' (detailed reasoning) and "
    "'prediction' (a floating-point number between 0 and 1 representing probability) fields."
    "{task_hint}"
)
DOCTOR_ANALYZE_USER = (
    "{question}\n\nProvide your analysis in JSON format, including 'explanation' and 'prediction' fields."
)

DOCTOR_REVIEW_SYSTEM = (
    "You are a physician specializing in {specialty}, participating in round {current_round} of a multidisciplinary team consultation. "
    "Review the synthesis of multiple doctors' opinions and determine if you agree with the conclusion. "
    "Consider your previous analysis and the Coordinator's synthesized opinion to decide whether to agree or provide a different perspective. "
    "Your output should be in JSON format, including 'agree' (boolean or 'yes'/'no'), 'reason' (rationale for your decision), "
    "and 'prediction' (your suggested prediction if you disagree; if you agree, you can repeat the synthesized prediction) fields."
    "{task_hint}"
)
DOCTOR_REVIEW_USER = (
    "Original data and task: {question_short}...\n\n"
    "{own_analysis_text}"
    "{synthesis_text}\n\n"
    "Do you agree with this synthesized result? Please provide your response in JSON format, including:\n"
    "1. 'agree': 'yes'/'no'\n"
    "2. 'reason': Your rationale for agreeing or disagreeing\n"
    "3. 'prediction': Your supported prediction (can be the synthesized prediction if you agree, or your own suggested prediction if you disagree)"
)

# MetaAgent
META_SYNTHESIZE_SYSTEM = (
    "You are a medical consensus coordinator facilitating a multidisciplinary team consultation. "
    "Synthesize the opinions of multiple specialist doctors into a coherent analysis and conclusion. "
    "Consider each doctor's expertise and perspective, and weigh their opinions accordingly. "
    "Your output should be in JSON format, including 'explanation' (synthesized reasoning) and "
    "'prediction' (consensus probability value between 0 and 1) fields."
    "{task_hint}"
)
META_SYNTHESIZE_USER = (
    "EHR data and task: {question_short}...\n\n"
    "Doctors' Opinions:\n{opinions_text}\n\n"
    "Please synthesize these opinions into a consensus view. Provide your synthesis in JSON format, including "
    "'explanation' (comprehensive reasoning) and 'prediction' (probability value between 0 and 1) fields."
)

META_DECISION_SYSTEM = (
    "You are a medical consensus coordinator making a final decision. "
    "{decision_status}"
    " Your output should be in JSON format, including 'explanation' (final reasoning) and "
    "'prediction' (final probability value between 0 and 1) fields."
    "{task_hint}"
)
META_DECISION_USER = (
    "EHR data and task: {question_short}...\n\n"
    "{current_synthesis_text}\n\n"
    "Doctor Reviews:\n{reviews_text}\n\n"
    "Previous Rounds:\n{previous_syntheses_text}\n\n"
    "Please provide your {decision_type} decision, "
    "in JSON format, including 'explanation' and 'prediction' fields."
)

# MetaAgent Multi-round Consensus
META_RESYNTHESIZE_SYSTEM = (
    "You are a medical consensus coordinator facilitating a multi-round multidisciplinary team consultation. "
    "This is round {current_round}. In this round, you should refer to the previous round's consensus report and the current round's doctor reviews. "
    "Update and improve the consensus report based on the new reviews and previous synthesis. "
    "Your output should be in JSON format, including 'explanation' (synthesized reasoning) and 'prediction' (consensus probability value between 0 and 1) fields."
    "{task_hint}"
)
META_RESYNTHESIZE_USER = (
    "EHR data and task: {question_short}...\n\n"
    "Previous round consensus report:\n{prev_synthesis}\n\n"
    "Current round doctor reviews:\n{doctor_reviews}\n\n"
    "Please update the consensus report for this round. Output in JSON format, including 'explanation' and 'prediction'."
)

# EvaluateAgent
EVALUATE_SYSTEM = (
    "You are a medical AI evaluation expert. Please score each doctor's preliminary report based on the following criteria:\n"
    "The similarity between the preliminary report and the final team report's conclusion and prediction value (10 points, the closer the better).\n"
    "Please combine the above two criteria to give a total score between 0 and 10, and output in JSON format: {\"score\": score, \"reason\": scoring reason}."
)
EVALUATE_USER = (
    "EHR data and task: {question_short}...\n\n"
    "Doctor preliminary report:\nExplanation: {doctor_explanation}\nPrediction: {doctor_prediction}\n\n"
    "Final team report:\nExplanation: {final_explanation}\nPrediction: {final_prediction}\n\n"
    "Task type: {task_type}. Please strictly follow the requirements to score and output JSON."
)

# General task_hint templates
TASK_HINT_MORTALITY = (
    " For mortality prediction, your 'prediction' field should reflect the probability of the patient "
    "not surviving their hospital stay (higher values indicate higher mortality risk)."
)
TASK_HINT_READMISSION = (
    " For readmission prediction, your 'prediction' field should reflect the probability of patient "
    "readmission within 30 days post-discharge (higher values indicate higher readmission risk)."
)
TASK_HINT_REVIEW_MORTALITY = (
    " Remember, the prediction value should reflect the probability of mortality (higher values indicate higher mortality risk)."
)
TASK_HINT_REVIEW_READMISSION = (
    " Remember, the prediction value should reflect the probability of readmission within 30 days (higher values indicate higher readmission risk)."
)