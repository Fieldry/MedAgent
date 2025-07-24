#!/bin/bash

# --- 配置 ---
# 设置最大并发任务数。一个好的起点是你的 CPU 核心数。
MAX_JOBS=24
# 设置日志文件存放目录
LOG_DIR="experiment_logs"

# --- 准备工作 ---
# 创建日志目录，如果不存在的话
mkdir -p "$LOG_DIR"
echo "Logs will be saved in the '$LOG_DIR' directory."

# --- 定义所有实验 ---
# 我们将所有命令预先生成并存储在一个数组中
commands=()

# --- 生成命令列表 ---

echo "Generating command list for all experiments..."

# --------------------------------------------------------------------------------
# Block 1.1: ColaCare with deepseek-v3-official and RAG
# --------------------------------------------------------------------------------
# PATH_VAL="medagentboard.ehr.03_multi_agent_colacare"
# LLM_VAL="deepseek-v3-official"
# DATASET_TASKS=("mimic-iv:mortality" "esrd:mortality" "obstetrics:sptb" "mimic-iv:readmission" "cdsl:mortality")
# for dt in "${DATASET_TASKS[@]}"; do
#     IFS=":" read -r DATASET TASK <<< "$dt"
#     # 创建一个对文件名友好的标识符，替换 '.' 为 '_'
#     SAFE_PATH_VAL="${PATH_VAL//./_}"
#     LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${LLM_VAL}__with_rag.log"
#     CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK} --meta_model ${LLM_VAL} --doctor_models ${LLM_VAL} ${LLM_VAL} ${LLM_VAL} --evaluate_model ${LLM_VAL} -mo ehr --use_rag"
#     # 将标准输出和标准错误都重定向到日志文件
#     commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
# done

# --------------------------------------------------------------------------------
# Block 1.2: ColaCare with deepseek-v3-official and no RAG
# # --------------------------------------------------------------------------------
# PATH_VAL="medagentboard.ehr.03_multi_agent_colacare"
# LLM_VAL="deepseek-v3-official"
# DATASET_TASKS=("mimic-iv:mortality" "esrd:mortality" "obstetrics:sptb" "mimic-iv:readmission" "cdsl:mortality")
# for dt in "${DATASET_TASKS[@]}"; do
#     IFS=":" read -r DATASET TASK <<< "$dt"
#     SAFE_PATH_VAL="${PATH_VAL//./_}"
#     LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${LLM_VAL}__no_rag.log"
#     CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK} --meta_model ${LLM_VAL} --doctor_models ${LLM_VAL} ${LLM_VAL} ${LLM_VAL} --evaluate_model ${LLM_VAL} -mo ehr"
#     commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
# done

# --------------------------------------------------------------------------------
# Block 2: MedAgent
# --------------------------------------------------------------------------------
# PATH_VAL="medagentboard.ehr.03_multi_agent_medagent"
# LLM_VAL="deepseek-v3-official" # 假设 MedAgent 内部使用这个模型
# DATASET_TASKS=("mimic-iv:mortality" "esrd:mortality" "obstetrics:sptb" "mimic-iv:readmission" "cdsl:mortality")
# for dt in "${DATASET_TASKS[@]}"; do
#     IFS=":" read -r DATASET TASK <<< "$dt"
#     SAFE_PATH_VAL="${PATH_VAL//./_}"
#     LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${LLM_VAL}.log"
#     CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK}"
#     commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
# done

# --------------------------------------------------------------------------------
# Block 3: ReConcile
# --------------------------------------------------------------------------------
# PATH_VAL="medagentboard.ehr.03_multi_agent_reconcile"
# LLM_VAL="deepseek-v3-official" # 假设 ReConcile 内部使用这个模型
# DATASET_TASKS=("mimic-iv:mortality" "esrd:mortality" "obstetrics:sptb" "mimic-iv:readmission" "cdsl:mortality")
# for dt in "${DATASET_TASKS[@]}"; do
#     IFS=":" read -r DATASET TASK <<< "$dt"
#     SAFE_PATH_VAL="${PATH_VAL//./_}"
#     LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${LLM_VAL}.log"
#     CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK}"
#     commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
# done

# --------------------------------------------------------------------------------
# Block 4: ColaCare with other LLMs
# --------------------------------------------------------------------------------
# PATH_VAL="medagentboard.ehr.03_multi_agent_colacare"
# LLMS=("deepseek-r1-official") # "claude4" "o4-mini" "qwen3"
# DATASET_TASKS=("mimic-iv:readmission" "mimic-iv:mortality")
# for llm in "${LLMS[@]}"; do
#     for dt in "${DATASET_TASKS[@]}"; do
#         IFS=":" read -r DATASET TASK <<< "$dt"
#         SAFE_PATH_VAL="${PATH_VAL//./_}"
#         LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${llm}__with_rag.log"
#         CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK} --meta_model ${llm} --doctor_models ${llm} ${llm} ${llm} --evaluate_model deepseek-v3-official -mo ehr"
#         commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
#     done
# done

# --------------------------------------------------------------------------------
# Block 5: Single LLM
# --------------------------------------------------------------------------------
PATH_VAL="medagentboard.ehr.03_single_llm"
LLM_VAL="deepseek-v3-official"
DATASET_TASKS=("mimic-iv:mortality" "mimic-iv:readmission" "obstetrics:sptb" "cdsl:mortality") # "esrd:mortality"
for dt in "${DATASET_TASKS[@]}"; do
    IFS=":" read -r DATASET TASK <<< "$dt"
    SAFE_PATH_VAL="${PATH_VAL//./_}"
    LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${LLM_VAL}__ZeroShotLLM.log"
    CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK} --model ${LLM_VAL}"
    commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
    LOG_FILENAME="${SAFE_PATH_VAL}__${DATASET}__${TASK}__${LLM_VAL}__FewShotLLM.log"
    CMD="python -m ${PATH_VAL} -d ${DATASET} -t ${TASK} --model ${LLM_VAL} --few_shot"
    commands+=("$CMD > \"${LOG_DIR}/${LOG_FILENAME}\" 2>&1")
done

# --- 执行实验 ---

total_commands=${#commands[@]}
echo "--------------------------------------------------"
echo "Total experiments to run: $total_commands"
echo "Max concurrent jobs: $MAX_JOBS"
echo "--------------------------------------------------"

# 循环执行所有命令
for ((i=0; i<total_commands; i++)); do
    while [[ $(jobs -p | wc -l) -ge $MAX_JOBS ]]; do
        wait -n
    done

    cmd_with_redirect="${commands[$i]}"
    echo "[$((i+1))/$total_commands] Spawning job. Full command with logging:"
    # 打印将要执行的完整命令，包括日志重定向，方便调试
    echo "  -> $cmd_with_redirect"

    # 使用 eval 在后台执行命令。重定向操作也会被正确解析。
    (
      eval "$cmd_with_redirect"
    ) &
done

# --- 等待所有任务完成 ---
echo "All jobs have been spawned. Waiting for the remaining jobs to complete..."
wait
echo "========================================="
echo "All experiments have finished."
echo "You can check the output of each job in the '$LOG_DIR' directory."
echo "========================================="