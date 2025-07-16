import math

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar


class SigmoidTemperatureScaling(nn.Module):
    """
    适用于Sigmoid输出的温度缩放校准模型。
    学习一个温度参数 T，并将 logits 除以 T 后再通过 Sigmoid 得到校准后的概率。
    主要用于二分类或多标签分类。
    """
    def __init__(self, initial_temperature=1.2): # 允许传入初始温度
        super(SigmoidTemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature, requires_grad=False)

    def forward(self, logits):
        """
        使用当前温度调整 logits。
        """
        return self.calibrate(logits)

    def calibrate(self, logits):
        """
        使用学习到的温度调整 logits。
        Args:
            logits (torch.Tensor): 模型的原始 logits 输出。
                                   对于二分类，形状通常为 (N, 1)。
                                   对于多标签分类，形状为 (N, C)，N是样本数，C是类别数。
        Returns:
            torch.Tensor: 校准后的 logits。
        """
        current_temp = torch.max(self.temperature, torch.tensor(1e-6, device=self.temperature.device))
        return logits / current_temp

    def fit(self, logits_cal, labels_cal):
        """
        在校准数据集上学习最优的温度参数 T。
        Args:
            logits_cal (torch.Tensor): 校准数据集上的原始 logits。
                                       二分类: (N, 1) 或 (N,)
                                       多标签: (N, C)
            labels_cal (torch.Tensor): 校准数据集上的真实标签 (0.0或1.0)。
                                       二分类: (N, 1) 或 (N,)
                                       多标签: (N, C)
        """
        logits_cal_np = logits_cal.detach().cpu().numpy()
        labels_cal_np = labels_cal.float().detach().cpu().numpy()

        # 定义目标函数 (BCE Loss)
        def bce_criterion_for_optimizer(temp_val_scalar):
            temp_val = float(temp_val_scalar)
            temp_val = max(temp_val, 1e-6)
            print(f"  Optimizer attempting temp_val (raw): {temp_val:.6f}, (bounded): {temp_val:.6f}")

            scaled_logits_np = logits_cal_np / temp_val
            if np.any(np.isinf(scaled_logits_np)) or np.any(np.isnan(scaled_logits_np)):
                print(f"    WARNING: scaled_logits_np contains Inf/NaN for temp_val={temp_val}.")
                return 1e20

            scaled_logits_tensor = torch.from_numpy(logits_cal_np / temp_val).float().to(logits_cal.device)
            labels_tensor = torch.from_numpy(labels_cal_np).float().to(logits_cal.device)

            # 确保标签和logits的形状匹配BCEWithLogitsLoss的要求
            # BCEWithLogitsLoss 期望 input 和 target 具有相同的形状
            if scaled_logits_tensor.ndim == 1: # (N,)
                scaled_logits_tensor = scaled_logits_tensor.unsqueeze(1) # (N,1)
            if labels_tensor.ndim == 1: # (N,)
                labels_tensor = labels_tensor.unsqueeze(1) # (N,1)

            # 如果是多标签且原始logits是 (N,C)，标签也应该是 (N,C)
            # 如果是二分类且原始logits是 (N,1) 或 (N,)，标签也应该是 (N,1) 或 (N,) 后调整为 (N,1)

            # 对于多标签情况，如果logits是(N, C)，labels也应该是(N, C)
            # 对于二分类，如果logits是(N, 1)或(N)，labels也应该是(N, 1)或(N) -> 调整为(N,1)
            # 此处假设传入的 labels_cal 和 logits_cal 已经基本匹配，除了可能需要 unsqueeze

            if scaled_logits_tensor.shape != labels_tensor.shape:
                # 尝试在二分类情况下修复维度不匹配
                if scaled_logits_tensor.shape[0] == labels_tensor.shape[0] and \
                   (scaled_logits_tensor.ndim == 1 or scaled_logits_tensor.shape[1] == 1) and \
                   (labels_tensor.ndim == 1 or labels_tensor.shape[1] == 1):
                    if scaled_logits_tensor.ndim == 1:
                        scaled_logits_tensor = scaled_logits_tensor.unsqueeze(1)
                    if labels_tensor.ndim == 1:
                        labels_tensor = labels_tensor.unsqueeze(1)
                else:
                    # 对于多标签，要求形状必须严格一致
                    if logits_cal_np.ndim > 1 and logits_cal_np.shape[1] > 1 : # 多标签
                         assert scaled_logits_tensor.shape == labels_tensor.shape, \
                             f"Shape mismatch for multi-label: logits {scaled_logits_tensor.shape}, labels {labels_tensor.shape}"


            loss_fn = nn.BCEWithLogitsLoss() # 适用于Sigmoid输出的logits
            loss = loss_fn(scaled_logits_tensor, labels_tensor)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN/Inf loss detected for temp_val: {temp_val}.")
                return 1e20

            print(f"    Loss for temp_val={temp_val:.6f}: {loss.item():.8f}") # 打印计算出的损失
            return loss.item()

        # 初始温度猜测
        # .item() 将单元素张量转换为Python标量
        initial_temp = self.temperature.item()
        if initial_temp <= 0: # 确保初始猜测是正数
            initial_temp = 1.5 # 默认一个合理值

        print("--- Manual BCE Criterion Check for Temperature Sensitivity ---")
        # 生成一系列测试温度点，例如从0.1到5.0，包含你的初始温度
        # 确保下限不小于 minimize 中的 bounds[0][0] (例如 1e-5)
        test_temps_manual = np.linspace(max(1e-5, initial_temp - 1.1), initial_temp + 3.8, num=30) # 更宽的范围，更多的点
        test_temps_manual = np.clip(test_temps_manual, 1e-5, None) # 再次确保下限

        temp_losses_manual = []
        print("  Temp   | Loss")
        print("  -------|----------")
        for tt_manual in test_temps_manual:
            try:
                # 直接调用你的目标函数来计算损失
                loss_val_manual = bce_criterion_for_optimizer(tt_manual)
                print(f"  {tt_manual:<6.4f} | {loss_val_manual:.8f}")
                if not (math.isnan(loss_val_manual) or math.isinf(loss_val_manual)):
                    temp_losses_manual.append((tt_manual, loss_val_manual))
            except Exception as e_manual_test:
                print(f"  {tt_manual:<6.4f} | ERROR: {e_manual_test}")
        print("----------------------------------------------------------")

        print(f"--- Starting optimization with minimize_scalar, initial_temp_guess_for_bracket_or_brent_is_around: {initial_temp} ---")

        try:
            res = minimize_scalar(
                bce_criterion_for_optimizer, # 你的目标函数
                bounds=(1e-5, 10.0),        # 设置一个合理的实际上下限，例如 (1e-5, 10.0)
                method='bounded'             # 'bounded' 方法对于有界单变量优化非常有效
                # 你也可以尝试不带 bracket/bounds 的 Brent 方法，但 bounded 通常更稳健
                # bracket=(0.1, initial_temp, 5.0) # 或者为Brent等方法提供一个包含最小值的区间
            )
            success = res.success
            best_temp = res.x
            message = res.message
            fun_val = res.fun

            print(f"minimize_scalar result: success={success}, best_temp={best_temp:.4f}, final_loss={fun_val:.6f}, message='{message}'")

            if success: # 或者检查 res.fun 是否是有效数值且比初始损失低
                self.temperature.data = torch.ones(1, device=logits_cal.device) * best_temp
                print(f"Optimal temperature found with minimize_scalar: {best_temp:.4f}")
            else:
                print(f"Temperature optimization with minimize_scalar failed or did not improve: {message}")
                print(f"Optimizer details: {res}")
                print("Using initial temperature or last successful temperature (if any).")
                # 保持 self.temperature 为优化前的值 (即 initial_temp)

        except Exception as e_scalar_opt:
            print(f"Error during minimize_scalar: {e_scalar_opt}")
            print("Using initial temperature due to optimization error.")
            # self.temperature 保持不变 (即 initial_temp)

        return self


def find_optimal_threshold(y_true_np, y_probs_np, metric='f1'):
    """
    在给定的真实标签和预测概率上寻找最优阈值。
    Args:
        y_true_np (np.array): 真实标签 (0 或 1)。
        y_probs_np (np.array): 预测概率。
        metric (str): 用于优化的指标 ('f1', 'youden', 'pr_auc_based').
    Returns:
        float: 最佳阈值。
        float: 该阈值下的最佳分数。
    """
    thresholds = np.linspace(0.01, 0.99, 200) # 候选阈值范围，可以调整数量
    best_threshold = 0.5
    best_score = -1

    if y_true_np.size == 0 or y_probs_np.size == 0:
        print("警告: 标签或概率为空，无法寻找最优阈值。返回默认值 0.5。")
        return 0.5, 0.0

    if len(np.unique(y_true_np)) < 2 and metric != 'accuracy_if_single_class': # 检查是否有多个类别
        print(f"警告: 验证集只包含一个类别 ({np.unique(y_true_np)}). 无法基于 {metric} 合理地选择阈值。返回默认阈值0.5。")
        # 在这种情况下，F1分数等可能未定义或无意义
        return 0.5, 0.0 # 或者返回一个基于多数类的简单阈值和对应的准确率

    print(f"开始寻找最优阈值，基于指标: {metric}")
    for threshold in thresholds:
        y_pred_binary = (y_probs_np >= threshold).astype(int)
        current_score = 0
        try:
            if metric == 'f1':
                # 对于F1，pos_label=1是通常情况，如果你的正类是0，需要修改
                current_score = f1_score(y_true_np, y_pred_binary, pos_label=1, zero_division=0)
            elif metric == 'youden':
                # Youden's J = Sensitivity + Specificity - 1
                # Sensitivity = TP / (TP + FN)
                # Specificity = TN / (TN + FP)
                tp = np.sum((y_pred_binary == 1) & (y_true_np == 1))
                fn = np.sum((y_pred_binary == 0) & (y_true_np == 1))
                tn = np.sum((y_pred_binary == 0) & (y_true_np == 0))
                fp = np.sum((y_pred_binary == 1) & (y_true_np == 0))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                current_score = sensitivity + specificity - 1
            elif metric == 'pr_auc_based': # 寻找最大化 F1 的点，或者在 PR 曲线上选择一个点
                # 这更复杂，通常直接优化F1更简单。
                # 如果要基于 PR 曲线，你可能需要找到平衡点，或 G-Mean 等。
                # 这里简单示例还是用F1
                current_score = f1_score(y_true_np, y_pred_binary, pos_label=1, zero_division=0)

            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
        except ValueError as e:
            # print(f"计算指标时出错 (阈值={threshold:.3f}): {e}。跳过此阈值。")
            pass # 有时当一个类别完全没有被预测时，某些指标会报错

    print(f"找到最优阈值: {best_threshold:.4f}，对应 {metric} 分数: {best_score:.4f}")
    return best_threshold, best_score