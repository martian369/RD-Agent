"""
多臂赌博机(MAB)算法实现，用于在量化金融场景中智能选择因子或模型进行优化

该模块实现了线性汤普森采样算法，用于在因子和模型之间进行智能选择，
以优化量化交易策略的性能指标。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np


@dataclass
class Metrics:
    """
    量化交易策略评估指标数据类
    
    这些指标用于评估因子或模型的性能，作为多臂赌博机算法的反馈信号
    """
    ic: float = 0.0          # Information Coefficient: 信息系数，衡量因子预测能力的指标，值域[-1, 1]，越大越好
    icir: float = 0.0        # IC Information Ratio: IC的稳定性指标，类似夏普比率，越大越好
    rank_ic: float = 0.0     # Rank Information Coefficient: 基于排序的IC，对异常值更鲁棒
    rank_icir: float = 0.0   # Rank IC Information Ratio: Rank IC的稳定性指标
    arr: float = 0.0         # Annualized Return: 年化超额收益率(考虑交易成本)
    ir: float = 0.0          # Information Ratio: 信息比率(考虑交易成本): 超额收益/跟踪误差
    mdd: float = 0.0         # Max Drawdown: 最大回撤(考虑交易成本): 从最高点到最低点的最大跌幅
    sharpe: float = 0.0      # Sharpe Ratio: 夏普比率近似: 收益与最大回撤的比值(简化计算)

    def as_vector(self) -> np.ndarray:
        """
        将指标转换为向量形式，用于算法计算
        
        返回:
            np.ndarray: 包含所有指标的numpy数组
            注意：-self.mdd是因为最大回撤越小越好，所以取负值使其符合越大越好的原则
        """
        return np.array(
            [
                self.ic,
                self.icir,
                self.rank_ic,
                self.rank_icir,
                self.arr,
                self.ir,
                -self.mdd,  # 最大回撤越小越好，所以取负值
                self.sharpe,
            ]
        )


def extract_metrics_from_experiment(experiment) -> Metrics:
    """
    从实验反馈中提取评估指标
    
    参数:
        experiment: 实验对象，包含运行结果
        
    返回:
        Metrics: 提取的评估指标对象
    """
    try:
        result = experiment.result
        # 提取各种评估指标
        ic = result.get("IC", 0.0)  # Information Coefficient: 衡量因子预测能力的指标，值域[-1, 1]，越大越好
        icir = result.get("ICIR", 0.0)  # IC Mean / IC Std: IC的稳定性指标，类似夏普比率，越大越好
        rank_ic = result.get("Rank IC", 0.0)  # Rank Information Coefficient: 基于排序的IC，对异常值更鲁棒
        rank_icir = result.get("Rank ICIR", 0.0)  # Rank IC Information Ratio: Rank IC的稳定性指标
        # 注意：键名后面有空格
        arr = result.get("1day.excess_return_with_cost.annualized_return ", 0.0)  # 年化超额收益率(考虑交易成本)
        ir = result.get("1day.excess_return_with_cost.information_ratio", 0.0)  # 信息比率(考虑交易成本): 超额收益/跟踪误差
        mdd = result.get("1day.excess_return_with_cost.max_drawdown", 1.0)  # 最大回撤(考虑交易成本): 从最高点到最低点的最大跌幅
        # 计算夏普比率近似值：收益与最大回撤的比值(简化计算)
        sharpe = arr / -mdd if mdd != 0 else 0.0  # 夏普比率近似: 收益与最大回撤的比值(简化计算)

        return Metrics(ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir, arr=arr, ir=ir, mdd=mdd, sharpe=sharpe)
    except Exception as e:
        print(f"提取指标时出错: {e}")
        return Metrics()


class LinearThompsonTwoArm:
    """
    线性汤普森采样双臂赌博机实现
    
    该类实现了线性汤普森采样算法，用于在两个选择（因子和模型）之间进行智能决策。
    算法通过维护每个臂的后验分布来学习最优选择。
    """
    
    def __init__(self, dim: int, prior_var: float = 1.0, noise_var: float = 1.0):
        """
        初始化线性汤普森采样双臂赌博机
        
        参数:
            dim (int): 特征维度，即指标向量的长度
            prior_var (float): 先验方差，用于初始化先验分布
            noise_var (float): 噪声方差，用于更新时的噪声建模
        """
        self.dim = dim
        self.noise_var = noise_var
        # 每个臂都有自己的后验分布：均值和协方差矩阵的逆（精度矩阵）
        # 初始化均值为零向量
        self.mean = {
            "factor": np.zeros(dim),
            "model": np.zeros(dim),
        }
        # 初始化精度矩阵为单位矩阵除以先验方差
        self.precision = {
            "factor": np.eye(dim) / prior_var,
            "model": np.eye(dim) / prior_var,
        }

    def sample_reward(self, arm: str, x: np.ndarray) -> float:
        """
        从指定臂的后验分布中采样奖励值
        
        参数:
            arm (str): 臂的名称（"factor"或"model"）
            x (np.ndarray): 特征向量（指标向量）
            
        返回:
            float: 采样的奖励值
        """
        P = self.precision[arm]
        # 确保精度矩阵是对称的
        P = 0.5 * (P + P.T)

        eps = 1e-6
        try:
            # 计算协方差矩阵（精度矩阵的逆）
            cov = np.linalg.inv(P + eps * np.eye(self.dim))
            # 使用Cholesky分解生成采样
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.dim)
            # 从后验分布中采样权重
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            # 如果矩阵求逆失败，直接使用均值
            w_sample = self.mean[arm]

        # 计算预测奖励值
        return float(np.dot(w_sample, x))

    def update(self, arm: str, x: np.ndarray, r: float) -> None:
        """
        根据观察到的奖励更新指定臂的后验分布
        
        参数:
            arm (str): 臂的名称（"factor"或"model"）
            x (np.ndarray): 特征向量（指标向量）
            r (float): 观察到的奖励值
        """
        P = self.precision[arm]
        # 更新精度矩阵：P = P + x*x^T/σ²
        P += np.outer(x, x) / self.noise_var
        self.precision[arm] = P
        # 更新均值：μ = P^(-1) * (P*μ + r*x/σ²)
        self.mean[arm] = np.linalg.solve(P, P @ self.mean[arm] + (r / self.noise_var) * x)

    def next_arm(self, x: np.ndarray) -> str:
        """
        根据当前特征向量选择下一个要拉动的臂
        
        参数:
            x (np.ndarray): 特征向量（指标向量）
            
        返回:
            str: 选择的臂名称（"factor"或"model"）
        """
        # 为每个臂采样奖励值
        scores = {arm: self.sample_reward(arm, x) for arm in ("factor", "model")}
        # 选择奖励值最大的臂
        return max(scores, key=scores.get)


class EnvController:
    """
    环境控制器，用于管理多臂赌博机算法在量化金融场景中的应用
    
    该类封装了线性汤普森采样算法，并提供了奖励计算和决策接口。
    """
    
    def __init__(self, weights: Tuple[float, ...] = None) -> None:
        """
        初始化环境控制器
        
        参数:
            weights (Tuple[float, ...]): 各指标的权重，用于计算综合奖励值
                默认权重为(0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2)
                对应于[ic, icir, rank_ic, rank_icir, arr, ir, -mdd, sharpe]
        """
        # 指标权重向量，用于计算综合奖励值
        # 权重反映了不同指标对策略性能的重要性
        self.weights = np.asarray(weights or (0.1, 0.1, 0.05, 0.05, 0.25, 0.15, 0.1, 0.2))
        # 初始化线性汤普森采样双臂赌博机
        self.bandit = LinearThompsonTwoArm(dim=8, prior_var=10.0, noise_var=0.5)

    def reward(self, m: Metrics) -> float:
        """
        根据评估指标计算奖励值
        
        参数:
            m (Metrics): 评估指标对象
            
        返回:
            float: 综合奖励值
        """
        # 将指标转换为向量并计算加权和
        return float(np.dot(self.weights, m.as_vector()))

    def decide(self, m: Metrics) -> str:
        """
        根据评估指标决定下一步选择因子还是模型
        
        参数:
            m (Metrics): 评估指标对象
            
        返回:
            str: 决策结果（"factor"或"model"）
        """
        x = m.as_vector()
        return self.bandit.next_arm(x)

    def record(self, m: Metrics, arm: str) -> None:
        """
        记录选择结果并更新赌博机模型
        
        参数:
            m (Metrics): 评估指标对象
            arm (str): 实际选择的臂（"factor"或"model"）
        """
        # 计算奖励值
        r = self.reward(m)
        # 更新赌博机模型
        self.bandit.update(arm, m.as_vector(), r)