# TODO: 如果 traces 将被传递到实例中，则移除 `self.scen`。

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Feedback
from rdagent.core.experiment import (
    ASpecificExp,
    ASpecificPlan,
    Experiment,
    ExperimentPlan,
)
from rdagent.core.knowledge_base import KnowledgeBase
from rdagent.core.scenario import Scenario

if TYPE_CHECKING:
    from rdagent.utils.workflow.loop import LoopBase


class Hypothesis:
    """
    假设类，表示一个可测试的假设或想法
    
    TODO: 我们可能有更好的名称。
    
    名称候选:
    - Belief（信念）
    """

    def __init__(
        self,
        hypothesis: str,
        reason: str,
        concise_reason: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
    ) -> None:
        """
        初始化假设对象
        
        Args:
            hypothesis: 假设的描述
            reason: 提出假设的原因
            concise_reason: 简洁的原因说明
            concise_observation: 简洁的观察结果
            concise_justification: 简洁的论证
            concise_knowledge: 简洁的知识总结
        """
        self.hypothesis: str = hypothesis
        self.reason: str = reason
        self.concise_reason: str = concise_reason
        self.concise_observation: str = concise_observation
        self.concise_justification: str = concise_justification
        self.concise_knowledge: str = concise_knowledge

    def __str__(self) -> str:
        """返回假设的字符串表示"""
        return f"""Hypothesis: {self.hypothesis}
Reason: {self.reason}"""

    # 来源: data_ana | model_nan = None


# 原始数据(仓库/数据/反馈路径) => 视图/总结 => 生成的假设


class ExperimentFeedback(Feedback):
    """
    实验反馈类，表示对实验结果的反馈
    """
    
    def __init__(
        self,
        reason: str,
        *,
        code_change_summary: str | None = None,
        decision: bool,
        eda_improvement: str | None = None,
        exception: Exception | None = None,
    ) -> None:
        """
        初始化实验反馈对象
        
        Args:
            reason: 反馈的原因
            code_change_summary: 代码变更总结
            decision: 决策结果（是否接受实验）
            eda_improvement: EDA（探索性数据分析）改进建议
            exception: 异常信息（如果实验失败）
        """
        self.decision = decision
        self.eda_improvement = eda_improvement
        self.reason = reason
        # Exception 不为 None 表示由于异常而无法生成可运行的实验
        # 可运行的结果并不总是好的
        self.exception: Exception | None = (
            exception  # 如果实验引发异常，它将被集成到反馈的一部分中
        )
        self.code_change_summary = code_change_summary

    def __bool__(self) -> bool:
        """返回决策结果"""
        return self.decision

    def __str__(self) -> str:
        """返回反馈的字符串表示"""
        res = f"Decision: {self.decision}\nReason: {self.reason}"
        code_change_summary = getattr(self, "code_change_summary", None)
        if code_change_summary is not None:
            res += "\nCode Change Summary: " + code_change_summary
        return res

    @classmethod
    def from_exception(cls, e: Exception) -> ExperimentFeedback:
        """
        从异常创建反馈的便捷方法
        
        Args:
            e: 异常对象
            
        Returns:
            ExperimentFeedback: 基于异常的反馈对象
        """
        return cls(decision=False, reason=f"The experiment fails due to {e!s}", exception=e)


class HypothesisFeedback(ExperimentFeedback):
    """
    假设反馈类，继承自实验反馈，包含更多关于假设的反馈信息
    """
    
    def __init__(
        self,
        observations: str,
        hypothesis_evaluation: str,
        new_hypothesis: str,
        reason: str,
        *,
        code_change_summary: str | None = None,
        decision: bool,
        eda_improvement: str | None = None,
        acceptable: bool | None = None,
    ) -> None:
        """
        初始化假设反馈对象
        
        Args:
            observations: 观察结果
            hypothesis_evaluation: 对假设的评估
            new_hypothesis: 新的假设
            reason: 反馈原因
            code_change_summary: 代码变更总结
            decision: 决策结果
            eda_improvement: EDA改进建议
            acceptable: 是否可接受
        """
        super().__init__(
            reason,
            decision=decision,
            code_change_summary=code_change_summary,
            eda_improvement=eda_improvement,
        )
        self.observations = observations
        self.hypothesis_evaluation = hypothesis_evaluation
        self.new_hypothesis = new_hypothesis
        self.acceptable = acceptable

    def __str__(self) -> str:
        """返回假设反馈的字符串表示"""
        return f"""{super().__str__()}
Observations: {self.observations}
Hypothesis Evaluation: {self.hypothesis_evaluation}
New Hypothesis: {self.new_hypothesis}"""


ASpecificScen = TypeVar("ASpecificScen", bound=Scenario)
ASpecificKB = TypeVar("ASpecificKB", bound=KnowledgeBase)


class Trace(Generic[ASpecificScen, ASpecificKB]):
    """
    跟踪类，用于跟踪实验历史和决策过程
    
    这个类维护了一个实验和反馈的历史记录，以及它们之间的关系图结构。
    """
    
    # 定义节点类型为包含实验和反馈的元组
    NodeType = tuple[Experiment, ExperimentFeedback]
    # 新根节点的表示
    NEW_ROOT: tuple = ()

    def __init__(self, scen: ASpecificScen, knowledge_base: ASpecificKB | None = None) -> None:
        """
        初始化跟踪对象
        
        Args:
            scen: 场景对象
            knowledge_base: 知识库对象
        """
        self.scen: ASpecificScen = scen

        # BEGIN: 图结构 -------------------------
        # 包含实验和反馈的元组列表，按时间组织
        self.hist: list[Trace.NodeType] = []
        # 表示DAG结构中父节点索引的元组列表
        self.dag_parent: list[tuple[int, ...]] = []
        # 定义:
        # - (,) 表示没有父节点（一棵树中的根节点）;
        # - (1,) 表示一个父节点;
        # - (1, 2) 表示两个父节点（尚未实现多父节点）。
        # 父关系的语法糖:
        # - 仅用于选择:
        #    - (-1,) 表示选择最后一个记录节点作为父节点。

        # 注意: hist 和 dag_parent 的顺序是按记录实验的顺序组织的。
        # 所以它可能与 loop_id 的顺序不同。
        # 所以我们需要一个额外的映射将排队ID映射回循环ID。
        self.idx2loop_id: dict[int, int] = {}

        # 设计讨论:
        # - 如果我们统一 loop_id 和排队ID，我们将减少认知负担。
        # - 如果我们对循环和排队使用不同的ID，我们就不必处理占位符逻辑。
        # END: 图结构 -------------------------

        # TODO: self.hist 现在是2元组，从其中移除假设，稍后更改旧代码。
        self.knowledge_base: ASpecificKB | None = knowledge_base
        # 当前选择，默认为最新记录
        self.current_selection: tuple[int, ...] = (-1,)

    def get_sota_hypothesis_and_experiment(self) -> tuple[Hypothesis | None, Experiment | None]:
        """
        获取最新的成功假设和实验
        
        Returns:
            tuple: 包含假设和实验的元组，如果没有找到则返回(None, None)
        """
        # TODO: 返回值与签名不一致。
        for experiment, feedback in self.hist[::-1]:
            if feedback.decision:
                return experiment.hypothesis, experiment

        return None, None

    def is_selection_new_tree(self, selection: tuple[int, ...] | None = None) -> bool:
        """
        检查当前跟踪是否为新树
        
        Args:
            selection: 选择的节点索引元组
            
        Returns:
            bool: 如果是新树则返回True，否则返回False
        """
        if selection is None:
            selection = self.get_current_selection()

        return selection == self.NEW_ROOT or len(self.dag_parent) == 0

    def get_current_selection(self) -> tuple[int, ...]:
        """
        获取当前选择
        
        Returns:
            tuple: 当前选择的节点索引元组
        """
        return self.current_selection

    def set_current_selection(self, selection: tuple[int, ...]) -> None:
        """
        设置当前选择
        
        Args:
            selection: 要设置的选择节点索引元组
        """
        self.current_selection = selection

    def get_parent_exps(
        self,
        selection: tuple[int, ...] | None = None,
    ) -> list[Trace.NodeType]:
        """
        收集给定选择的所有祖先节点
        
        Args:
            selection: 选择的节点索引元组
            
        Returns:
            list: 按[根->...->父->当前节点]顺序排列的祖先节点列表
        """
        if selection is None:
            selection = self.get_current_selection()

        if self.is_selection_new_tree(selection):
            return []

        return [self.hist[i] for i in self.get_parents(selection[0])]

    def exp2idx(self, exp: Experiment | list[Experiment]) -> int | list[int] | None:
        """
        将实验转换为索引
        
        Args:
            exp: 实验对象或实验对象列表
            
        Returns:
            int | list[int] | None: 实验对应的索引或索引列表
        """
        if isinstance(exp, list):
            exps: list[Experiment] = exp

            # 保持顺序
            exp_to_index: dict[Experiment, int] = {_exp: i for i, (_exp, _) in enumerate(self.hist)}
            return [exp_to_index[_exp] for _exp in exps]
        for i, (_exp, _) in enumerate(self.hist):
            if _exp == exp:
                return i
        return None

    def idx2exp(self, idx: int | list[int]) -> Experiment | list[Experiment]:
        """
        将索引转换为实验
        
        Args:
            idx: 索引或索引列表
            
        Returns:
            Experiment | list[Experiment]: 索引对应的实验或实验列表
        """
        if isinstance(idx, list):
            idxs: list[int] = idx
            return [self.hist[_idx][0] for _idx in idxs]
        return self.hist[idx][0]

    def is_parent(self, parent_idx: int, child_idx: int) -> bool:
        """
        检查一个节点是否为另一个节点的父节点
        
        Args:
            parent_idx: 父节点索引
            child_idx: 子节点索引
            
        Returns:
            bool: 如果parent_idx是child_idx的父节点则返回True，否则返回False
        """
        ancestors = self.get_parents(child_idx)
        return parent_idx in ancestors

    def get_parents(self, child_idx: int) -> list[int]:
        """
        获取一个节点的所有祖先节点
        
        Args:
            child_idx: 子节点索引
            
        Returns:
            list: 祖先节点索引列表
        """
        if self.is_selection_new_tree((child_idx,)):
            return []

        ancestors: list[int] = []
        curr = child_idx
        while True:
            ancestors.insert(0, curr)
            parent_tuple = self.dag_parent[curr]
            if not parent_tuple or parent_tuple[0] == curr:
                break
            curr = parent_tuple[0]

        return ancestors


class CheckpointSelector:
    """
    检查点选择器类
    
    在跟踪中，我们可以从任何检查点开始（我们将用变量`from_checkpoint_idx`表示）
    """

    @abstractmethod
    def get_selection(self, trace: Trace) -> tuple[int, ...] | None:
        """
        获取选择的检查点
        
        checkpoint_idx 表示我们想要创建新节点的位置。
        返回值应该是目标节点的索引（新生成节点的父节点）。
        - `(-1, )` 表示从跟踪中的最新试验开始 - 默认值

          - 注意: 我们不鼓励使用此选项; 当我们有多个跟踪时会令人困惑。

        - `(idx, )` 表示从跟踪中的第`idx`个试验开始。
        - `None` 表示从头开始（开始新的跟踪）


        - 更高级的选择策略在 `select.py` 中
        """


class SOTAexpSelector:
    """
    SOTA实验选择器类，用于从跟踪中选择SOTA实验进行提交
    """

    @abstractmethod
    def get_sota_exp_to_submit(self, trace: Trace) -> Experiment | None:
        """
        从跟踪中选择SOTA实验进行提交
        
        Args:
            trace: 跟踪对象
            
        Returns:
            Experiment | None: 要提交的SOTA实验，如果没有则返回None
        """


class ExpPlanner(ABC, Generic[ASpecificPlan]):
    """
    实验规划器抽象类
    
    规划器应基于跟踪生成实验计划。
    """

    def __init__(self, scen: Scenario) -> None:
        """
        初始化实验规划器
        
        Args:
            scen: 场景对象
        """
        self.scen = scen

    @abstractmethod
    def plan(self, trace: Trace) -> ASpecificPlan:
        """
        基于跟踪生成实验计划
        
        Args:
            trace: 跟踪对象
            
        Returns:
            ASpecificPlan: 实验计划
        """


class ExpGen(ABC):
    """
    实验生成器抽象类
    """

    def __init__(self, scen: Scenario) -> None:
        """
        初始化实验生成器
        
        Args:
            scen: 场景对象
        """
        self.scen = scen

    @abstractmethod
    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Experiment:
        """
        基于跟踪生成实验
        
        规划是生成的一部分，但由于我们可能支持多阶段规划，
        我们需要将计划作为可选参数传递。

        `ExpGen().gen()` 的作用类似于

        .. code-block:: python

            # ExpGen().gen() ==
            Hypothesis2Experiment().convert(
                HypothesisGen().gen(trace)
            )
        """

    async def async_gen(self, trace: Trace, loop: LoopBase) -> Experiment:
        """
        异步生成实验，并决定是否停止生成并将控制权交给其他例程。
        
        Args:
            trace: 跟踪对象
            loop: 循环对象
            
        Returns:
            Experiment: 生成的实验
        """
        # 我们在这里给出默认实现。
        # 提案设置为尽力在最大并行级别生成实验。
        while True:
            if loop.get_unfinished_loop_cnt(loop.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                return self.gen(trace)
            await asyncio.sleep(1)


class HypothesisGen(ABC):
    """
    假设生成器抽象类
    """

    def __init__(self, scen: Scenario) -> None:
        """
        初始化假设生成器
        
        Args:
            scen: 场景对象
        """
        self.scen = scen

    @abstractmethod
    def gen(
        self,
        trace: Trace,
        plan: ExperimentPlan | None = None,
    ) -> Hypothesis:
        """
        基于跟踪生成假设
        
        变量 `scenario_desc` 的动机:
            - 模拟数据科学家正在观察场景。

        scenario_desc 可能包括:
            - 数据观察:
                - 原始或衍生
            - 任务信息:
        """


class Hypothesis2Experiment(ABC, Generic[ASpecificExp]):
    """
    假设到实验的转换器抽象类
    
    [抽象描述 => 具体描述] => 代码实现卡片
    """

    @abstractmethod
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> ASpecificExp:
        """
        将假设连接到实现
        
        Args:
            hypothesis: 假设对象
            trace: 跟踪对象
            
        Returns:
            ASpecificExp: 转换后的具体实验
        """
        ...


# 布尔值, 原因, 置信度等


class Experiment2Feedback(ABC):
    """
    实验到反馈的转换器抽象类
    
    从**已执行**的不同任务的实现中生成关于假设的反馈
    & 与之前性能的比较
    """

    def __init__(self, scen: Scenario) -> None:
        """
        初始化实验到反馈的转换器
        
        Args:
            scen: 场景对象
        """
        self.scen = scen

    @abstractmethod
    def generate_feedback(self, exp: Experiment, trace: Trace) -> ExperimentFeedback:
        """
        生成对假设的反馈
        
        `exp` 应该被执行，并且结果应该包括在内，以及与之前结果的比较（由LLM完成）。
        例如: Qlib的`mlflow`将被包括在内。
        
        Args:
            exp: 实验对象
            trace: 跟踪对象
            
        Returns:
            ExperimentFeedback: 生成的反馈
        """
        error_message = "generate_feedback method is not implemented."
        raise NotImplementedError(error_message)