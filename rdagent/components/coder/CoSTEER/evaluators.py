from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Evaluator, Feedback
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger

if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario

# TODO:
# 1. It seems logically sound, but we currently lack a scenario to apply it.
# 2. If it proves to be useful, relocate it to a more general location.
#
# class FBWorkspaceExeFeedback(Feedback):
#     """
#     It pairs with FBWorkspace in the abstract level.
#     """
#     # ws: FBWorkspace   # potential
#     stdout: str


@dataclass
class CoSTEERSingleFeedback(Feedback):
    # TODO: (xiao)
    # it should be more general class for FBWorkspaceExeFeedback
    # A better name of it may be NormalFeedback
    # TODO: It should be a general feeddback for CoSTEERR
    """
    数据加载器评估的反馈信息。
    设计与实现代码的阶段对齐
    - 执行 -> 返回值 -> 代码 -> 最终决策
    """
    execution: str  # 执行反馈：代码执行过程中的输出信息
    # execution_feedback
    return_checking: str | None  # 返回值检查：包括测试中的所有检查（关于生成值的约束条件）
    # value_feedback, shape_feedback, value_generated_flag
    code: str  # 代码反馈：对代码实现质量的评估
    final_decision: bool | None = None  # 最终决策：该实现是否通过评估

    @staticmethod
    def val_and_update_init_dict(data: dict) -> dict:
        # TODO: (bowen) use a more general method to validate and update the data dictionary before init, like pydantic
        """
        验证并转换给定数据字典中的'final_decision'字段。
        
        主要功能：
        1. 验证数据字典中是否包含必需的'final_decision'字段
        2. 将字符串形式的布尔值转换为真正的布尔值
        3. 验证其他关键字段的数据类型
        
        参数：
        - data (dict): 包含'final_decision'字段的数据字典
        
        返回值：
        - dict: 更新后的数据字典，其中'final_decision'被转换为布尔值
        
        异常：
        - ValueError: 当'final_decision'不存在或不是布尔类型时抛出
        """
        # 检查必需字段是否存在
        if "final_decision" not in data:
            raise ValueError("'final_decision' is required")

        # 将字符串形式的布尔值转换为真正的布尔值
        if isinstance(data["final_decision"], str):
            if data["final_decision"] == "false" or data["final_decision"] == "False":
                data["final_decision"] = False
            elif data["final_decision"] == "true" or data["final_decision"] == "True":
                data["final_decision"] = True

        # 验证final_decision是否为布尔类型
        if not isinstance(data["final_decision"], bool):
            raise ValueError(f"'final_decision' must be a boolean, not {type(data['final_decision'])}")

        # 验证其他关键字段的数据类型
        for attr in "execution", "return_checking", "code":
            if data[attr] is not None and not isinstance(data[attr], str):
                raise ValueError(f"'{attr}' must be a string, not {type(data[attr])}")
        return data

    @classmethod
    def merge(cls, feedback_li: list["CoSTEERSingleFeedback"]) -> "CoSTEERSingleFeedback":
        # NOTE:
        # Here we don't know the detailed design of each feedback, we just know they are CoSTEERSingleFeedback
        # So we merge them only based on CoSTEERSingleFeedback's attributes
        # **So some information may be lost when we have different types of feedbacks**
        # If you have more sophisticated sub class of CoSTEERSingleFeedback, you should override this method
        # to avoid the loss of information.
        """
        合并多个CoSTEERSingleFeedback对象为一个反馈对象。
        
        主要功能：
        1. 基于第一个反馈对象创建新的反馈对象
        2. 合并所有反馈对象的final_decision（使用AND逻辑）
        3. 合并所有反馈对象的execution、return_checking和code字段
        
        参数：
        - feedback_li (list[CoSTEERSingleFeedback]): 需要合并的反馈对象列表
        
        返回值：
        - CoSTEERSingleFeedback: 合并后的反馈对象
        
        关键逻辑：
        - final_decision使用all()函数进行合并，只有所有反馈都为True时才返回True
        - execution、return_checking和code字段通过换行符连接
        - 如果某个字段为None，则在合并时被忽略
        """

        # 基于第一个反馈对象创建副本
        fb = deepcopy(feedback_li[0])

        # 合并final_decision：只有所有评估器都通过时才为True
        fb.final_decision = all(fb.final_decision for fb in feedback_li)
        
        # 合并execution、return_checking和code字段
        for attr in "execution", "return_checking", "code":
            setattr(
                fb,
                attr,
                "\n\n".join([getattr(_fb, attr) for _fb in feedback_li if getattr(_fb, attr) is not None]),
            )
        return fb

    def __str__(self) -> str:
        """
        将反馈对象转换为格式化的字符串表示。
        
        返回值：
        - str: 格式化的反馈信息字符串，包含执行信息、返回值检查、代码反馈和最终决策
        """
        return f"""------------------Execution------------------
{self.execution}
------------------Return Checking------------------
{self.return_checking if self.return_checking is not None else 'No return checking'}
------------------Code------------------
{self.code}
------------------Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""

    def __bool__(self):
        """
        使反馈对象可以作为布尔值使用，直接返回final_decision的值。
        
        返回值：
        - bool: final_decision的值，如果为True表示实现成功，False表示实现失败
        """
        return self.final_decision


class CoSTEERSingleFeedbackDeprecated(CoSTEERSingleFeedback):
    """这是所有代码生成器对单个实现反馈的基类"""

    def __init__(
        self,
        execution_feedback: str = None,
        shape_feedback: str = None,
        code_feedback: str = None,
        value_feedback: str = None,
        final_decision: bool = None,
        final_feedback: str = None,
        value_generated_flag: bool = None,
        final_decision_based_on_gt: bool = None,
    ) -> None:
        """
        初始化CoSTEERSingleFeedbackDeprecated对象。
        
        参数：
        - execution_feedback (str, optional): 执行反馈信息
        - shape_feedback (str, optional): 形状反馈信息（注意：此属性不够通用）
        - code_feedback (str, optional): 代码反馈信息
        - value_feedback (str, optional): 值反馈信息
        - final_decision (bool, optional): 最终决策结果
        - final_feedback (str, optional): 最终反馈信息
        - value_generated_flag (bool, optional): 值生成标志
        - final_decision_based_on_gt (bool, optional): 基于真实值的最终决策
        """
        self.execution_feedback = execution_feedback
        self.code_feedback = code_feedback
        self.value_feedback = value_feedback
        self.final_decision = final_decision
        self.final_feedback = final_feedback
        self.value_generated_flag = value_generated_flag
        self.final_decision_based_on_gt = final_decision_based_on_gt

        # TODO:
        # Not general enough. So we should not put them in the general costeer feedback
        # Instead, we should create subclass for it.
        self.shape_feedback = shape_feedback  # Not general enough. So

    @property
    def execution(self):
        """
        获取执行反馈信息。
        
        返回值：
        - str: execution_feedback的值
        """
        return self.execution_feedback

    @execution.setter
    def execution(self, value):
        """
        设置执行反馈信息。
        
        参数：
        - value (str): 新的执行反馈信息
        """
        self.execution_feedback = value

    @property
    def return_checking(self):
        """
        获取返回值检查信息。
        如果value_generated_flag为True，则返回值反馈和形状反馈的组合；否则返回None。
        
        返回值：
        - str | None: 返回值检查信息或None
        """
        if self.value_generated_flag:
            return f"value feedback: {self.value_feedback}\n\nshape feedback: {self.shape_feedback}"
        return None

    @return_checking.setter
    def return_checking(self, value):
        """
        设置返回值检查信息。
        由于return_checking是从value_feedback和shape_feedback派生的，所以这里不需要做任何事情。
        
        参数：
        - value: 设置的值（在此实现中被忽略）
        """
        # Since return_checking is derived from value_feedback and shape_feedback,
        # we don't need to do anything here
        self.value_feedback = value
        self.shape_feedback = value

    @property
    def code(self):
        """
        获取代码反馈信息。
        
        返回值：
        - str: code_feedback的值
        """
        return self.code_feedback

    @code.setter
    def code(self, value):
        """
        设置代码反馈信息。
        
        参数：
        - value (str): 新的代码反馈信息
        """
        self.code_feedback = value

    def __str__(self) -> str:
        """
        将反馈对象转换为格式化的字符串表示。
        
        返回值：
        - str: 格式化的反馈信息字符串，包含各种反馈信息和最终决策
        """
        return f"""------------------Execution Feedback------------------
{self.execution_feedback if self.execution_feedback is not None else 'No execution feedback'}
------------------Shape Feedback------------------
{self.shape_feedback if self.shape_feedback is not None else 'No shape feedback'}
------------------Code Feedback------------------
{self.code_feedback if self.code_feedback is not None else 'No code feedback'}
------------------Value Feedback------------------
{self.value_feedback if self.value_feedback is not None else 'No value feedback'}
------------------Final Feedback------------------
{self.final_feedback if self.final_feedback is not None else 'No final feedback'}
------------------Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""


class CoSTEERMultiFeedback(Feedback):
    """反馈信息容器，包含一个列表，每个元素对应一个因子实现的反馈信息。"""

    def __init__(self, feedback_list: List[CoSTEERSingleFeedback]) -> None:
        """
        初始化CoSTEERMultiFeedback对象。
        
        参数：
        - feedback_list (List[CoSTEERSingleFeedback]): 反馈信息列表
        """
        self.feedback_list = feedback_list

    def __getitem__(self, index: int) -> CoSTEERSingleFeedback:
        """
        通过索引获取反馈信息。
        
        参数：
        - index (int): 反馈信息在列表中的索引
        
        返回值：
        - CoSTEERSingleFeedback: 指定索引位置的反馈信息
        """
        return self.feedback_list[index]

    def __len__(self) -> int:
        """
        获取反馈信息列表的长度。
        
        返回值：
        - int: 反馈信息列表的长度
        """
        return len(self.feedback_list)

    def append(self, feedback: CoSTEERSingleFeedback) -> None:
        """
        向反馈信息列表中添加新的反馈信息。
        
        参数：
        - feedback (CoSTEERSingleFeedback): 要添加的反馈信息
        """
        self.feedback_list.append(feedback)

    def __iter__(self):
        """
        使CoSTEERMultiFeedback对象可迭代。
        
        返回值：
        - iterator: 反馈信息列表的迭代器
        """
        return iter(self.feedback_list)

    def is_acceptable(self) -> bool:
        """
        检查所有反馈是否都可接受。
        
        返回值：
        - bool: 如果所有反馈都可接受则返回True，否则返回False
        """
        return all(feedback.is_acceptable() for feedback in self.feedback_list)

    def finished(self) -> bool:
        """
        检查所有反馈是否都已完成（通过评估）。
        在某些实现中，任务可能多次失败，导致代理跳过实现，这会产生None反馈。
        我们希望接受正确的部分并忽略None反馈。
        
        返回值：
        - bool: 如果所有非None反馈都已完成则返回True，否则返回False
        """
        return all(feedback.final_decision for feedback in self.feedback_list if feedback is not None)

    def __bool__(self) -> bool:
        """
        使反馈对象可以作为布尔值使用，检查所有反馈是否都通过。
        
        返回值：
        - bool: 如果所有反馈都通过则返回True，否则返回False
        """
        return all(feedback.final_decision for feedback in self.feedback_list)


class CoSTEEREvaluator(Evaluator):
    def __init__(
        self,
        scen: "Scenario",  # 场景对象，包含评估所需的上下文信息
    ) -> None:
        """
        初始化CoSTEEREvaluator对象。
        
        参数：
        - scen (Scenario): 场景对象，包含评估所需的上下文信息
        """
        self.scen = scen

    # TODO:
    # I think we should have unified interface for all evaluates, for examples.
    # So we should adjust the interface of other factors
    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """
        抽象评估方法，需要子类实现具体的评估逻辑。
        
        参数：
        - target_task (Task): 需要评估的目标任务
        - implementation (Workspace): 实现的工作区对象
        - gt_implementation (Workspace): 真实实现的工作区对象（用于对比）
        - **kwargs: 其他关键字参数
        
        返回值：
        - CoSTEERSingleFeedback: 评估反馈信息
        
        异常：
        - NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError("Please implement the `evaluator` method")


class CoSTEERMultiEvaluator(CoSTEEREvaluator):
    """用于实验评估的多任务评估器。由于有多个任务，所以会返回一个评估反馈列表"""

    def __init__(self, single_evaluator: CoSTEEREvaluator | list[CoSTEEREvaluator], *args, **kwargs) -> None:
        """
        初始化CoSTEERMultiEvaluator对象。
        
        参数：
        - single_evaluator (CoSTEEREvaluator | list[CoSTEEREvaluator]): 单个评估器或评估器列表
        - *args: 传递给父类的其他位置参数
        - **kwargs: 传递给父类的其他关键字参数
        """
        super().__init__(*args, **kwargs)
        self.single_evaluator = single_evaluator

    def evaluate(
        self,
        evo: EvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERMultiFeedback:
        """
        对演化项中的所有子任务进行评估。
        
        主要功能：
        1. 对每个子任务使用单个或多个评估器进行评估
        2. 使用多进程并行处理提高评估效率
        3. 合并多个评估器的反馈结果
        4. 返回包含所有子任务评估结果的多反馈对象
        
        参数：
        - evo (EvolvingItem): 演化项对象，包含多个子任务及相关实现
        - queried_knowledge (QueriedKnowledge, optional): 查询到的知识，默认为None
        - **kwargs: 其他关键字参数
        
        返回值：
        - CoSTEERMultiFeedback: 包含所有子任务评估反馈的多反馈对象
        
        关键逻辑：
        1. 将单个评估器转换为评估器列表
        2. 使用多进程并行评估每个子任务
        3. 合并多个评估器对同一任务的评估结果
        4. 记录最终决策结果并更新任务状态
        """
        # 将单个评估器转换为评估器列表
        eval_l = self.single_evaluator if isinstance(self.single_evaluator, list) else [self.single_evaluator]

        # 1) 评估每个子任务
        task_li_feedback_li = []
        # task_li_feedback_li: List[List[CoSTEERSingleFeedback]]
        # 示例：
        # 如果有2个评估器和3个子任务，每个评估器的evaluate方法返回包含3个CoSTEERSingleFeedback的列表，
        # 那么task_li_feedback_li将是：
        # [
        #   [feedback_1_1, feedback_1_2, feedback_1_3],  # 第1个评估器对所有子任务的评估结果
        #   [feedback_2_1, feedback_2_2, feedback_2_3],  # 第2个评估器对所有子任务的评估结果
        # ]
        # 其中feedback_i_j是第i个评估器对第j个子任务的反馈。
        for ev in eval_l:
            multi_implementation_feedback = multiprocessing_wrapper(
                [
                    (
                        ev.evaluate,
                        (
                            evo.sub_tasks[index],
                            evo.sub_workspace_list[index],
                            evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None,
                            queried_knowledge,
                        ),
                    )
                    for index in range(len(evo.sub_tasks))
                ],
                n=min(RD_AGENT_SETTINGS.multi_proc_n, len(evo.sub_tasks)),
            )
            task_li_feedback_li.append(multi_implementation_feedback)

        # 2) 沿子任务合并反馈以聚合多个评估反馈
        merged_task_feedback = []
        # task_li_feedback_li[0]是第一个评估器对不同任务的反馈列表
        for task_id, fb in enumerate(task_li_feedback_li[0]):
            fb = fb.merge([fb_li[task_id] for fb_li in task_li_feedback_li])
            merged_task_feedback.append(fb)
        # merged_task_feedback: List[CoSTEERSingleFeedback]
        # 示例：
        # [
        #   CoSTEERSingleFeedback(final_decision=True, execution="...", return_checking="...", code="..."),
        #   CoSTEERSingleFeedback(final_decision=False, execution="...", return_checking="...", code="..."),
        #   ...
        # ]
        # 每个元素对应跨所有评估器的某个子任务的合并反馈。
        # merged_task_feedback[i]是第i个子任务的合并反馈

        # 记录最终决策结果
        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in merged_task_feedback
        ]
        logger.info(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        # TODO: 这是为了与factor_implementation兼容；
        for index in range(len(evo.sub_tasks)):
            if final_decision[index]:
                evo.sub_tasks[index].factor_implementation = True

        return CoSTEERMultiFeedback(merged_task_feedback)