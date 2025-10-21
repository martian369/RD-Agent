from __future__ import annotations

from abc import abstractmethod

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evolving_framework import EvolvingStrategy, EvoStep, QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.core.utils import multiprocessing_wrapper


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    # 可选键，用于存储演化主体变化的摘要信息
    KEY_CHANGE_SUMMARY = "__change_summary__"  

    def __init__(self, scen: Scenario, settings: CoSTEERSettings):
        """
        初始化多进程演化策略对象。
        
        参数：
        - scen (Scenario): 场景对象，包含演化所需的上下文信息
        - settings (CoSTEERSettings): CoSTEER配置设置对象
        """
        super().__init__(scen)
        self.settings = settings

    @abstractmethod
    def implement_one_task(
        self,
        target_task: Task,  # 目标任务对象
        queried_knowledge: QueriedKnowledge | None = None,  # 查询到的知识，可选参数
        workspace: FBWorkspace | None = None,  # 当前工作区对象，可选参数
        prev_task_feedback: CoSTEERSingleFeedback | None = None,  # 上一次任务反馈，可选参数
    ) -> dict[str, str]:  # FIXME: fix interface of previous implement
        """
        实现单个任务的方法，输入任务和当前工作区，输出需要应用到工作区的修改。
        (即用新内容替换文件<filename>的内容)
        
        主要功能：
        1. 根据目标任务和相关知识生成代码实现
        2. 返回需要更新的文件字典
        
        参数：
        ----------
        target_task : Task
            需要实现的目标任务对象
            
        queried_knowledge : QueriedKnowledge | None
            查询到的相关知识，用于指导实现过程
            
        workspace : FBWorkspace | None
            当前工作区对象，包含现有的代码和资源
            
        prev_task_feedback : CoSTEERSingleFeedback | None
            上一次演化步骤的任务反馈信息
            None表示这是第一次循环
            
        返回值：
        ------
        dict[str, str]: 需要更新的工作区文件字典 {<文件名>: <文件内容>}
        - 特殊键: self.KEY_CHANGE_SUMMARY，用于存储变化摘要
        
        异常：
        ------
        NotImplementedError: 抽象方法，需要子类实现
        """
        raise NotImplementedError

    @abstractmethod
    def assign_code_list_to_evo(self, code_list: list[dict], evo: EvolvingItem) -> None:
        """
        将代码列表分配给演化项。
        
        主要功能：
        1. 由于implement_one_task以`workspace`作为输入并输出`modification`
        2. 我们需要将实现应用到演化项中
        
        参数：
        ----------
        code_list : list[dict]
            代码列表，与演化项的子任务对齐
            如果某个任务未实现，则在列表中放置None
            
        evo : EvolvingItem
            演化项对象，需要被分配代码实现
            
        返回值：
        ------
        None: 该方法直接修改传入的演化项对象
            
        异常：
        ------
        NotImplementedError: 抽象方法，需要子类实现
        """
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: EvolvingItem,  # 需要演化的项
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,  # 查询到的知识
        evolving_trace: list[EvoStep] = [],  # 演化轨迹列表
        **kwargs,  # 其他关键字参数
    ) -> EvolvingItem:
        """
        执行演化过程，对演化项中的子任务进行实现。
        
        主要功能：
        1. 识别需要演化的任务
        2. 利用已有的成功知识直接应用
        3. 对新任务使用多进程并行实现
        4. 将实现结果分配给演化项
        
        参数：
        ----------
        evo : EvolvingItem
            需要演化的项，包含子任务和工作区信息
            
        queried_knowledge : CoSTEERQueriedKnowledge | None
            查询到的CoSTEER知识，包含成功和失败任务的信息
            
        evolving_trace : list[EvoStep]
            演化轨迹列表，记录之前的演化步骤和反馈
            
        **kwargs : 
            其他关键字参数
            
        返回值：
        ------
        EvolvingItem: 演化后的项，包含更新的实现代码
            
        关键逻辑步骤：
        ------
        1. 初始化代码列表，所有任务初始为None
        2. 遍历所有子任务，识别需要演化的任务：
           - 如果任务在成功知识字典中，直接使用已有实现
           - 如果任务既不在成功知识也不在失败知识中，标记为待完成
        3. 获取上一次演化的反馈信息
        4. 使用多进程并行实现待完成的任务
        5. 将实现结果分配给演化项并返回
        """
        # 初始化代码列表，所有任务初始为None
        code_list = [None for _ in range(len(evo.sub_tasks))]

        # 1.找出需要evolve的task
        to_be_finished_task_index: list[int] = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            # 如果任务在成功知识字典中，直接使用已有实现
            if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                # NOTE: very weird logic:
                # it depends on the knowledge to set the already finished task

                code_list[index] = list(queried_knowledge.success_task_to_knowledge_dict[
                    target_task_desc
                ].implementation.file_dict.values())[0]
            # 如果任务既不在成功知识也不在失败知识中，标记为待完成
            elif (
                target_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        # 获取上一次演化的反馈信息
        last_feedback = None
        if len(evolving_trace) > 0:
            last_feedback = evolving_trace[-1].feedback
            # 确保反馈类型正确
            assert isinstance(last_feedback, CoSTEERMultiFeedback)

        # 使用多进程并行实现待完成的任务
        result = multiprocessing_wrapper(
            [
                (
                    self.implement_one_task,
                    (
                        evo.sub_tasks[target_index],
                        queried_knowledge,
                        evo.experiment_workspace,
                        None if last_feedback is None else last_feedback[target_index],
                    ),
                )
                for target_index in to_be_finished_task_index
            ],
            n=min(RD_AGENT_SETTINGS.multi_proc_n, len(to_be_finished_task_index))
            ,
        )
        
        # 将实现结果填入代码列表对应位置
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        # 将代码列表分配给演化项
        evo = self.assign_code_list_to_evo(code_list, evo)

        return evo