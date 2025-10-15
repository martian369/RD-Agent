from __future__ import annotations

import copy
import json
import pickle
import random
import re
from itertools import combinations
from pathlib import Path
from typing import List, Union

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.knowledge_management.graph import (
    UndirectedGraph,
    UndirectedNode,
)
from rdagent.core.evolving_agent import Feedback
from rdagent.core.evolving_framework import (
    EvolvableSubjects,
    EvolvingKnowledgeBase,
    EvoStep,
    Knowledge,
    QueriedKnowledge,
    RAGStrategy,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import (
    APIBackend,
    calculate_embedding_distance_between_str_list,
)
from rdagent.utils.agent.tpl import T


class CoSTEERKnowledge(Knowledge):
    """
    CoSTEER知识类，用于存储任务、实现和反馈信息
    
    该类封装了特定任务的实现代码和对应的反馈信息，是知识库中的基本单元。
    """
    def __init__(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        feedback: Feedback,
    ) -> None:
        """
        初始化CoSTEER知识对象
        
        Parameters:
        -----------
        target_task : Task
            目标任务对象
        implementation : FBWorkspace
            任务的实现工作区
        feedback : Feedback
            任务实现的反馈信息
        """
        self.target_task = target_task
        self.implementation = implementation.copy()
        self.feedback = feedback

    def get_implementation_and_feedback_str(self) -> str:
        """
        获取实现代码和反馈信息的字符串表示
        
        Returns:
        --------
        str
            包含实现代码和反馈信息的格式化字符串
        """
        return f"""------------------implementation code:------------------
{self.implementation.all_codes}
------------------implementation feedback:------------------
{self.feedback!s}
"""


class CoSTEERRAGStrategy(RAGStrategy):
    """
    CoSTEER检索增强生成策略基类
    
    该类提供了知识库的加载、保存和管理功能，是RAG策略的基础实现。
    """
    def __init__(self, *args, dump_knowledge_base_path: Path = None, **kwargs):
        """
        初始化RAG策略
        
        Parameters:
        -----------
        dump_knowledge_base_path : Path, optional
            知识库保存路径
        """
        super().__init__(*args, **kwargs)
        self.dump_knowledge_base_path = dump_knowledge_base_path

    def load_or_init_knowledge_base(
        self, former_knowledge_base_path: Path = None, component_init_list: list = [], evolving_version: int = 2
    ) -> EvolvingKnowledgeBase:
        """
        加载或初始化知识库
        
        Parameters:
        -----------
        former_knowledge_base_path : Path, optional
            原有知识库路径
        component_init_list : list
            组件初始化列表
        evolving_version : int
            演化版本号
            
        Returns:
        --------
        EvolvingKnowledgeBase
            加载或初始化的知识库对象
        """
        if former_knowledge_base_path is not None and former_knowledge_base_path.exists():
            knowledge_base = pickle.load(open(former_knowledge_base_path, "rb"))
            if evolving_version == 1 and not isinstance(knowledge_base, CoSTEERKnowledgeBaseV1):
                raise ValueError("The former knowledge base is not compatible with the current version")
            elif evolving_version == 2 and not isinstance(
                knowledge_base,
                CoSTEERKnowledgeBaseV2,
            ):
                raise ValueError("The former knowledge base is not compatible with the current version")
        else:
            knowledge_base = (
                CoSTEERKnowledgeBaseV2(
                    init_component_list=component_init_list,
                )
                if evolving_version == 2
                else CoSTEERKnowledgeBaseV1()
            )
        return knowledge_base

    def dump_knowledge_base(self):
        """
        保存知识库到文件
        """
        if self.dump_knowledge_base_path is None:
            logger.warning("Dump knowledge base path is not set, skip dumping.")
        else:
            if not self.dump_knowledge_base_path.parent.exists():
                self.dump_knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dump_knowledge_base_path, "wb") as f:
                pickle.dump(self.knowledgebase, f)

    def load_dumped_knowledge_base(self, *args, **kwargs):
        """
        加载已保存的知识库
        """
        if self.dump_knowledge_base_path is None:
            logger.warning("Dump knowledge base path is not set, skip dumping.")
        elif not Path(self.dump_knowledge_base_path).exists():
            logger.info(f"Dumped knowledge base {self.dump_knowledge_base_path} does not exist, skip loading.")
        else:
            with open(self.dump_knowledge_base_path, "rb") as f:
                self.knowledgebase = pickle.load(f)
            logger.info(f"Loaded dumped knowledge base from {self.dump_knowledge_base_path}")


class CoSTEERQueriedKnowledge(QueriedKnowledge):
    """
    CoSTEER查询知识类
    
    用于存储查询到的知识信息，包括成功任务到知识的映射和失败任务信息集合。
    """
    def __init__(self, success_task_to_knowledge_dict: dict = {}, failed_task_info_set: set = set()) -> None:
        """
        初始化查询知识对象
        
        Parameters:
        -----------
        success_task_to_knowledge_dict : dict
            成功任务到知识的映射字典
        failed_task_info_set : set
            失败任务信息集合
        """
        self.success_task_to_knowledge_dict = success_task_to_knowledge_dict
        self.failed_task_info_set = failed_task_info_set


class CoSTEERKnowledgeBaseV1(EvolvingKnowledgeBase):
    """
    CoSTEER知识库V1版本
    
    第一版知识库实现，使用简单的字典和集合来存储实现轨迹和成功任务信息。
    """
    def __init__(self, path: str | Path = None) -> None:
        """
        初始化V1版本知识库
        
        Parameters:
        -----------
        path : str | Path, optional
            知识库路径
        """
        self.implementation_trace: dict[str, CoSTEERKnowledge] = dict()
        self.success_task_info_set: set[str] = set()

        self.task_to_embedding = dict()
        super().__init__(path)

    def query(self) -> CoSTEERQueriedKnowledge | None:
        """
        查询知识库获取查询知识
        
        Returns:
        --------
        CoSTEERQueriedKnowledge | None
            查询到的知识对象，该方法未实现
        """
        raise NotImplementedError


class CoSTEERQueriedKnowledgeV1(CoSTEERQueriedKnowledge):
    """
    CoSTEER V1版本查询知识类
    
    扩展了基础查询知识类，增加了失败轨迹和相似成功知识的存储。
    """
    def __init__(
        self,
        *args,
        task_to_former_failed_traces: dict = {},
        task_to_similar_task_successful_knowledge: dict = {},
        **kwargs,
    ) -> None:
        """
        初始化V1版本查询知识对象
        
        Parameters:
        -----------
        task_to_former_failed_traces : dict
            任务到先前失败轨迹的映射
        task_to_similar_task_successful_knowledge : dict
            任务到相似任务成功知识的映射
        """
        self.task_to_former_failed_traces = task_to_former_failed_traces
        self.task_to_similar_task_successful_knowledge = task_to_similar_task_successful_knowledge
        super().__init__(*args, **kwargs)


class CoSTEERRAGStrategyV1(CoSTEERRAGStrategy):
    """
    CoSTEER V1版本RAG策略
    
    第一版检索增强生成策略实现。
    """
    def __init__(self, settings: CoSTEERSettings, *args, **kwargs) -> None:
        """
        初始化V1版本RAG策略
        
        Parameters:
        -----------
        settings : CoSTEERSettings
            CoSTEER设置对象
        """
        super().__init__(*args, **kwargs)
        self.current_generated_trace_count = 0
        self.settings = settings

    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep],
        *,
        return_knowledge: bool = False,
    ) -> Knowledge | None:
        """
        生成知识（未实现）
        
        Parameters:
        -----------
        evolving_trace : list[EvoStep]
            演化轨迹列表
        return_knowledge : bool
            是否返回知识对象
            
        Returns:
        --------
        Knowledge | None
            知识对象或None
        """
        raise NotImplementedError(
            "This method should be considered as an un-implemented method because we encourage everyone to use v2."
        )
        if len(evolving_trace) == self.current_generated_trace_count:
            return
        else:
            for trace_index in range(
                self.current_generated_trace_count,
                len(evolving_trace),
            ):
                evo_step = evolving_trace[trace_index]
                implementations = evo_step.evolvable_subjects
                feedback = evo_step.feedback
                for task_index in range(len(implementations.sub_tasks)):
                    target_task = implementations.sub_tasks[task_index]
                    target_task_information = target_task.get_task_information()
                    implementation = implementations.sub_workspace_list[task_index]
                    single_feedback = feedback[task_index]
                    if single_feedback is None:
                        continue
                    single_knowledge = CoSTEERKnowledge(
                        target_task=target_task,
                        implementation=implementation,
                        feedback=single_feedback,
                    )
                    if target_task_information not in self.knowledgebase.success_task_info_set:
                        self.knowledgebase.implementation_trace.setdefault(
                            target_task_information,
                            [],
                        ).append(single_knowledge)

                        if single_feedback.final_decision == True:
                            self.knowledgebase.success_task_info_set.add(
                                target_task_information,
                            )
            self.current_generated_trace_count = len(evolving_trace)

    def query(
        self,
        evo: EvolvableSubjects,
        evolving_trace: list[EvoStep],
    ) -> CoSTEERQueriedKnowledge | None:
        """
        查询知识（未实现）
        
        Parameters:
        -----------
        evo : EvolvableSubjects
            可演化主体对象
        evolving_trace : list[EvoStep]
            演化轨迹列表
            
        Returns:
        --------
        CoSTEERQueriedKnowledge | None
            查询到的知识对象
        """
        raise NotImplementedError(
            "This method should be considered as an un-implemented method because we encourage everyone to use v2."
        )
        v1_query_former_trace_limit = self.settings.v1_query_former_trace_limit
        v1_query_similar_success_limit = self.settings.v1_query_similar_success_limit
        fail_task_trial_limit = self.settings.fail_task_trial_limit

        queried_knowledge = CoSTEERQueriedKnowledgeV1()
        for target_task in evo.sub_tasks:
            target_task_information = target_task.get_task_information()
            if target_task_information in self.knowledgebase.success_task_info_set:
                queried_knowledge.success_task_to_knowledge_dict[target_task_information] = (
                    self.knowledgebase.implementation_trace[target_task_information][-1]
                )
            elif (
                len(
                    self.knowledgebase.implementation_trace.setdefault(
                        target_task_information,
                        [],
                    ),
                )
                >= fail_task_trial_limit
            ):
                queried_knowledge.failed_task_info_set.add(target_task_information)
            else:
                queried_knowledge.task_to_former_failed_traces[target_task_information] = (
                    self.knowledgebase.implementation_trace.setdefault(
                        target_task_information,
                        [],
                    )[-v1_query_former_trace_limit:]
                )

                knowledge_base_success_task_list = list(
                    self.knowledgebase.success_task_info_set,
                )
                similarity = calculate_embedding_distance_between_str_list(
                    [target_task_information],
                    knowledge_base_success_task_list,
                )[0]
                similar_indexes = sorted(
                    range(len(similarity)),
                    key=lambda i: similarity[i],
                    reverse=True,
                )[:v1_query_similar_success_limit]
                similar_successful_knowledge = [
                    self.knowledgebase.implementation_trace.setdefault(
                        knowledge_base_success_task_list[index],
                        [],
                    )[-1]
                    for index in similar_indexes
                ]
                queried_knowledge.task_to_similar_task_successful_knowledge[target_task_information] = (
                    similar_successful_knowledge
                )
        return queried_knowledge


class CoSTEERQueriedKnowledgeV2(CoSTEERQueriedKnowledgeV1):
    """
    CoSTEER V2版本查询知识类
    
    扩展了V1版本查询知识类，增加了相似错误成功知识的存储。
    """
    # Aggregation of knowledge
    def __init__(
        self,
        task_to_former_failed_traces: dict = {},
        task_to_similar_task_successful_knowledge: dict = {},
        task_to_similar_error_successful_knowledge: dict = {},
        **kwargs,
    ) -> None:
        """
        初始化V2版本查询知识对象
        
        Parameters:
        -----------
        task_to_former_failed_traces : dict
            任务到先前失败轨迹的映射
        task_to_similar_task_successful_knowledge : dict
            任务到相似任务成功知识的映射
        task_to_similar_error_successful_knowledge : dict
            任务到相似错误成功知识的映射
        """
        self.task_to_similar_error_successful_knowledge = task_to_similar_error_successful_knowledge
        super().__init__(
            task_to_former_failed_traces=task_to_former_failed_traces,
            task_to_similar_task_successful_knowledge=task_to_similar_task_successful_knowledge,
            **kwargs,
        )


class CoSTEERRAGStrategyV2(CoSTEERRAGStrategy):
    """
    CoSTEER V2版本RAG策略
    
    第二版检索增强生成策略实现，使用图结构进行知识管理。
    """
    def __init__(self, settings: CoSTEERSettings, *args, **kwargs) -> None:
        """
        初始化V2版本RAG策略
        
        Parameters:
        -----------
        settings : CoSTEERSettings
            CoSTEER设置对象
        """
        super().__init__(*args, **kwargs)
        self.current_generated_trace_count = 0
        self.settings = settings

    def generate_knowledge(
        self,
        evolving_trace: list[EvoStep],
        *,
        return_knowledge: bool = False,
    ) -> Knowledge | None:
        """
        生成知识
        
        从演化轨迹中提取知识并存储到知识库中。
        
        Parameters:
        -----------
        evolving_trace : list[EvoStep]
            演化轨迹列表
        return_knowledge : bool
            是否返回知识对象
            
        Returns:
        --------
        Knowledge | None
            知识对象或None
        """
        if len(evolving_trace) == self.current_generated_trace_count:
            return None

        else:
            for trace_index in range(self.current_generated_trace_count, len(evolving_trace)):
                evo_step = evolving_trace[trace_index]
                implementations = evo_step.evolvable_subjects
                feedback = evo_step.feedback
                for task_index in range(len(implementations.sub_tasks)):
                    target_task = implementations.sub_tasks[task_index]
                    target_task_information = target_task.get_task_information()
                    implementation = implementations.sub_workspace_list[task_index]
                    single_feedback: CoSTEERSingleFeedback = feedback[task_index]
                    if implementation is None or single_feedback is None:
                        continue
                    single_knowledge = CoSTEERKnowledge(
                        target_task=target_task,
                        implementation=implementation,
                        feedback=single_feedback,
                    )
                    if (
                        target_task_information not in self.knowledgebase.success_task_to_knowledge_dict
                        and implementation is not None
                    ):
                        if target_task_information not in self.knowledgebase.task_to_component_nodes:
                            self.knowledgebase.task_to_component_nodes[
                                target_task_information
                            ] = self.analyze_component(
                                target_task_information,
                            )
                        self.knowledgebase.working_trace_knowledge.setdefault(target_task_information, []).append(
                            single_knowledge,
                        )  # save to working trace
                        if single_feedback.final_decision == True:
                            self.knowledgebase.success_task_to_knowledge_dict.setdefault(
                                target_task_information,
                                single_knowledge,
                            )
                            # Do summary for the last step and update the knowledge graph
                            self.knowledgebase.update_success_task(
                                target_task_information,
                            )
                        else:
                            # generate error node and store into knowledge base
                            error_analysis_result = []
                            if single_feedback.return_checking:
                                error_analysis_result = self.analyze_error(
                                    single_feedback.return_checking,
                                    feedback_type="value",
                                )
                            else:
                                error_analysis_result = self.analyze_error(
                                    single_feedback.execution,
                                    feedback_type="execution",
                                )
                            self.knowledgebase.working_trace_error_analysis.setdefault(
                                target_task_information,
                                [],
                            ).append(
                                error_analysis_result,
                            )  # save to working trace error record, for graph update

            self.current_generated_trace_count = len(evolving_trace)
            return None

    def query(self, evo: EvolvableSubjects, evolving_trace: list[EvoStep]) -> CoSTEERQueriedKnowledge | None:
        """
        查询知识
        
        根据可演化主体和演化轨迹查询相关知识。
        
        Parameters:
        -----------
        evo : EvolvableSubjects
            可演化主体对象
        evolving_trace : list[EvoStep]
            演化轨迹列表
            
        Returns:
        --------
        CoSTEERQueriedKnowledge | None
            查询到的知识对象
        """
        conf_knowledge_sampler = self.settings.v2_knowledge_sampler
        queried_knowledge_v2 = CoSTEERQueriedKnowledgeV2(
            success_task_to_knowledge_dict=self.knowledgebase.success_task_to_knowledge_dict,
        )

        queried_knowledge_v2 = self.former_trace_query(
            evo,
            queried_knowledge_v2,
            self.settings.v2_query_former_trace_limit,
            self.settings.v2_add_fail_attempt_to_latest_successful_execution,
        )
        queried_knowledge_v2 = self.component_query(
            evo,
            queried_knowledge_v2,
            self.settings.v2_query_component_limit,
            knowledge_sampler=conf_knowledge_sampler,
        )
        queried_knowledge_v2 = self.error_query(
            evo,
            queried_knowledge_v2,
            self.settings.v2_query_error_limit,
            knowledge_sampler=conf_knowledge_sampler,
        )
        return queried_knowledge_v2

    def analyze_component(
        self,
        target_task_information,
    ) -> list[UndirectedNode]:  # Hardcode: certain component nodes
        """
        分析任务组件
        
        使用LLM分析目标任务信息中的组件节点。
        
        Parameters:
        -----------
        target_task_information : str
            目标任务信息
            
        Returns:
        --------
        list[UndirectedNode]
            组件节点列表
        """
        all_component_nodes = self.knowledgebase.graph.get_all_nodes_by_label_list(["component"])
        if not len(all_component_nodes):
            return []
        all_component_content = ""
        for _, component_node in enumerate(all_component_nodes):
            all_component_content += f"{component_node.content}, \n"
        analyze_component_system_prompt = T(".prompts:analyze_component_prompt_v1_system").r(
            all_component_content=all_component_content,
        )

        analyze_component_user_prompt = target_task_information
        try:
            component_no_list = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    system_prompt=analyze_component_system_prompt,
                    user_prompt=analyze_component_user_prompt,
                    json_mode=True,
                    json_target_type=List[int],
                ),
            )["component_no_list"]
            return [all_component_nodes[index - 1] for index in sorted(list(set(component_no_list)))]
        except:
            logger.warning("Error when analyzing components.")
            analyze_component_user_prompt = "Your response is not a valid component index list."

        return []

    def analyze_error(
        self,
        single_feedback,
        feedback_type="execution",
    ) -> list[
        UndirectedNode | str
    ]:  # Hardcode: Raised errors, existed error nodes + not existed error nodes(here, they are strs)
        """
        分析错误信息
        
        从反馈信息中提取错误类型和内容。
        
        Parameters:
        -----------
        single_feedback : str
            单个反馈信息
        feedback_type : str
            反馈类型（"execution" 或 "value"）
            
        Returns:
        --------
        list[UndirectedNode | str]
            错误节点或错误内容列表
        """
        if feedback_type == "execution":
            match = re.search(
                r'File "(?P<file>.+)", line (?P<line>\d+), in (?P<function>.+)\n\s+(?P<error_line>.+)\n(?P<error_type>\w+): (?P<error_message>.+)',
                single_feedback,
            )
            if match:
                error_details = match.groupdict()
                # last_traceback = f'File "{error_details["file"]}", line {error_details["line"]}, in {error_details["function"]}\n    {error_details["error_line"]}'
                error_type = error_details["error_type"]
                error_line = error_details["error_line"]
                error_contents = [f"ErrorType: {error_type}" + "\n" + f"Error line: {error_line}"]
            else:
                error_contents = ["Undefined Error"]
        elif feedback_type == "value":  # value check error
            value_check_types = r"The source dataframe and the ground truth dataframe have different rows count.|The source dataframe and the ground truth dataframe have different index.|Some values differ by more than the tolerance of 1e-6.|No sufficient correlation found when shifting up|Something wrong happens when naming the multi indices of the dataframe."
            error_contents = re.findall(value_check_types, single_feedback)
        else:
            error_contents = ["Undefined Error"]

        all_error_nodes = self.knowledgebase.graph.get_all_nodes_by_label_list(["error"])
        if not len(all_error_nodes):
            return error_contents
        else:
            error_list = []
            for error_content in error_contents:
                for error_node in all_error_nodes:
                    if error_content == error_node.content:
                        error_list.append(error_node)
                    else:
                        error_list.append(error_content)
                    if error_list[-1] in error_list[:-1]:
                        error_list.pop()

            return error_list

    def former_trace_query(
        self,
        evo: EvolvableSubjects,
        queried_knowledge_v2: CoSTEERQueriedKnowledgeV2,
        v2_query_former_trace_limit: int = 5,
        v2_add_fail_attempt_to_latest_successful_execution: bool = False,
    ) -> Union[CoSTEERQueriedKnowledge, set]:
        """
        查询先前轨迹知识
        
        查询工作轨迹中的先前知识，并找出尝试次数超过限制的失败任务。
        
        Parameters:
        -----------
        evo : EvolvableSubjects
            可演化主体对象
        queried_knowledge_v2 : CoSTEERQueriedKnowledgeV2
            V2版本查询知识对象
        v2_query_former_trace_limit : int
            查询先前轨迹的限制数量
        v2_add_fail_attempt_to_latest_successful_execution : bool
            是否添加失败尝试到最新成功执行
            
        Returns:
        --------
        Union[CoSTEERQueriedKnowledge, set]
            查询知识对象或集合
        """
        fail_task_trial_limit = self.settings.fail_task_trial_limit

        for target_task in evo.sub_tasks:
            target_task_information = target_task.get_task_information()
            # 检查任务是否已成功或已标记为失败
            if (
                target_task_information not in self.knowledgebase.success_task_to_knowledge_dict
                and target_task_information in self.knowledgebase.working_trace_knowledge
                and len(self.knowledgebase.working_trace_knowledge[target_task_information]) >= fail_task_trial_limit
            ):
                queried_knowledge_v2.failed_task_info_set.add(target_task_information)

            # 查询任务的先前失败轨迹
            if (
                target_task_information not in self.knowledgebase.success_task_to_knowledge_dict
                and target_task_information not in queried_knowledge_v2.failed_task_info_set
                and target_task_information in self.knowledgebase.working_trace_knowledge
            ):
                former_trace_knowledge = copy.copy(
                    self.knowledgebase.working_trace_knowledge[target_task_information],
                )
                # 在先前轨迹查询中，我们将按照以下顺序删除正确的轨迹：
                # [..., value_generated_flag为True, value_generated_flag为False, ...]
                # 因为我们认为这种顺序意味着试验的恶化（就像错误的梯度下降）
                current_index = 1
                while current_index < len(former_trace_knowledge):
                    # 如果当前轨迹没有返回检查而前一个轨迹有返回检查，则删除当前轨迹
                    if (
                        not former_trace_knowledge[current_index].feedback.return_checking
                        and former_trace_knowledge[current_index - 1].feedback.return_checking
                    ):
                        former_trace_knowledge.pop(current_index)
                    else:
                        current_index += 1

                latest_attempt = None
                # 如果最后一次成功执行不是工作轨迹中的最后一个，这意味着我们已经尝试过纠正它
                # 我们应该告诉代理这个失败的尝试，以避免将来出现无限循环
                if v2_add_fail_attempt_to_latest_successful_execution:
                    if (
                        len(former_trace_knowledge) > 0
                        and len(self.knowledgebase.working_trace_knowledge[target_task_information]) > 1
                        and self.knowledgebase.working_trace_knowledge[target_task_information].index(
                            former_trace_knowledge[-1]
                        )
                        < len(self.knowledgebase.working_trace_knowledge[target_task_information]) - 1
                    ):
                        latest_attempt = self.knowledgebase.working_trace_knowledge[target_task_information][-1]

                # 保存最近的失败轨迹和最新的尝试
                queried_knowledge_v2.task_to_former_failed_traces[target_task_information] = (
                    former_trace_knowledge[-v2_query_former_trace_limit:],
                    latest_attempt,
                )
            else:
                queried_knowledge_v2.task_to_former_failed_traces[target_task_information] = ([], None)

        return queried_knowledge_v2

    def component_query(
        self,
        evo: EvolvableSubjects,
        queried_knowledge_v2: CoSTEERQueriedKnowledgeV2,
        v2_query_component_limit: int = 5,
        knowledge_sampler: float = 1.0,
    ) -> CoSTEERQueriedKnowledge | None:
        """
        组件查询
        
        根据任务组件信息查询相似的成功任务知识。
        
        Parameters:
        -----------
        evo : EvolvableSubjects
            可演化主体对象
        queried_knowledge_v2 : CoSTEERQueriedKnowledgeV2
            V2版本查询知识对象
        v2_query_component_limit : int
            组件查询限制数量
        knowledge_sampler : float
            知识采样率
            
        Returns:
        --------
        CoSTEERQueriedKnowledge | None
            查询知识对象
        """
        for target_task in evo.sub_tasks:
            target_task_information = target_task.get_task_information()
            # 如果任务已成功或已标记为失败，则清空相似任务成功知识
            if (
                target_task_information in self.knowledgebase.success_task_to_knowledge_dict
                or target_task_information in queried_knowledge_v2.failed_task_info_set
            ):
                queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = []
            else:
                # 如果任务未分析过组件，则先分析组件
                if target_task_information not in self.knowledgebase.task_to_component_nodes:
                    self.knowledgebase.task_to_component_nodes[target_task_information] = self.analyze_component(
                        target_task_information,
                    )
                component_analysis_result = self.knowledgebase.task_to_component_nodes[target_task_information]

                # 如果组件分析结果大于1个，则通过交集查询任务描述节点
                if len(component_analysis_result) > 1:
                    task_des_node_list = self.knowledgebase.graph_query_by_intersection(
                        component_analysis_result,
                        constraint_labels=["task_description"],
                    )
                    # 计算每个组件的约束数量
                    single_component_constraint = (v2_query_component_limit // len(component_analysis_result)) + 1
                else:
                    task_des_node_list = []
                    single_component_constraint = v2_query_component_limit
                
                queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = []
                
                # 遍历组件节点，查询相关的任务描述节点
                for component_node in component_analysis_result:
                    # 反向迭代，与交集搜索的折衷
                    count = 0
                    for task_des_node in self.knowledgebase.graph_query_by_node(
                        node=component_node,
                        step=1,
                        constraint_labels=["task_description"],
                        block=True,
                    )[::-1]:
                        if task_des_node not in task_des_node_list:
                            task_des_node_list.append(task_des_node)
                            count += 1
                        if count >= single_component_constraint:
                            break

                # 遍历任务描述节点，查询相关的成功实现节点
                for node in task_des_node_list:
                    for searched_node in self.knowledgebase.graph_query_by_node(
                        node=node,
                        step=50,
                        constraint_labels=[
                            "task_success_implement",
                        ],
                        block=True,
                    ):
                        # 如果找到成功实现节点，则获取对应的知识
                        if searched_node.label == "task_success_implement":
                            target_knowledge = self.knowledgebase.node_to_implementation_knowledge_dict[
                                searched_node.id
                            ]
                        # 避免重复添加相同知识
                        if (
                            target_knowledge
                            not in queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                                target_task_information
                            ]
                        ):
                            queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                                target_task_information
                            ].append(target_knowledge)

                # 最后添加基于嵌入相似度的相关知识
                knowledge_base_success_task_list = list(self.knowledgebase.success_task_to_knowledge_dict)

                # 计算目标任务与知识库中成功任务的嵌入距离
                similarity = calculate_embedding_distance_between_str_list(
                    [target_task_information],
                    knowledge_base_success_task_list,
                )[0]
                # 按相似度排序，获取最相似的任务索引
                similar_indexes = sorted(
                    range(len(similarity)),
                    key=lambda i: similarity[i],
                    reverse=True,
                )
                # 获取基于嵌入相似度的成功知识
                embedding_similar_successful_knowledge = [
                    self.knowledgebase.success_task_to_knowledge_dict[knowledge_base_success_task_list[index]]
                    for index in similar_indexes
                ]
                # 将嵌入相似的知识添加到查询结果中
                for knowledge in embedding_similar_successful_knowledge:
                    if (
                        knowledge
                        not in queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information]
                    ):
                        queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information].append(
                            knowledge
                        )

                # 根据知识采样率进行采样
                if knowledge_sampler > 0:
                    queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = [
                        knowledge
                        for knowledge in queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                            target_task_information
                        ]
                        if random.uniform(0, 1) <= knowledge_sampler
                    ]

                # 确保至少一半的知识来自GT（Ground Truth）
                queried_knowledge_list = queried_knowledge_v2.task_to_similar_task_successful_knowledge[
                    target_task_information
                ]
                # 筛选出基于GT的决策知识
                queried_from_gt_knowledge_list = [
                    knowledge
                    for knowledge in queried_knowledge_list
                    if knowledge.feedback is not None
                    and (
                        hasattr(knowledge.feedback, "final_decision_based_on_gt")
                        and knowledge.feedback.final_decision_based_on_gt == True
                    )
                ]
                # 筛选出非GT的知识
                queried_without_gt_knowledge_list = [
                    knowledge for knowledge in queried_knowledge_list if knowledge not in queried_from_gt_knowledge_list
                ]
                # 计算来自GT的知识数量，确保至少占一半
                queried_from_gt_knowledge_count = max(
                    min((v2_query_component_limit // 2 + 1), len(queried_from_gt_knowledge_list)),
                    v2_query_component_limit - len(queried_without_gt_knowledge_list),
                )
                # 重新组织查询知识列表，确保GT知识占一半以上
                queried_knowledge_v2.task_to_similar_task_successful_knowledge[target_task_information] = (
                    queried_from_gt_knowledge_list[:queried_from_gt_knowledge_count]
                    + queried_without_gt_knowledge_list[: v2_query_component_limit - queried_from_gt_knowledge_count]
                )

        return queried_knowledge_v2

    def error_query(
        self,
        evo: EvolvableSubjects,
        queried_knowledge_v2: CoSTEERQueriedKnowledgeV2,
        v2_query_error_limit: int = 5,
        knowledge_sampler: float = 1.0,
    ) -> CoSTEERQueriedKnowledge | None:
        """
        错误查询
        
        根据错误信息查询相似的成功任务知识。
        
        Parameters:
        -----------
        evo : EvolvableSubjects
            可演化主体对象
        queried_knowledge_v2 : CoSTEERQueriedKnowledgeV2
            V2版本查询知识对象
        v2_query_error_limit : int
            错误查询限制数量
        knowledge_sampler : float
            知识采样率
            
        Returns:
        --------
        CoSTEERQueriedKnowledge | None
            查询知识对象
        """
        for task_index, target_task in enumerate(evo.sub_tasks):
            target_task_information = target_task.get_task_information()
            queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = []
            # 如果任务已成功或已标记为失败，则清空相似错误成功知识
            if (
                target_task_information in self.knowledgebase.success_task_to_knowledge_dict
                or target_task_information in queried_knowledge_v2.failed_task_info_set
            ):
                queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = []
            else:
                queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = []
                # 如果存在工作轨迹错误分析记录且不为空，并且存在先前失败轨迹
                if (
                    target_task_information in self.knowledgebase.working_trace_error_analysis
                    and len(self.knowledgebase.working_trace_error_analysis[target_task_information]) > 0
                    and len(queried_knowledge_v2.task_to_former_failed_traces[target_task_information]) > 0
                ):
                    # 获取查询的最后轨迹
                    queried_last_trace = queried_knowledge_v2.task_to_former_failed_traces[target_task_information][0][
                        -1
                    ]
                    # 获取目标索引
                    target_index = self.knowledgebase.working_trace_knowledge[target_task_information].index(
                        queried_last_trace,
                    )
                    # 获取最后知识的错误分析结果
                    last_knowledge_error_analysis_result = self.knowledgebase.working_trace_error_analysis[
                        target_task_information
                    ][target_index]
                else:
                    last_knowledge_error_analysis_result = []

                # 处理错误节点
                error_nodes = []
                for error_node in last_knowledge_error_analysis_result:
                    # 如果错误节点不是UndirectedNode类型，则通过内容获取节点
                    if not isinstance(error_node, UndirectedNode):
                        error_node = self.knowledgebase.graph_get_node_by_content(content=error_node)
                        if error_node is None:
                            continue
                    error_nodes.append(error_node)

                # 如果错误节点数量大于1，则通过交集查询任务轨迹节点
                if len(error_nodes) > 1:
                    task_trace_node_list = self.knowledgebase.graph_query_by_intersection(
                        error_nodes,
                        constraint_labels=["task_trace"],
                        output_intersection_origin=True,
                    )
                    # 计算每个错误的约束数量
                    single_error_constraint = (v2_query_error_limit // len(error_nodes)) + 1
                else:
                    task_trace_node_list = []
                    single_error_constraint = v2_query_error_limit
                
                # 遍历错误节点，查询相关的任务轨迹节点
                for error_node in error_nodes:
                    # 反向迭代，与交集搜索的折衷
                    count = 0
                    for task_trace_node in self.knowledgebase.graph_query_by_node(
                        node=error_node,
                        step=1,
                        constraint_labels=["task_trace"],
                        block=True,
                    )[::-1]:
                        if task_trace_node not in task_trace_node_list:
                            task_trace_node_list.append([[error_node], task_trace_node])
                            count += 1
                        if count >= single_error_constraint:
                            break

                # for error_node in last_knowledge_error_analysis_result:
                #     if not isinstance(error_node, UndirectedNode):
                #         error_node = self.knowledgebase.graph_get_node_by_content(content=error_node)
                #         if error_node is None:
                #             continue
                #     for searched_node in self.knowledgebase.graph_query_by_node(
                #         node=error_node,
                #         step=1,
                #         constraint_labels=["task_trace"],
                #         block=True,
                #     ):
                #         if searched_node not in [node[0] for node in task_trace_node_list]:
                #             task_trace_node_list.append((searched_node, error_node.content))

                same_error_success_knowledge_pair_list = []
                same_error_success_node_set = set()
                # 遍历任务轨迹节点，查询相关的成功实现节点
                for error_node_list, trace_node in task_trace_node_list:
                    for searched_trace_success_node in self.knowledgebase.graph_query_by_node(
                        node=trace_node,
                        step=50,
                        constraint_labels=[
                            "task_trace",
                            "task_success_implement",
                            "task_description",
                        ],
                        block=True,
                    ):
                        # 如果找到未处理过的成功实现节点
                        if (
                            searched_trace_success_node not in same_error_success_node_set
                            and searched_trace_success_node.label == "task_success_implement"
                        ):
                            same_error_success_node_set.add(searched_trace_success_node)

                            # 获取轨迹知识和成功知识
                            trace_knowledge = self.knowledgebase.node_to_implementation_knowledge_dict[trace_node.id]
                            success_knowledge = self.knowledgebase.node_to_implementation_knowledge_dict[
                                searched_trace_success_node.id
                            ]
                            # 构造错误内容字符串
                            error_content = ""
                            for index, error_node in enumerate(error_node_list):
                                error_content += f"{index+1}. {error_node.content}; "
                            # 添加到相同错误成功知识对列表
                            same_error_success_knowledge_pair_list.append(
                                (
                                    error_content,
                                    (trace_knowledge, success_knowledge),
                                ),
                            )

                # 根据知识采样率进行采样
                if knowledge_sampler > 0:
                    same_error_success_knowledge_pair_list = [
                        knowledge
                        for knowledge in same_error_success_knowledge_pair_list
                        if random.uniform(0, 1) <= knowledge_sampler
                    ]

                # 限制相同错误成功知识对的数量
                same_error_success_knowledge_pair_list = same_error_success_knowledge_pair_list[:v2_query_error_limit]
                # 保存查询结果
                queried_knowledge_v2.task_to_similar_error_successful_knowledge[target_task_information] = (
                    same_error_success_knowledge_pair_list
                )

        return queried_knowledge_v2


class CoSTEERKnowledgeBaseV2(EvolvingKnowledgeBase):
    """
    CoSTEER知识库V2版本
    
    第二版知识库实现，使用无向图结构进行知识管理，支持更复杂的知识查询和关联。
    """
    def __init__(self, init_component_list=None, path: str | Path = None) -> None:
        """
        初始化V2版本知识库
        
        Parameters:
        -----------
        init_component_list : list, optional
            初始化组件列表
        path : str | Path, optional
            知识库路径
        """
        """
        Load knowledge, offer brief information of knowledge and common handle interfaces
        """
        self.graph: UndirectedGraph = UndirectedGraph(Path.cwd() / "graph.pkl")
        logger.info(f"CoSTEER Knowledge Graph loaded, size={self.graph.size()}")

        if init_component_list:
            for component in init_component_list:
                exist_node = self.graph.get_node_by_content(content=component)
                node = exist_node if exist_node else UndirectedNode(content=component, label="component")
                self.graph.add_nodes(node=node, neighbors=[])

        # A dict containing all working trace until they fail or succeed
        self.working_trace_knowledge = {}

        # A dict containing error analysis each step aligned with working trace
        self.working_trace_error_analysis = {}

        # Add already success task
        self.success_task_to_knowledge_dict = {}

        # key:node_id(for task trace and success implement), value:knowledge instance(aka 'CoSTEERKnowledge')
        self.node_to_implementation_knowledge_dict = {}

        # store the task description to component nodes
        self.task_to_component_nodes = {}

    def get_all_nodes_by_label(self, label: str) -> list[UndirectedNode]:
        """
        根据标签获取所有节点
        
        Parameters:
        -----------
        label : str
            节点标签
            
        Returns:
        --------
        list[UndirectedNode]
            具有指定标签的所有节点列表
        """
        return self.graph.get_all_nodes_by_label(label)

    def update_success_task(
        self,
        success_task_info: str,
    ):  # Transfer the success tasks' working trace to knowledge storage & graph
        """
        更新成功任务
        
        将成功任务的工作轨迹转移到知识存储和图结构中。
        
        Parameters:
        -----------
        success_task_info : str
            成功任务信息
        """
        success_task_trace = self.working_trace_knowledge[success_task_info]
        success_task_error_analysis_record = (
            self.working_trace_error_analysis[success_task_info]
            if success_task_info in self.working_trace_error_analysis
            else []
        )
        task_des_node = UndirectedNode(content=success_task_info, label="task_description")
        self.graph.add_nodes(
            node=task_des_node,
            neighbors=self.task_to_component_nodes[success_task_info],
        )  # 1st version, we assume that all component nodes are given
        for index, trace_unit in enumerate(success_task_trace):  # every unit: single_knowledge
            neighbor_nodes = [task_des_node]
            if index != len(success_task_trace) - 1:
                trace_node = UndirectedNode(
                    content=trace_unit.get_implementation_and_feedback_str(),
                    label="task_trace",
                )
                self.node_to_implementation_knowledge_dict[trace_node.id] = trace_unit
                for node_index, error_node in enumerate(success_task_error_analysis_record[index]):
                    if type(error_node).__name__ == "str":
                        queried_node = self.graph.get_node_by_content(content=error_node)
                        if queried_node is None:
                            new_error_node = UndirectedNode(content=error_node, label="error")
                            self.graph.add_node(node=new_error_node)
                            success_task_error_analysis_record[index][node_index] = new_error_node
                        else:
                            success_task_error_analysis_record[index][node_index] = queried_node
                neighbor_nodes.extend(success_task_error_analysis_record[index])
                self.graph.add_nodes(node=trace_node, neighbors=neighbor_nodes)
            else:
                success_node = UndirectedNode(
                    content=trace_unit.get_implementation_and_feedback_str(),
                    label="task_success_implement",
                )
                self.graph.add_nodes(node=success_node, neighbors=neighbor_nodes)
                self.node_to_implementation_knowledge_dict[success_node.id] = trace_unit

    def query(self):
        """
        查询知识库（未实现）
        """
        pass

    def graph_get_node_by_content(self, content: str) -> UndirectedNode:
        """
        根据内容获取图节点
        
        Parameters:
        -----------
        content : str
            节点内容
            
        Returns:
        --------
        UndirectedNode
            匹配的节点对象
        """
        return self.graph.get_node_by_content(content=content)

    def graph_query_by_content(
        self,
        content: Union[str, list[str]],
        topk_k: int = 5,
        step: int = 1,
        constraint_labels: list[str] = None,
        constraint_node: UndirectedNode = None,
        similarity_threshold: float = 0.0,
        constraint_distance: float = 0,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        根据内容查询图
        
        通过内容相似性和连接关系搜索图，返回空列表如果没有节点链接近约束节点。
        
        Parameters:
        -----------
        content : Union[str, list[str]]
            查询内容
        topk_k : int
            每个查询的上限数量
        step : int
            查询步数
        constraint_labels : list[str]
            输出节点的标签约束
        constraint_node : UndirectedNode
            约束节点
        similarity_threshold : float
            相似度阈值
        constraint_distance : float
            约束距离
        block : bool
            是否阻止搜索流经非约束类型节点
            
        Returns:
        --------
        list[UndirectedNode]
            节点列表
        """

        return self.graph.query_by_content(
            content=content,
            topk_k=topk_k,
            step=step,
            constraint_labels=constraint_labels,
            constraint_node=constraint_node,
            similarity_threshold=similarity_threshold,
            constraint_distance=constraint_distance,
            block=block,
        )

    def graph_query_by_node(
        self,
        node: UndirectedNode,
        step: int = 1,
        constraint_labels: list[str] = None,
        constraint_node: UndirectedNode = None,
        constraint_distance: float = 0,
        block: bool = False,
    ) -> list[UndirectedNode]:
        """
        根据节点查询图
        
        通过连接关系搜索图，返回空列表如果没有节点链接近约束节点。
        
        Parameters:
        -----------
        node : UndirectedNode
            起始节点
        step : int
            最大搜索步数
        constraint_labels : list[str]
            输出节点的标签
        constraint_node : UndirectedNode
            约束节点
        constraint_distance : float
            输出节点与约束节点的最大距离
        block : bool
            除了起始节点外，搜索只能流经约束标签类型的节点
            
        Returns:
        --------
        list[UndirectedNode]
            节点列表
        """
        nodes = self.graph.query_by_node(
            node=node,
            step=step,
            constraint_labels=constraint_labels,
            constraint_node=constraint_node,
            constraint_distance=constraint_distance,
            block=block,
        )
        return nodes

    def graph_query_by_intersection(
        self,
        nodes: list[UndirectedNode],
        steps: int = 1,
        constraint_labels: list[str] = None,
        output_intersection_origin: bool = False,
    ) -> list[UndirectedNode] | list[list[list[UndirectedNode], UndirectedNode]]:
        """
        根据节点交集查询图
        
        通过节点交集搜索图，高频交叉的节点在列表中有优先顺序。
        
        Parameters:
        -----------
        nodes : list[UndirectedNode]
            节点列表
        steps : int
            最大搜索步数
        constraint_labels : list[str]
            输出节点的标签
        output_intersection_origin : bool
            输出形成交集节点的原始节点列表
            
        Returns:
        --------
        list[UndirectedNode] | list[list[list[UndirectedNode], UndirectedNode]]
            节点列表或包含原始节点信息的列表
        """
        node_count = len(nodes)
        assert node_count >= 2, "nodes length must >=2"
        intersection_node_list = []
        if output_intersection_origin:
            origin_list = []
        
        # 从节点数量开始，递减到2，生成所有可能的组合
        for k in range(node_count, 1, -1):
            # 生成k个节点的所有可能组合
            possible_combinations = combinations(nodes, k)
            for possible_combination in possible_combinations:
                node_list = list(possible_combination)
                # 获取节点列表的交集节点
                intersection_node_list.extend(
                    self.graph.get_nodes_intersection(node_list, steps=steps, constraint_labels=constraint_labels),
                )
                # 如果需要输出交集的原始节点，则保存原始节点列表
                if output_intersection_origin:
                    for _ in range(len(intersection_node_list)):
                        origin_list.append(node_list)
        
        # 按频率排序交集节点列表
        intersection_node_list_sort_by_freq = []
        for index, node in enumerate(intersection_node_list):
            # 避免重复添加相同节点
            if node not in intersection_node_list_sort_by_freq:
                # 如果需要输出原始节点信息，则添加包含原始节点的列表
                if output_intersection_origin:
                    intersection_node_list_sort_by_freq.append([origin_list[index], node])
                else:
                    intersection_node_list_sort_by_freq.append(node)

        return intersection_node_list_sort_by_freq