from __future__ import annotations

import json
import re
from typing import Dict

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.log import rdagent_logger as logger


class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        """
        初始化因子多进程演化策略对象。
        
        参数：
        - *args: 传递给父类的可变位置参数
        - **kwargs: 传递给父类的可变关键字参数
        
        属性：
        - num_loop (int): 循环计数器，用于跟踪演化循环次数
        - haveSelected (bool): 选择标志，表示是否已进行过选择操作
        """
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def error_summary(
        self,
        target_task: FactorTask,  # 目标因子任务对象
        queried_former_failed_knowledge_to_render: list,  # 查询到的先前失败知识列表
        queried_similar_error_knowledge_to_render: list,  # 查询到的相似错误知识列表
    ) -> str:
        """
        对错误进行总结，生成错误摘要。
        
        主要功能：
        1. 构建错误总结的系统提示和用户提示
        2. 动态调整提示长度以适应LLM的token限制
        3. 调用LLM生成错误总结
        
        参数：
        ----------
        target_task : FactorTask
            目标因子任务对象
            
        queried_former_failed_knowledge_to_render : list
            查询到的先前失败知识列表，用于构建提示
            
        queried_similar_error_knowledge_to_render : list
            查询到的相似错误知识列表，用于构建提示
            
        返回值：
        ------
        str: 错误总结内容，由LLM生成的文本
        """
        # 构建错误总结的系统提示
        error_summary_system_prompt = T(".prompts:evolving_strategy_error_summary_v2_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            factor_information_str=target_task.get_task_information(),
            code_and_feedback=queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str(),
        )
        
        # 最多尝试10次来减少error_summary_user_prompt的长度
        for _ in range(10):  
            # 构建错误总结的用户提示
            error_summary_user_prompt = T(".prompts:evolving_strategy_error_summary_v2_user").r(
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
            )
            
            # 检查提示长度是否超过LLM的token限制
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < APIBackend().chat_token_limit
            ):
                break
            # 如果超过限制且相似错误知识列表不为空，则移除最后一个元素
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        
        # 调用LLM生成错误总结，不使用JSON模式
        with logger.tag(f"coding"):
            error_summary_critics = APIBackend(
                use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
            ).build_messages_and_create_chat_completion(
                user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
            )
        return error_summary_critics

    def implement_one_task(
        self,
        target_task: FactorTask,  # 目标因子任务对象
        queried_knowledge: CoSTEERQueriedKnowledge,  # 查询到的知识
        workspace: FBWorkspace | None = None,  # 工作区对象，可选参数
        prev_task_feedback: CoSTEERSingleFeedback | None = None,  # 上一次任务反馈，可选参数
    ) -> str:
        """
        实现单个因子任务，生成因子代码。
        
        主要功能：
        1. 获取与目标任务相关的各种知识（成功知识、错误知识、失败知识等）
        2. 构建系统提示和用户提示
        3. 动态调整提示长度以适应LLM的token限制
        4. 调用LLM生成因子实现代码
        5. 处理解析LLM响应并提取代码
        
        参数：
        ----------
        target_task : FactorTask
            目标因子任务对象
            
        queried_knowledge : CoSTEERQueriedKnowledge
            查询到的CoSTEER知识，包含成功和失败任务的信息
            
        workspace : FBWorkspace | None
            当前工作区对象，包含现有的代码和资源，默认为None
            
        prev_task_feedback : CoSTEERSingleFeedback | None
            上一次任务反馈信息，默认为None
            
        返回值：
        ------
        str: 生成的因子代码内容，如果失败则返回空字符串
            
        关键逻辑步骤：
        ------
        1. 获取目标任务信息和相关知识
        2. 根据知识类型获取相似的成功知识和错误知识
        3. 获取先前失败的知识轨迹
        4. 构建系统提示和用户提示
        5. 动态调整提示长度防止超限
        6. 如果启用错误总结且满足条件，则生成错误总结
        7. 调用LLM生成代码实现
        8. 解析LLM响应并提取代码
        9. 如果解析失败，尝试提取Python代码块
        10. 最多重试10次，如果都失败则返回空字符串
        """
        # 获取目标因子任务信息
        target_factor_task_information = target_task.get_task_information()

        # 获取相似的成功知识
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )  # A list, [success task implement knowledge]

        # 根据知识类型获取相似的错误知识
        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        # 获取先前失败的知识
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        # 用于渲染的先前失败知识
        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        # 获取最新尝试到最新成功执行的信息
        latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
            target_factor_task_information
        ][1]
        
        # 构建系统提示
        system_prompt = T(".prompts:evolving_strategy_factor_implementation_v1_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
            queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
        )
        
        # 用于渲染的相似成功知识和错误知识
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
        
        # 动态地防止prompt超长，最多尝试10次
        for _ in range(10):  
            # 总结error（可选）
            # 如果是V2知识且启用错误总结且有相似错误知识和先前失败知识，则生成错误总结
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and len(queried_similar_error_knowledge_to_render) != 0
                and len(queried_former_failed_knowledge_to_render) != 0
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None
                
            # 构建user_prompt，开始写代码
            user_prompt = T(".prompts:evolving_strategy_factor_implementation_v2_user").r(
                factor_information_str=target_factor_task_information,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                error_summary_critics=error_summary_critics,
                latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
            )
            
            # 检查提示长度是否超过LLM的token限制
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < APIBackend().chat_token_limit
            ):
                break
            # 如果超过限制，按优先级缩减知识列表
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > len(
                queried_similar_error_knowledge_to_render,
            ):
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        
        # 最多重试10次来获取代码
        for _ in range(10):
            try:
                # 调用LLM生成代码实现，使用JSON模式
                with logger.tag(f"coding"):
                    response = APIBackend(
                        use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                    ).build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        json_target_type=Dict[str, str],
                    )

                try:
                    # 尝试解析JSON响应并提取代码
                    code = json.loads(response)["code"]
                except json.decoder.JSONDecodeError:
                    # 如果JSON解析失败，尝试提取Python代码块
                    match = re.search(r"```python(.*?)```", response, re.DOTALL)
                    if match:
                        code = match.group(1).strip()
                    else:
                        raise  # 继续重试

                return code

            except (json.decoder.JSONDecodeError, KeyError):
                # 如果解析失败，继续重试
                pass
        else:
            return ""  # 如果10次尝试后仍无法获取代码，则返回空字符串

    def assign_code_list_to_evo(self, code_list, evo):
        """
        将代码列表分配给演化项。
        
        主要功能：
        1. 遍历代码列表中的每个代码项
        2. 为非空代码项创建或更新对应的因子工作区
        3. 将代码注入到工作区的factor.py文件中
        
        参数：
        ----------
        code_list : list
            代码列表，与演化项的子任务对齐
            
        evo : EvolvingItem
            演化项对象，需要被分配代码实现
            
        返回值：
        ------
        EvolvingItem: 更新后的演化项对象
        """
        # 遍历演化项的所有子任务
        for index in range(len(evo.sub_tasks)):
            # 如果代码为空，则跳过
            if code_list[index] is None:
                continue
            # 如果子工作区为空，则创建新的因子工作区
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            # 将代码注入到工作区的factor.py文件中
            evo.sub_workspace_list[index].inject_files(**{"factor.py": code_list[index]})
        return evo