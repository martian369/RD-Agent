"""
量化金融（因子和模型）工作流程，包含会话控制功能
"""

import asyncio
from typing import Any

import fire

from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
from rdagent.components.workflow.conf import BasePropSetting
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.proposal import (
    Experiment2Feedback,
    Hypothesis2Experiment,
    HypothesisFeedback,
    HypothesisGen,
)
from rdagent.core.scenario import Scenario
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.proposal.quant_proposal import QuantTrace


class QuantRDLoop(RDLoop):
    # 定义在循环中可以跳过的异常类型
    skip_loop_error = (
        FactorEmptyError,      # 因子为空错误
        ModelEmptyError,       # 模型为空错误
    )

    def __init__(self, PROP_SETTING: BasePropSetting):
        """
        初始化量化研发循环
        
        Args:
            PROP_SETTING: 基础属性设置对象，包含各种组件的配置信息
        """
        # 创建场景实例
        scen: Scenario = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")

        # 初始化假设生成器
        self.hypothesis_gen: HypothesisGen = import_class(PROP_SETTING.quant_hypothesis_gen)(scen)
        logger.log_object(self.hypothesis_gen, tag="quant hypothesis generator")

        # 初始化因子和模型的假设到实验转换器
        self.factor_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.factor_hypothesis2experiment
        )()
        logger.log_object(self.factor_hypothesis2experiment, tag="factor hypothesis2experiment")
        
        self.model_hypothesis2experiment: Hypothesis2Experiment = import_class(
            PROP_SETTING.model_hypothesis2experiment
        )()
        logger.log_object(self.model_hypothesis2experiment, tag="model hypothesis2experiment")

        # 初始化因子和模型的开发者（编码器）
        self.factor_coder: Developer = import_class(PROP_SETTING.factor_coder)(scen)
        logger.log_object(self.factor_coder, tag="factor coder")
        self.model_coder: Developer = import_class(PROP_SETTING.model_coder)(scen)
        logger.log_object(self.model_coder, tag="model coder")

        # 初始化因子和模型的运行器
        self.factor_runner: Developer = import_class(PROP_SETTING.factor_runner)(scen)
        logger.log_object(self.factor_runner, tag="factor runner")
        self.model_runner: Developer = import_class(PROP_SETTING.model_runner)(scen)
        logger.log_object(self.model_runner, tag="model runner")

        # 初始化因子和模型的总结器（实验到反馈的转换器）
        self.factor_summarizer: Experiment2Feedback = import_class(PROP_SETTING.factor_summarizer)(scen)
        logger.log_object(self.factor_summarizer, tag="factor summarizer")
        self.model_summarizer: Experiment2Feedback = import_class(PROP_SETTING.model_summarizer)(scen)
        logger.log_object(self.model_summarizer, tag="model summarizer")

        # 创建量化跟踪对象
        self.trace = QuantTrace(scen=scen)
        super(RDLoop, self).__init__()

    async def direct_exp_gen(self, prev_out: dict[str, Any]):
        """
        直接实验生成方法
        
        Args:
            prev_out: 前一个输出结果字典
            
        Returns:
            包含假设和实验生成结果的字典
        """
        while True:
            # 检查未完成的循环数量是否小于最大并行数
            if self.get_unfinished_loop_cnt(self.loop_idx) < RD_AGENT_SETTINGS.get_max_parallel():
                # 提出假设
                hypo = self._propose()
                # 确保假设操作是因子或模型之一
                assert hypo.action in ["factor", "model"]
                # 根据假设操作类型转换为实验
                if hypo.action == "factor":
                    exp = self.factor_hypothesis2experiment.convert(hypo, self.trace)
                else:
                    exp = self.model_hypothesis2experiment.convert(hypo, self.trace)
                logger.log_object(exp.sub_tasks, tag="experiment generation")
                return {"propose": hypo, "exp_gen": exp}
            # 等待1秒后继续检查
            await asyncio.sleep(1)

    def coding(self, prev_out: dict[str, Any]):
        """
        编码阶段：根据实验生成结果进行代码开发
        
        Args:
            prev_out: 前一个输出结果字典
            
        Returns:
            开发完成的实验对象
        """
        # 根据假设操作类型调用相应的开发者进行代码开发
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        logger.log_object(exp, tag="coder result")
        return exp

    def running(self, prev_out: dict[str, Any]):
        """
        运行阶段：执行开发完成的代码
        
        Args:
            prev_out: 前一个输出结果字典
            
        Returns:
            运行完成的实验对象
            
        Raises:
            FactorEmptyError: 当因子提取失败时抛出
        """
        # 根据假设操作类型调用相应的运行器执行代码
        if prev_out["direct_exp_gen"]["propose"].action == "factor":
            exp = self.factor_runner.develop(prev_out["coding"])
            # 如果因子提取失败，记录错误并抛出异常
            if exp is None:
                logger.error(f"因子提取失败。")
                raise FactorEmptyError("因子提取失败。")
        elif prev_out["direct_exp_gen"]["propose"].action == "model":
            exp = self.model_runner.develop(prev_out["coding"])
        logger.log_object(exp, tag="runner result")
        return exp

    def feedback(self, prev_out: dict[str, Any]):
        """
        反馈阶段：根据运行结果生成反馈
        
        Args:
            prev_out: 前一个输出结果字典
        """
        # 获取异常信息（如果存在）
        e = prev_out.get(self.EXCEPTION_KEY, None)
        if e is not None:
            # 如果存在异常，创建反馈对象
            feedback = HypothesisFeedback(
                observations=str(e),           # 观察结果（异常信息）
                hypothesis_evaluation="",      # 假设评估（空）
                new_hypothesis="",             # 新假设（空）
                reason="",                     # 原因（空）
                decision=False,                # 决策（失败）
            )
            logger.log_object(feedback, tag="feedback")
            # 将实验和反馈添加到历史记录中
            self.trace.hist.append((prev_out["direct_exp_gen"]["exp_gen"], feedback))
        else:
            # 如果没有异常，根据假设操作类型生成反馈
            if prev_out["direct_exp_gen"]["propose"].action == "factor":
                feedback = self.factor_summarizer.generate_feedback(prev_out["running"], self.trace)
            elif prev_out["direct_exp_gen"]["propose"].action == "model":
                feedback = self.model_summarizer.generate_feedback(prev_out["running"], self.trace)
            logger.log_object(feedback, tag="feedback")
            # 将运行结果和反馈添加到历史记录中
            self.trace.hist.append((prev_out["running"], feedback))


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
):
    """
    量化金融因子的自动研发进化循环主函数
    
    参数说明:
    - path: 会话路径，如果为None则创建新的量化循环，否则从指定路径加载
    - step_n: 步骤数量，控制执行的步骤数
    - loop_n: 循环数量，控制执行的循环次数
    - all_duration: 总执行时长，控制整个过程的执行时间
    - checkout: 是否检出，控制是否从指定路径检出会话
    
    可以通过以下方式继续运行会话:
    .. code-block:: python
        dotenv run -- python rdagent/app/qlib_rd_loop/quant.py $LOG_PATH/__session__/1/0_propose  --step_n 1   # `step_n` 是可选参数
    """
    if path is None:
        # 如果路径为空，创建新的量化循环实例
        quant_loop = QuantRDLoop(QUANT_PROP_SETTING)
    else:
        # 如果路径不为空，从指定路径加载量化循环实例
        quant_loop = QuantRDLoop.load(path, checkout=checkout)

    # 异步运行量化循环
    asyncio.run(quant_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    # 使用fire库将main函数暴露为命令行接口
    fire.Fire(main)