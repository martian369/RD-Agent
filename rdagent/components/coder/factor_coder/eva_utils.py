import io
import json
from abc import abstractmethod
from typing import Dict, Tuple

import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.experiment import Task, Workspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.log import rdagent_logger as logger



class FactorEvaluator:
    """因子评估器基类
    虽然初始化方法与Evaluator相同，但我们想强调它们是不同的
    """

    def __init__(self, scen=None) -> None:
        """初始化因子评估器
        
        Parameters
        ----------
        scen : 场景对象，可选
            用于获取场景描述信息
        """
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        **kwargs,
    ) -> Tuple[str, object]:
        """评估因子实现的核心方法（抽象方法，需子类实现）
        
        可以通过以下方式获取数据框:
        .. code-block:: python
            _, gen_df = implementation.execute()
            _, gt_df = gt_implementation.execute()

        Parameters
        ----------
        target_task : Task
            目标任务对象
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象
        **kwargs : dict
            其他关键字参数

        Returns
        -------
        Tuple[str, object]
            - str: 评估结果的文本描述
            - object: 可比较的指标（布尔值、整数、浮点数等），仅文本结果的评估器返回None
        """
        raise NotImplementedError("Please implement the `evaluator` method")

    def _get_df(self, gt_implementation: Workspace, implementation: Workspace):
        """获取真实值和生成值的数据框
        
        Parameters
        ----------
        gt_implementation : Workspace
            真实值实现的工作区对象
        implementation : Workspace
            实现的工作区对象

        Returns
        -------
        tuple
            (真实值数据框, 生成值数据框)
        """
        # 获取真实值数据框
        if gt_implementation is not None:
            _, gt_df = gt_implementation.execute()
            if isinstance(gt_df, pd.Series):
                gt_df = gt_df.to_frame("gt_factor")
            if isinstance(gt_df, pd.DataFrame):
                gt_df = gt_df.sort_index()
        else:
            gt_df = None

        # 获取生成值数据框
        _, gen_df = implementation.execute()
        if isinstance(gen_df, pd.Series):
            gen_df = gen_df.to_frame("source_factor")
        if isinstance(gen_df, pd.DataFrame):
            gen_df = gen_df.sort_index()
        return gt_df, gen_df

    def __str__(self) -> str:
        """返回类名字符串表示
        
        Returns
        -------
        str
            类名
        """
        return self.__class__.__name__


class FactorCodeEvaluator(FactorEvaluator):
    """因子代码评估器
    用于评估因子实现代码的质量
    """
    
    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        execution_feedback: str,
        value_feedback: str = "",
        gt_implementation: Workspace = None,
        **kwargs,
    ):
        """评估因子实现代码
        
        Parameters
        ----------
        target_task : FactorTask
            目标因子任务
        implementation : Workspace
            实现的工作区对象
        execution_feedback : str
            执行反馈信息
        value_feedback : str, 可选
            值反馈信息，默认为空字符串
        gt_implementation : Workspace, 可选
            真实值实现的工作区对象，默认为None
        **kwargs : dict
            其他关键字参数

        Returns
        -------
        tuple
            (代码评估反馈, None)
        """
        # 获取因子信息和代码
        factor_information = target_task.get_task_information()
        code = implementation.all_codes

        # 构建系统提示词
        system_prompt = T(".prompts:evaluator_code_feedback_v1_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(
                    target_task,
                    filtered_tag="feature",
                    simple_background=FACTOR_COSTEER_SETTINGS.simple_background,
                )
                if self.scen is not None
                else "No scenario description."
            )
        )

        # 处理执行反馈内容长度，确保不超过token限制
        execution_feedback_to_render = execution_feedback
        for _ in range(10):  # 最多尝试10次分割内容
            user_prompt = T(".prompts:evaluator_code_feedback_v1_user").r(
                factor_information=factor_information,
                code=code,
                execution_feedback=execution_feedback_to_render,
                value_feedback=value_feedback,
                gt_code=gt_implementation.code if gt_implementation else None,
            )
            # 检查token数量是否超过限制
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > APIBackend().chat_token_limit
            ):
                # 如果超过限制，将执行反馈内容减半
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break
        
        # 调用API进行代码评估
        with logger.tag(f"coding"):
            critic_response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=False,
            )

        return critic_response, None


class FactorInfEvaluator(FactorEvaluator):
    """因子无穷值评估器
    检查生成的数据框中是否包含无穷值
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现中是否包含无穷值
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (评估反馈信息, 是否包含无穷值的布尔值)
        """
        # 获取数据框
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        
        # 统计无穷值数量
        INF_count = gen_df.isin([float("inf"), -float("inf")]).sum().sum()
        if INF_count == 0:
            return "The source dataframe does not have any infinite values.", True
        else:
            return (
                f"The source dataframe has {INF_count} infinite values. Please check the implementation.",
                False,
            )


class FactorSingleColumnEvaluator(FactorEvaluator):
    """因子单列评估器
    检查生成的数据框是否只有一列
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的数据框是否只有一列
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (评估反馈信息, 是否只有一列的布尔值)
        """
        # 获取数据框
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        
        # 检查列数
        if len(gen_df.columns) == 1:
            return "The source dataframe has only one column which is correct.", True
        else:
            return (
                "The source dataframe has more than one column. Please check the implementation. We only evaluate the first column.",
                False,
            )


class FactorOutputFormatEvaluator(FactorEvaluator):
    """因子输出格式评估器
    评估生成数据框的输出格式是否符合要求
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的输出格式
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (格式评估反馈信息, 格式是否正确的布尔值)
        """
        # 获取数据框
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Skip the evaluation of the output format.",
                False,
            )
        
        # 获取数据框信息字符串
        buffer = io.StringIO()
        gen_df.info(buf=buffer)
        gen_df_info_str = f"The user is currently working on a feature related task.\nThe output dataframe info is:\n{buffer.getvalue()}"
        
        # 构建系统提示词
        system_prompt = T(".prompts:evaluator_output_format_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(implementation.target_task, filtered_tag="feature")
                if self.scen is not None
                else "No scenario description."
            )
        )

        # 最大尝试次数
        max_attempts = 3
        attempts = 0
        final_evaluation_dict = None

        # 循环尝试获取评估结果
        while attempts < max_attempts:
            try:
                # 创建API后端实例
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                # 调用API进行格式评估
                resp = api.build_messages_and_create_chat_completion(
                    user_prompt=gen_df_info_str,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, str | bool | int],
                )
                # 解析响应
                resp_dict = json.loads(resp)
                resp_dict["output_format_decision"] = str(resp_dict["output_format_decision"]).lower() in ["true", "1"]

                return (
                    str(resp_dict["output_format_feedback"]),
                    resp_dict["output_format_decision"],
                )
            except (KeyError, json.JSONDecodeError) as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise KeyError(
                        "Wrong JSON Response or missing 'output_format_decision' or 'output_format_feedback' key after multiple attempts."
                    ) from e

        return "Failed to evaluate output format after multiple attempts.", False


class FactorDatetimeDailyEvaluator(FactorEvaluator):
    """因子日期时间日频评估器
    检查生成数据框的日期时间索引是否为日频数据
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str | object]:
        """评估因子实现的数据框日期时间索引是否为日频
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str | object]
            (评估反馈信息, 是否为日频数据的布尔值)
        """
        # 获取数据框
        _, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return "The source dataframe is None. Skip the evaluation of the datetime format.", False

        # 检查是否存在datetime索引
        if "datetime" not in gen_df.index.names:
            return "The source dataframe does not have a datetime index. Please check the implementation.", False

        # 尝试转换datetime索引
        try:
            pd.to_datetime(gen_df.index.get_level_values("datetime"))
        except Exception:
            return (
                f"The source dataframe has a datetime index but it is not in the correct format (maybe a regular string or other objects). Please check the implementation.\n The head of the output dataframe is: \n{gen_df.head()}",
                False,
            )

        # 检查时间间隔是否为日频
        time_diff = pd.to_datetime(gen_df.index.get_level_values("datetime")).to_series().diff().dropna().unique()
        if pd.Timedelta(minutes=1) in time_diff:
            return (
                "The generated dataframe is not daily. The implementation is definitely wrong. Please check the implementation.",
                False,
            )
        return "The generated dataframe is daily.", True


class FactorRowCountEvaluator(FactorEvaluator):
    """因子行数评估器
    比较生成数据框与真实值数据框的行数比例
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的数据框行数与真实值数据框行数的比例
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (行数比例评估反馈信息, 行数比例值)
        """
        # 获取数据框
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        
        # 计算行数比例
        ratio = min(len(gen_df), len(gt_df)) / max(len(gen_df), len(gt_df))
        return (
            (
                f"The ratio of rows count in the source dataframe to the ground truth dataframe is {ratio:.2f}. "
                + "Please verify the implementation. "
                if ratio <= 0.99
                else ""
            ),
            ratio,
        )


class FactorIndexEvaluator(FactorEvaluator):
    """因子索引评估器
    比较生成数据框与真实值数据框的索引相似度
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的数据框索引与真实值数据框索引的相似度
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (索引相似度评估反馈信息, 索引相似度值)
        """
        # 获取数据框
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        
        # 计算索引相似度
        gen_index_set, gt_index_set = set(gen_df.index), set(gt_df.index)
        similarity = len(gen_index_set.intersection(gt_index_set)) / len(gen_index_set.union(gt_index_set))
        return (
            (
                f"The source dataframe and the ground truth dataframe have different index with a similarity of {similarity:.2%}. The similarity is calculated by the number of shared indices divided by the union indices. "
                + "Please check the implementation."
                if similarity <= 0.99
                else ""
            ),
            similarity,
        )


class FactorMissingValuesEvaluator(FactorEvaluator):
    """因子缺失值评估器
    比较生成数据框与真实值数据框的缺失值数量
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的数据框缺失值与真实值数据框缺失值是否一致
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (缺失值评估反馈信息, 是否一致的布尔值)
        """
        # 获取数据框
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        
        # 比较缺失值数量
        if gen_df.isna().sum().sum() == gt_df.isna().sum().sum():
            return "Both dataframes have the same missing values.", True
        else:
            return (
                f"The dataframes do not have the same missing values. The source dataframe has {gen_df.isna().sum().sum()} missing values, while the ground truth dataframe has {gt_df.isna().sum().sum()} missing values. Please check the implementation.",
                False,
            )


class FactorEqualValueRatioEvaluator(FactorEvaluator):
    """因子等值比例评估器
    计算生成数据框与真实值数据框中相等值的比例
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的数据框与真实值数据框中相等值的比例
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (等值比例评估反馈信息, 等值比例值)
        """
        # 获取数据框
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                -1,
            )
        
        # 计算等值比例
        try:
            close_values = gen_df.sub(gt_df).abs().lt(1e-6)
            result_int = close_values.astype(int)
            pos_num = result_int.sum().sum()
            acc_rate = pos_num / close_values.size
        except:
            close_values = gen_df
        if close_values.all().iloc[0]:
            return (
                "All values in the dataframes are equal within the tolerance of 1e-6.",
                acc_rate,
            )
        else:
            return (
                "Some values differ by more than the tolerance of 1e-6. Check for rounding errors or differences in the calculation methods.",
                acc_rate,
            )


class FactorCorrelationEvaluator(FactorEvaluator):
    """因子相关性评估器
    计算生成数据框与真实值数据框的相关性指标
    """
    
    def __init__(self, hard_check: bool, *args, **kwargs) -> None:
        """初始化相关性评估器
        
        Parameters
        ----------
        hard_check : bool
            是否进行严格检查
        *args : tuple
            其他位置参数
        **kwargs : dict
            其他关键字参数
        """
        super().__init__(*args, **kwargs)
        self.hard_check = hard_check

    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
    ) -> Tuple[str, object]:
        """评估因子实现的数据框与真实值数据框的相关性
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象

        Returns
        -------
        Tuple[str, object]
            (相关性评估反馈信息, 相关性值或是否高度相关的布尔值)
        """
        # 获取数据框
        gt_df, gen_df = self._get_df(gt_implementation, implementation)
        if gen_df is None:
            return (
                "The source dataframe is None. Please check the implementation.",
                False,
            )
        
        # 合并数据框并计算相关性
        concat_df = pd.concat([gen_df, gt_df], axis=1)
        concat_df.columns = ["source", "gt"]
        ic = concat_df.groupby("datetime").apply(lambda df: df["source"].corr(df["gt"])).dropna().mean()
        ric = (
            concat_df.groupby("datetime")
            .apply(lambda df: df["source"].corr(df["gt"], method="spearman"))
            .dropna()
            .mean()
        )

        # 根据是否严格检查返回相应结果
        if self.hard_check:
            if ic > 0.99 and ric > 0.99:
                return (
                    f"The dataframes are highly correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}.",
                    True,
                )
            else:
                return (
                    f"The dataframes are not sufficiently high correlated. The ic is {ic:.6f} and the rankic is {ric:.6f}. Investigate the factors that might be causing the discrepancies and ensure that the logic of the factor calculation is consistent.",
                    False,
                )
        else:
            return f"The ic is ({ic:.6f}) and the rankic is ({ric:.6f}).", ic


class FactorValueEvaluator(FactorEvaluator):
    """因子值评估器
    综合评估因子实现的值质量，包含多个子评估器
    """
    
    def evaluate(
        self,
        implementation: Workspace,
        gt_implementation: Workspace,
        version: int = 1,  # 1 for qlib factors and 2 for kaggle factors
        **kwargs,
    ) -> Tuple:
        """综合评估因子实现的值质量
        
        Parameters
        ----------
        implementation : Workspace
            实现的工作区对象
        gt_implementation : Workspace
            真实值实现的工作区对象
        version : int, 可选
            版本号，1表示qlib因子，2表示kaggle因子，默认为1
        **kwargs : dict
            其他关键字参数

        Returns
        -------
        Tuple
            (综合评估反馈信息, 值检查决策结果)
        """
        # 存储各个评估器的结论
        conclusions = []

        # 初始化结果变量
        row_result = 0
        index_result = 0
        output_format_result = None
        equal_value_ratio_result = 0
        high_correlation_result = False
        row_result = None

        # 检查数据框是否只有一列（仅适用于qlib因子）
        if version == 1:
            feedback_str, _ = FactorSingleColumnEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)
        elif version == 2:
            input_shape = self.scen.input_shape
            _, gen_df = self._get_df(gt_implementation, implementation)
            if gen_df.shape[-1] > input_shape[-1]:
                conclusions.append(
                    "Output dataframe has more columns than input feature which is not acceptable in feature processing tasks. Please check the implementation to avoid generating too many columns. Consider this implementation as a failure."
                )

        # 检查无穷值
        feedback_str, inf_evaluate_res = FactorInfEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)

        # 检查数据框索引格式
        feedback_str, _ = FactorOutputFormatEvaluator(self.scen).evaluate(implementation, gt_implementation)
        conclusions.append(feedback_str)
        
        # 检查日期时间是否为日频（仅适用于qlib因子）
        if version == 1:
            feedback_str, daily_check_result = FactorDatetimeDailyEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)
        else:
            daily_check_result = None

        # 如果提供了真实值实现，则进行详细检查
        if gt_implementation is not None:
            # 检查行数比例
            feedback_str, row_result = FactorRowCountEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            # 检查索引相似度
            feedback_str, index_result = FactorIndexEvaluator(self.scen).evaluate(implementation, gt_implementation)
            conclusions.append(feedback_str)

            # 检查缺失值
            feedback_str, output_format_result = FactorMissingValuesEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            # 检查等值比例
            feedback_str, equal_value_ratio_result = FactorEqualValueRatioEvaluator(self.scen).evaluate(
                implementation, gt_implementation
            )
            conclusions.append(feedback_str)

            # 检查相关性（仅当索引相似度足够高时）
            if index_result > 0.99:
                feedback_str, high_correlation_result = FactorCorrelationEvaluator(
                    hard_check=True, scen=self.scen
                ).evaluate(implementation, gt_implementation)
            else:
                high_correlation_result = False
                feedback_str = "The source dataframe and the ground truth dataframe have different index. Give up comparing the values and correlation because it's useless"
            conclusions.append(feedback_str)

        # 合并所有评估结论
        conclusion_str = "\n".join(conclusions)

        # 根据评估结果做出决策
        if gt_implementation is not None and (equal_value_ratio_result > 0.99) or high_correlation_result:
            decision_from_value_check = True
        elif (
            row_result is not None
            and row_result <= 0.99
            or output_format_result is False
            or daily_check_result is False
            or inf_evaluate_res is False
        ):
            decision_from_value_check = False
        else:
            decision_from_value_check = None
        return conclusion_str, decision_from_value_check


class FactorFinalDecisionEvaluator(FactorEvaluator):
    """因子最终决策评估器
    基于执行反馈、代码反馈和值反馈做出最终决策
    """
    
    def evaluate(
        self,
        target_task: FactorTask,
        execution_feedback: str,
        value_feedback: str,
        code_feedback: str,
        **kwargs,
    ) -> Tuple:
        """基于多个反馈信息做出因子实现的最终决策
        
        Parameters
        ----------
        target_task : FactorTask
            目标因子任务
        execution_feedback : str
            执行反馈信息
        value_feedback : str
            值反馈信息
        code_feedback : str
            代码反馈信息
        **kwargs : dict
            其他关键字参数

        Returns
        -------
        Tuple
            (最终决策结果, 最终反馈信息)
        """
        # 构建系统提示词
        system_prompt = T(".prompts:evaluator_final_decision_v1_system").r(
            scenario=(
                self.scen.get_scenario_all_desc(target_task, filtered_tag="feature")
                if self.scen is not None
                else "No scenario description."
            )
        )
        
        # 处理执行反馈内容长度，确保不超过token限制
        execution_feedback_to_render = execution_feedback

        for _ in range(10):  # 最多尝试10次分割内容
            user_prompt = T(".prompts:evaluator_final_decision_v1_user").r(
                factor_information=target_task.get_task_information(),
                execution_feedback=execution_feedback_to_render,
                code_feedback=code_feedback,
                value_feedback=(
                    value_feedback
                    if value_feedback is not None
                    else "No Ground Truth Value provided, so no evaluation on value is performed."
                ),
            )
            # 检查token数量是否超过限制
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > APIBackend().chat_token_limit
            ):
                # 如果超过限制，将执行反馈内容减半
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        # 最大尝试次数
        final_evaluation_dict = None
        attempts = 0
        max_attempts = 3

        # 循环尝试获取最终评估结果
        while attempts < max_attempts:
            try:
                # 创建API后端实例
                api = APIBackend() if attempts == 0 else APIBackend(use_chat_cache=False)
                # 调用API进行最终决策评估
                final_evaluation_dict = json.loads(
                    api.build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        seed=attempts,  # 防止缓存导致的无用重试
                        json_target_type=Dict[str, str | bool | int],
                    ),
                )
                # 获取最终决策和反馈
                final_decision = final_evaluation_dict["final_decision"]
                final_feedback = final_evaluation_dict["final_feedback"]

                # 转换决策结果为布尔值
                final_decision = str(final_decision).lower() in ["true", "1"]
                return final_decision, final_feedback

            except json.JSONDecodeError as e:
                raise ValueError("Failed to decode JSON response from API.") from e
            except KeyError as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise KeyError(
                        "Response from API is missing 'final_decision' or 'final_feedback' key after multiple attempts."
                    ) from e

        return None, None