from pathlib import Path

import pandas as pd
from pandarallel import pandarallel

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle

pandarallel.initialize(verbose=1)

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

DIRNAME = Path(__file__).absolute().resolve().parent
DIRNAME_local = Path.cwd()

# class QlibFactorExpWorkspace:

#     def prepare():
#         # create a folder;
#         # copy template
#         # place data inside the folder `combined_factors`
#         #
#     def execute():
#         de = DockerEnv()
#         de.run(local_path=self.ws_path, entry="qrun conf_baseline.yaml")

# TODO: supporting multiprocessing and keep previous results


class QlibFactorRunner(CachedRunner[QlibFactorExperiment]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - price-volume data dumper
    - `data.py` + Adaptor to Factor implementation
    - results in `mlflow`
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.DataFrame:
        """
        计算新因子与SOTA因子之间的信息系数（IC）。
        
        参数:
            concat_feature (pd.DataFrame): 包含SOTA因子和新因子的合并数据框。
            SOTA_feature_column_size (int): SOTA因子的列数。
            new_feature_columns_size (int): 新因子的列数。
            
        返回:
            pd.DataFrame: 新因子与SOTA因子之间的IC值。
        """
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        """
        去除与SOTA因子高度相关的新因子，避免因子冗余。
        
        参数:
            SOTA_feature (pd.DataFrame): 已有的SOTA因子数据。
            new_feature (pd.DataFrame): 新研发的因子数据。
            
        返回:
            pd.DataFrame: 去重后的新因子数据。
        """
        # calculate the IC between each column of SOTA_feature and new_feature
        # if the IC is larger than a threshold, remove the new_feature column
        # return the new_feature

        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        return new_feature.iloc[:, IC_max[IC_max < 0.99].index]

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        """
        Generate the experiment by processing and combining factor data,
        then passing the combined data to Docker for backtest results.
        """
        # 如果存在基础实验且最后一个基础实验的结果为空，则递归执行基础实验
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            logger.info(f"Baseline experiment execution ...")
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        # 如果存在基础实验，则进行因子合并和回测
        if exp.based_experiments:
            SOTA_factor = None
            # 筛选出所有QlibFactorExperiment类型的基础实验
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            # 如果有多个基础实验，则处理SOTA因子数据
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            logger.info(f"New factor processing ...")
            # 处理新因子数据
            new_factors = process_factor_data(exp)

            # 检查新因子是否为空
            if new_factors.empty:
                raise FactorEmptyError("Factors failed to run on the full sample, this round of experiment failed.")

            # 如果SOTA因子存在且非空，则进行去重并合并因子
            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                # 检查去重后的新因子是否为空
                if new_factors.empty:
                    raise FactorEmptyError(
                        "The factors generated in this round are highly similar to the previous factors. Please change the direction for creating new factors."
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                # 如果没有SOTA因子，则直接使用新因子
                combined_factors = new_factors

            # 对合并后的因子进行排序和去重
            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            num_features = RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns)
            logger.info(f"Factor data processing completed.")

            # 由于rdagent和qlib docker镜像中numpy版本的差异，
            # `combined_factors_df.pkl`文件在qlib docker中无法正确加载，
            # 因此我们将`combined_factors_df`的文件类型从pkl改为parquet。
            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

            # 将合并后的因子保存到工作区
            combined_factors.to_parquet(target_path, engine="pyarrow")

            # 检查是否存在历史模型实验
            exist_sota_model_exp = False
            for base_exp in reversed(exp.based_experiments):
                if isinstance(base_exp, QlibModelExperiment):
                    sota_model_exp = base_exp
                    exist_sota_model_exp = True
                    break
            logger.info(f"Experiment execution ...")
            # 如果存在历史模型实验，则使用该模型进行回测
            if exist_sota_model_exp:
                exp.experiment_workspace.inject_files(
                    **{"model.py": sota_model_exp.sub_workspace_list[0].file_dict["model.py"]}
                )
                env_to_use = {"PYTHONPATH": "./"}
                sota_training_hyperparameters = sota_model_exp.sub_tasks[0].training_hyperparameters
                # 如果存在训练超参数，则更新环境变量
                if sota_training_hyperparameters:
                    env_to_use.update(
                        {
                            "n_epochs": str(sota_training_hyperparameters.get("n_epochs", "100")),
                            "lr": str(sota_training_hyperparameters.get("lr", "2e-4")),
                            "early_stop": str(sota_training_hyperparameters.get("early_stop", 10)),
                            "batch_size": str(sota_training_hyperparameters.get("batch_size", 256)),
                            "weight_decay": str(sota_training_hyperparameters.get("weight_decay", 0.0001)),
                        }
                    )
                sota_model_type = sota_model_exp.sub_tasks[0].model_type
                # 根据模型类型设置相应的环境变量
                if sota_model_type == "TimeSeries":
                    env_to_use.update(
                        {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                    )
                elif sota_model_type == "Tabular":
                    env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})

                # 使用模型和合并因子进行回测
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_combined_factors_sota_model.yaml", run_env=env_to_use
                )
            else:
                # 使用LGBM模型和合并因子进行回测
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name=(
                        f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors.yaml"
                    )
                )
        else:
            # 如果没有基础实验，则直接执行实验
            logger.info(f"Experiment execution ...")
            result, stdout = exp.experiment_workspace.execute(
                qlib_config_name=(
                    f"conf_baseline.yaml" if len(exp.based_experiments) == 0 else "conf_combined_factors.yaml"
                )
            )

        # 检查回测结果是否为空
        if result is None:
            logger.error(f"Failed to run this experiment, because {stdout}")
            raise FactorEmptyError(f"Failed to run this experiment, because {stdout}")

        # 存储回测结果
        exp.result = result
        exp.stdout = stdout

        return exp