"""
因子基准测试评估脚本

该脚本用于评估因子实现的性能，通过加载测试用例并使用指定的方法进行因子开发，
然后通过一系列评估器对生成的因子进行评估。
"""

# 从qlib_rd_loop配置中导入因子属性设置
from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
# 从benchmark组件导入基准测试设置类
from rdagent.components.benchmark.conf import BenchmarkSettings
# 从benchmark组件导入因子实现评估类
from rdagent.components.benchmark.eval_method import FactorImplementEval
# 从核心模块导入场景基类
from rdagent.core.scenario import Scenario
# 从核心工具模块导入类导入函数
from rdagent.core.utils import import_class
# 从日志模块导入RD-Agent日志记录器
from rdagent.log import rdagent_logger as logger
# 从qlib场景导入因子测试用例加载器
from rdagent.scenarios.qlib.factor_experiment_loader.json_loader import (
    FactorTestCaseLoaderFromJsonFile,
)

if __name__ == "__main__":
    # 1.读取基准测试设置
    # 创建BenchmarkSettings实例，从环境变量或默认值加载配置
    # 主要配置包括：测试数据路径、测试轮数、测试用例数量、测试方法类路径等
    bs = BenchmarkSettings()

    # 2.读取并准备评估数据
    # 使用FactorTestCaseLoaderFromJsonFile从JSON文件加载测试用例
    # bench_data_path: 测试数据文件路径，包含因子描述、公式、变量和真实代码等信息
    test_cases = FactorTestCaseLoaderFromJsonFile().load(bs.bench_data_path)

    # 3.声明要测试的方法并传递参数
    # 导入并实例化场景类，用于提供因子开发所需的上下文环境
    # FACTOR_PROP_SETTING.scen: 场景类路径，如"rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    scen: Scenario = import_class(FACTOR_PROP_SETTING.scen)()
    
    # 导入并实例化因子生成方法类
    # bs.bench_method_cls: 测试方法类路径，如"rdagent.components.coder.factor_coder.FactorCoSTEER"
    # **bs.bench_method_extra_kwargs: 额外的关键字参数，用于配置测试方法
    generate_method = import_class(bs.bench_method_cls)(scen=scen, **bs.bench_method_extra_kwargs)
    
    # 4.声明评估方法并传递参数
    # 创建FactorImplementEval实例，用于执行因子实现的评估
    # method: 要测试的因子生成方法
    # test_cases: 测试用例集合
    # scen: 场景实例
    # catch_eval_except: 是否捕获评估过程中的异常，默认为True，避免评估器异常导致整个评估流程中断
    # test_round: 测试轮数，每轮都会使用相同的方法重新生成因子
    eval_method = FactorImplementEval(
        method=generate_method,
        test_cases=test_cases,
        scen=scen,
        catch_eval_except=True,
        test_round=bs.bench_test_round,
    )

    # 5.运行评估
    # 调用eval_method.develop()方法生成因子实现
    # 然后调用eval_method.eval()方法对生成的因子进行评估
    # 返回评估结果，包含每轮测试中每个因子的评估指标
    res = eval_method.eval(eval_method.develop())

    # 6.保存结果
    # 使用日志记录器记录评估结果对象，便于后续分析和可视化
    logger.log_object(res)