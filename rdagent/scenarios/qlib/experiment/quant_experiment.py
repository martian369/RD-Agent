from copy import deepcopy
from pathlib import Path

# 因子相关导入
from rdagent.components.coder.factor_coder.config import get_factor_env
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)

# 模型相关导入
from rdagent.components.coder.model_coder.conf import get_model_env
from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.agent.tpl import T


class QlibFactorExperiment(FactorExperiment[FactorTask, QlibFBWorkspace, FactorFBWorkspace]):
    """
    Qlib因子实验类
    
    该类用于在Qlib框架下进行因子实验，继承自FactorExperiment基类。
    它使用QlibFBWorkspace作为工作空间，并加载因子模板。
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        初始化Qlib因子实验
        
        Parameters:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        # 设置实验工作空间，使用因子模板文件夹
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")


class QlibModelExperiment(ModelExperiment[ModelTask, QlibFBWorkspace, ModelFBWorkspace]):
    """
    Qlib模型实验类
    
    该类用于在Qlib框架下进行模型实验，继承自ModelExperiment基类。
    它使用QlibFBWorkspace作为工作空间，并加载模型模板。
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        初始化Qlib模型实验
        
        Parameters:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        # 设置实验工作空间，使用模型模板文件夹
        self.experiment_workspace = QlibFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class QlibQuantScenario(Scenario):
    """
    Qlib量化场景类
    
    该类定义了Qlib量化场景的背景、数据源、接口规范等信息。
    它为因子和模型实验提供统一的场景描述。
    """
    def __init__(self) -> None:
        """
        初始化Qlib量化场景
        """
        super().__init__()
        # 深拷贝数据文件夹介绍
        self._source_data = deepcopy(get_data_folder_intro())

        # 加载富样式描述和实验设置
        self._rich_style_description = deepcopy(T(".prompts:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(".prompts:qlib_factor_experiment_setting").r())

    def background(self, tag=None) -> str:
        """
        获取场景背景信息
        
        Parameters:
            tag: 标签，可以是None、"factor"或"model"
            
        Returns:
            str: 场景背景描述
        """
        assert tag in [None, "factor", "model"]
        quant_background = "The background of the scenario is as follows:\n" + T(".prompts:qlib_quant_background").r(
            runtime_environment=self.get_runtime_environment(),
        )
        factor_background = "This time, I need your help with the research and development of the factor. The background of the factor scenario is as follows:\n" + T(
            ".prompts:qlib_factor_background"
        ).r(
            runtime_environment=self.get_runtime_environment(tag="factor"),
        )
        model_background = "This time, I need your help with the research and development of the model. The background of the model scenario is as follows:\n" + T(
            ".prompts:qlib_model_background"
        ).r(
            runtime_environment=self.get_runtime_environment(tag="model"),
        )

        # TODO: 这里还有一些问题
        if tag is None:
            return quant_background + "\n" + factor_background + "\n" + model_background
        elif tag == "factor":
            return factor_background
        else:
            return model_background

    def get_source_data_desc(self) -> str:
        """
        获取源数据描述
        
        Returns:
            str: 源数据描述信息
        """
        return self._source_data

    def output_format(self, tag=None) -> str:
        """
        获取输出格式规范
        
        Parameters:
            tag: 标签，可以是None、"factor"或"model"
            
        Returns:
            str: 输出格式描述
        """
        assert tag in [None, "factor", "model"]
        # 因子输出格式
        factor_output_format = (
            "The factor code should output the following format:\n" + T(".prompts:qlib_factor_output_format").r()
        )
        # 模型输出格式
        model_output_format = (
            "The model code should output the following format:\n" + T(".prompts:qlib_model_output_format").r()
        )

        if tag is None:
            return factor_output_format + "\n" + model_output_format
        elif tag == "factor":
            return factor_output_format
        else:
            return model_output_format

    def interface(self, tag=None) -> str:
        """
        获取接口规范
        
        Parameters:
            tag: 标签，可以是None、"factor"或"model"
            
        Returns:
            str: 接口规范描述
        """
        assert tag in [None, "factor", "model"]
        # 因子接口规范
        factor_interface = (
            "The factor code should be written in the following interface:\n" + T(".prompts:qlib_factor_interface").r()
        )
        # 模型接口规范
        model_interface = (
            "The model code should be written in the following interface:\n" + T(".prompts:qlib_model_interface").r()
        )

        if tag is None:
            return factor_interface + "\n" + model_interface
        elif tag == "factor":
            return factor_interface
        else:
            return model_interface

    def simulator(self, tag=None) -> str:
        """
        获取模拟器信息
        
        Parameters:
            tag: 标签，可以是None、"factor"或"model"
            
        Returns:
            str: 模拟器描述
        """
        assert tag in [None, "factor", "model"]
        factor_simulator = "The factor code will be sent to the simulator:\n" + T(".prompts:qlib_factor_simulator").r()
        model_simulator = "The model code will be sent to the simulator:\n" + T(".prompts:qlib_model_simulator").r()

        if tag is None:
            return factor_simulator + "\n" + model_simulator
        elif tag == "factor":
            return factor_simulator
        else:
            return model_simulator

    @property
    def rich_style_description(self) -> str:
        """
        获取富样式描述
        
        Returns:
            str: 富样式描述信息
        """
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        """
        获取实验设置
        
        Returns:
            str: 实验设置信息
        """
        return self._experiment_setting

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
        action: str | None = None,
    ) -> str:
        """
        获取场景完整描述
        
        Parameters:
            task: 任务对象
            filtered_tag: 过滤标签
            simple_background: 是否使用简单背景
            action: 动作类型
            
        Returns:
            str: 场景完整描述
        """
        def common_description(action: str | None = None) -> str:
            """
            获取通用描述
            
            Parameters:
                action: 动作类型
                
            Returns:
                str: 通用描述
            """
            return f"""\n------Background of the scenario------
{self.background(action)}
------The source dataset you can use------
{self.get_source_data_desc()}
"""

        # TODO: 处理源数据这里还有一些问题
        def source_data() -> str:
            """
            获取源数据描述
            
            Returns:
                str: 源数据描述
            """
            return f"""
------The source data you can use------
{self.get_source_data_desc()}
"""

        def interface(tag: str | None) -> str:
            """
            获取接口描述
            
            Parameters:
                tag: 标签
                
            Returns:
                str: 接口描述
            """
            return f"""
------The interface you should follow to write the runnable code------
{self.interface(tag)}
"""

        def output(tag: str | None) -> str:
            """
            获取输出格式描述
            
            Parameters:
                tag: 标签
                
            Returns:
                str: 输出格式描述
            """
            return f"""
------The output of your code should be in the format------
{self.output_format(tag)}
"""

        def simulator(tag: str | None) -> str:
            """
            获取模拟器描述
            
            Parameters:
                tag: 标签
                
            Returns:
                str: 模拟器描述
            """
            return f"""
------The simulator user can use to test your solution------
{self.simulator(tag)}
"""

        if simple_background:
            return common_description()
        elif filtered_tag == "hypothesis_and_experiment" or filtered_tag == "feedback":
            return common_description() + simulator(None)
        elif filtered_tag == "factor" or filtered_tag == "feature" or filtered_tag == "factors":
            return common_description("factor") + interface("factor") + output("factor") + simulator("factor")
        elif filtered_tag == "model" or filtered_tag == "model tuning":
            return common_description("model") + interface("model") + output("model") + simulator("model")
        elif action == "factor" or action == "model":
            return common_description(action) + interface(action) + output(action) + simulator(action)

    def get_runtime_environment(self, tag: str = None) -> str:
        """
        获取运行时环境信息
        
        Parameters:
            tag: 标签，可以是None、"factor"或"model"
            
        Returns:
            str: 运行时环境信息
        """
        assert tag in [None, "factor", "model"]

        if tag is None or tag == "factor":
            # 使用因子环境获取运行时环境
            # factor_env = get_factor_env()
            factor_stdout = get_runtime_environment_by_env()
            if tag == "factor":
                stdout = factor_stdout

        if tag is None or tag == "model":
            # 使用模型环境获取运行时环境
            # model_env = get_model_env()
            model_stdout = get_runtime_environment_by_env()
            if tag == "model":
                stdout = model_stdout

        if tag is None:
            # 组合两个环境的输出
            stdout = (
                "=== [生成因子的环境] ===\n"
                + factor_stdout.strip()
                + "\n\n=== [训练模型的环境] ===\n"
                + model_stdout.strip()
            )

        return stdout