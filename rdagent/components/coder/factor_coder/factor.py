from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash


class FactorTask(CoSTEERTask):
    """因子任务类，用于定义和描述一个因子的相关信息"""
    
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化因子任务
        
        参数:
            factor_name (str): 因子名称
            factor_description (str): 因子描述信息
            factor_formulation (str): 因子计算公式
            *args: 可变位置参数，传递给父类
            variables (dict): 因子使用的变量字典，默认为空字典
            resource (str): 因子所需资源信息，默认为None
            factor_implementation (bool): 是否已实现因子，默认为False
            **kwargs: 可变关键字参数，传递给父类
        """
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """获取因子描述信息，为了保持兼容性而保留的方法"""
        return self.description

    def get_task_information(self):
        """
        获取任务完整信息字符串
        
        返回:
            str: 包含因子名称、描述、公式和变量信息的格式化字符串
        """
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_brief_information(self):
        """
        获取任务简要信息字符串
        
        返回:
            str: 包含因子名称、描述、公式和变量信息的格式化字符串
        """
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_information_and_implementation_result(self):
        """
        获取任务信息和实现结果的字典形式
        
        返回:
            dict: 包含因子各项信息的字典，所有值都转换为字符串形式
        """
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict_data: dict):
        """
        从字典创建FactorTask对象的静态方法
        
        参数:
            dict_data (dict): 包含因子任务信息的字典
            
        返回:
            FactorTask: 创建的因子任务对象
        """
        return FactorTask(**dict_data)

    def __repr__(self) -> str:
        """
        返回因子任务的字符串表示
        
        返回:
            str: 格式化的因子任务表示字符串
        """
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    因子文件基础工作空间类，通过编写代码到文件来实现因子。
    输入数据和输出因子值也写入到文件中。
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化因子工作空间
        
        参数:
            *args: 可变位置参数，传递给父类
            raise_exception (bool): 是否抛出异常，默认为False
            **kwargs: 可变关键字参数，传递给父类
        """
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        """
        计算工作空间的哈希值，用于缓存机制
        
        参数:
            data_type (str): 数据类型标识，默认为"Debug"
            
        返回:
            str: 哈希值字符串，如果代码未设置则返回None
        """
        return (
            md5_hash(data_type + self.file_dict["factor.py"])
            if ("factor.py" in self.file_dict and not self.raise_exception)
            else None
        )

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        执行因子实现并获取因子值，具体步骤如下：
        1. 在工作空间路径下创建工作目录
        2. 将代码写入工作空间路径下的文件
        3. 将所有源数据链接到工作空间路径文件夹
        如果call_factor_py为True：
            4. 执行代码
        否则：
            4. 从模板生成脚本以导入factor.py并转储获取因子值到result.h5
        5. 从工作空间路径文件夹中的输出文件读取因子值
        
        参数:
            data_type (str): 数据类型标识，默认为"Debug"
            
        返回:
            Tuple[str, pd.DataFrame]: 包含执行反馈信息和因子值DataFrame的元组
            
        缓存机制说明:
        1. 我们会存储函数的返回值以确保其按预期行为工作。
        - 缓存的信息将包括一个元组：(execution_feedback, executed_factor_value_dataframe, Optional[Exception])
        """
        self.before_execute()
        
        # 检查代码是否已设置
        if self.file_dict is None or "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return self.FB_CODE_NOT_SET, None
                
        with FileLock(self.workspace_path / "execution.lock"):
            # 根据任务版本确定源数据路径
            if self.target_task.version == 1:
                source_data_path = (
                    Path(
                        FACTOR_COSTEER_SETTINGS.data_folder_debug,
                    )
                    if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                    else Path(
                        FACTOR_COSTEER_SETTINGS.data_folder,
                    )
                )
            elif self.target_task.version == 2:
                # TODO you can change the name of the data folder for a better understanding
                source_data_path = Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / KAGGLE_IMPLEMENT_SETTING.competition

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = self.workspace_path / f"factor.py"

            self.link_all_files_in_folder_to_workspace(source_data_path, self.workspace_path)

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None

            # 根据任务版本确定执行代码路径
            if self.target_task.version == 1:
                execution_code_path = code_path
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            # 执行代码并捕获可能的异常
            try:
                # 使用subprocess执行Python代码
                # 命令格式：{配置的Python解释器路径} {执行代码路径}
                # cwd参数设置工作目录为当前工作空间路径
                # stderr=subprocess.STDOUT将错误输出重定向到标准输出
                # timeout参数设置执行超时时间，防止代码执行时间过长
                subprocess.check_output(
                    f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                )
                # 如果执行成功，设置执行成功标志为True
                execution_success = True
            # 捕获代码执行过程中出现的错误
            except subprocess.CalledProcessError as e:
                import site

                # 获取错误输出信息并进行路径替换处理，保护系统路径信息
                execution_feedback = (
                    e.output.decode()
                    .replace(str(execution_code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                # 如果错误信息过长，进行截取处理
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                # 根据是否设置抛出异常标志来决定是抛出异常还是记录异常
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)
            # 捕获代码执行超时的错误
            except subprocess.TimeoutExpired:
                # 记录超时错误信息，包含设置的超时时间
                execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                # 根据是否设置抛出异常标志来决定是抛出异常还是记录异常
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)

            # 读取输出文件中的因子值
            workspace_output_file_path = self.workspace_path / "result.h5"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        """
        返回工作空间的字符串表示
        
        返回:
            str: 格式化的工作空间表示字符串
        """
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        """
        返回工作空间的字符串表示
        
        返回:
            str: 格式化的工作空间表示字符串
        """
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        """
        从文件夹创建FactorFBWorkspace对象的静态方法
        
        参数:
            task (FactorTask): 目标因子任务对象
            path (Union[str, Path]): 文件夹路径
            **kwargs: 其他关键字参数
            
        返回:
            FactorFBWorkspace: 创建的因子工作空间对象
        """
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment