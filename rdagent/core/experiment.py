from __future__ import annotations

import io
import os
import platform
import re
import shutil
import typing
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Feedback

if TYPE_CHECKING:
    from rdagent.utils.env import EnvResult


if typing.TYPE_CHECKING:
    from rdagent.core.proposal import Hypothesis
    from rdagent.utils.env import Env

"""
RD-Agent 实验管理模块

这个文件包含了 RD-Agent 中所有关于任务组织的核心类。它定义了实验、任务、工作空间等关键概念，
为整个 RD-Agent 系统提供了任务执行和实验管理的基础框架。

主要功能：
1. 任务抽象和管理 (Task, AbsTask)
2. 工作空间管理 (Workspace, FBWorkspace) 
3. 实验计划和执行 (Experiment, ExperimentPlan)
4. 文件注入和代码管理
5. 工作空间检查点和恢复机制

这个模块是 RD-Agent 的核心组件，负责协调整个研发流程中的任务执行和实验管理。
"""


class AbsTask(ABC):
    """
    抽象任务基类
    
    定义了任务的基本接口，所有具体任务类型都需要继承此类。
    任务版本用于区分不同类型的任务执行方式（如 qlib 任务和 kaggle 任务）。
    """
    
    def __init__(self, name: str, version: int = 1) -> None:
        """
        初始化任务
        
        Args:
            name: 任务名称
            version: 任务版本，默认为1
                    由于 qlib 任务执行和 kaggle 任务执行方式不同，需要版本号来区分
                    TODO: 未来可能会统一这些执行方式
        """
        self.version = version
        self.name = name

    @abstractmethod
    def get_task_information(self) -> str:
        """
        获取任务信息字符串，用于构建唯一标识符
        
        Returns:
            str: 包含任务信息的字符串
        """


class UserInstructions(list[str]):
    """
    用户指令类
    
    继承自 list[str]，用于存储和管理用户提供的指令。
    这些指令具有最高优先级，会以特定格式显示。
    """
    
    def __str__(self) -> str:
        """
        将用户指令格式化为字符串
        
        Returns:
            str: 格式化后的用户指令字符串，如果为空则返回空字符串
        """
        if self:
            return ("\nUser Instructions (Top priority!):\n" + "\n".join(f"- {ui}" for ui in self)) if self else ""
        return ""


class Task(AbsTask):
    """
    具体任务实现类
    
    继承自 AbsTask，提供了任务的具体实现。
    包含任务描述和用户指令等详细信息。
    """
    
    def __init__(
        self,
        name: str,
        version: int = 1,
        description: str = "",
        user_instructions: UserInstructions | None = None,
    ) -> None:
        """
        初始化任务
        
        Args:
            name: 任务名称
            version: 任务版本，默认为1
            description: 任务描述
            user_instructions: 用户指令列表
        """
        super().__init__(name, version)
        self.description = description
        self.user_instructions = user_instructions

    def get_task_information(self) -> str:
        """
        获取任务信息字符串
        
        Returns:
            str: 包含任务名称、描述和用户指令的完整信息字符串
        """
        return f"Task Name: {self.name}\nDescription: {self.description}{self.user_instructions!s}"

    def __repr__(self) -> str:
        """
        返回任务的字符串表示
        
        Returns:
            str: 格式化的任务表示字符串
        """
        return f"<{self.__class__.__name__} {self.name}>"


# 类型变量定义
ASpecificTask = TypeVar("ASpecificTask", bound=Task)  # 特定任务类型
ASpecificFeedback = TypeVar("ASpecificFeedback", bound=Feedback)  # 特定反馈类型


@dataclass
class RunningInfo:
    """
    运行信息数据类
    
    存储实验运行过程中的结果和运行时间等关键信息。
    """
    result: object = None  # 实验结果，在不同场景下可以是不同类型
    running_time: float | None = None  # 运行时间（秒）


class Workspace(ABC, Generic[ASpecificTask, ASpecificFeedback]):
    """
    抽象工作空间基类
    
    工作空间是存储任务实现的地方，随着开发者实现任务而不断演进。
    要获取工作空间的快照，需要调用 `copy` 方法来获取工作空间的副本。
    
    这是一个泛型抽象类，支持不同类型的任务和反馈。
    """

    def __init__(self, target_task: ASpecificTask | None = None) -> None:
        """
        初始化工作空间
        
        Args:
            target_task: 目标任务，可选
        """
        self.target_task: ASpecificTask | None = target_task
        self.feedback: ASpecificFeedback | None = None
        self.running_info: RunningInfo = RunningInfo()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> object | None:
        """
        执行工作空间中的任务
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            object | None: 执行结果
            
        Raises:
            NotImplementedError: 如果子类没有实现此方法
        """
        error_message = "execute method is not implemented."
        raise NotImplementedError(error_message)

    @abstractmethod
    def copy(self) -> Workspace:
        """
        复制工作空间
        
        Returns:
            Workspace: 工作空间的副本
            
        Raises:
            NotImplementedError: 如果子类没有实现此方法
        """
        error_message = "copy method is not implemented."
        raise NotImplementedError(error_message)

    @property
    @abstractmethod
    def all_codes(self) -> str:
        """
        获取工作空间中所有代码文件的内容
        
        Returns:
            str: 所有代码文件的字符串表示
        """

    @abstractmethod
    def create_ws_ckp(self) -> None:
        """
        创建工作空间检查点
        
        当工作空间可以就地修改时，提供创建检查点和恢复的支持。
        检查点用于保存当前工作空间状态，以便后续恢复。
        """

    @abstractmethod
    def recover_ws_ckp(self) -> None:
        """
        从检查点恢复工作空间
        
        恢复由 :py:meth:`create_ws_ckp` 创建的检查点。
        """


ASpecificWS = TypeVar("ASpecificWS", bound=Workspace)  # 特定工作空间类型


class WsLoader(ABC, Generic[ASpecificTask, ASpecificWS]):
    """
    工作空间加载器抽象基类
    
    负责根据任务加载相应的工作空间实例。
    """
    
    @abstractmethod
    def load(self, task: ASpecificTask) -> ASpecificWS:
        """
        根据任务加载工作空间
        
        Args:
            task: 目标任务
            
        Returns:
            ASpecificWS: 加载的工作空间实例
            
        Raises:
            NotImplementedError: 如果子类没有实现此方法
        """
        error_message = "load method is not implemented."
        raise NotImplementedError(error_message)


class FBWorkspace(Workspace):
    """
    基于文件的任务工作空间
    
    实现的任务将是一个包含相关元素的文件夹：
    - 数据文件
    - 代码工作空间
    - 输出文件（执行后生成最终输出文件）
    
    典型的 FBWorkspace 管道运行方式：
    （我们没有将其添加为方法，因为可能需要根据需求向 `prepare` 或 `execute` 传递参数）
    
    示例代码：
    .. code-block:: python
    
        def run_pipeline(self, **files: str):
            self.prepare()
            self.inject_files(**files)
            self.execute()
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        初始化基于文件的工作空间
        
        Args:
            *args: 传递给父类的参数
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        # 注入到文件夹中的代码，存储在变量中以重现之前的结果
        self.file_dict: dict[str, Any] = {}
        # 工作空间路径，使用 UUID 确保唯一性
        self.workspace_path: Path = RD_AGENT_SETTINGS.workspace_path / uuid.uuid4().hex
        # 由 ``create_ws_ckp`` 创建的内存检查点数据
        self.ws_ckp: bytes | None = None
        # 与工作空间前一版本的变更摘要
        self.change_summary: str | None = None

    @staticmethod
    def _format_code_dict(code_dict: dict[str, str]) -> str:
        """
        将代码字典格式化为字符串的辅助函数
        
        Args:
            code_dict: 文件名到代码内容的映射字典
            
        Returns:
            str: 格式化后的代码字符串
        """
        code_string = ""
        for file_name in sorted(code_dict.keys()):
            code_string += f"\nFile Path: {file_name}\n```\n{code_dict[file_name]}\n```"
        return code_string

    @property
    def all_codes(self) -> str:
        """
        获取工作空间中所有代码文件的内容（排除测试文件）
        
        Returns:
            str: 所有 Python 代码文件的字符串表示
        """
        filtered_dict = {k: v for k, v in self.file_dict.items() if k.endswith(".py") and "test" not in k}
        return self._format_code_dict(filtered_dict)

    def get_codes(self, pattern: str) -> str:
        """
        获取匹配特定模式的代码文件内容（排除测试文件）
        
        Args:
            pattern: 文件名匹配的正则表达式模式
            
        Returns:
            str: 匹配的 Python 代码文件的字符串表示
        """
        filtered_dict = {
            k: v for k, v in self.file_dict.items() if re.search(pattern, k) and k.endswith(".py") and "test" not in k
        }
        return self._format_code_dict(filtered_dict)

    def prepare(self) -> None:
        """
        准备工作空间（除了注入的代码）
        
        准备内容包括：
        - 数据文件
        - 文档文件
        
        典型用法：不同方法共享相同数据，数据通过参数传递
        """
        self.workspace_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def link_all_files_in_folder_to_workspace(data_path: Path, workspace_path: Path) -> None:
        """
        将文件夹中的所有文件链接到工作空间
        
        支持跨平台的文件链接：
        - Linux/Darwin: 使用符号链接
        - Windows: 使用硬链接
        
        Args:
            data_path: 源数据文件夹路径
            workspace_path: 目标工作空间路径
        """
        data_path = Path(data_path).absolute()  # 处理相对路径，避免改变工作目录时失效
        workspace_path = Path(workspace_path)
        for data_file_path in data_path.iterdir():
            workspace_data_file_path = workspace_path / data_file_path.name
            if workspace_data_file_path.exists():
                workspace_data_file_path.unlink()
            if platform.system() in ("Linux", "Darwin"):
                workspace_data_file_path.symlink_to(data_file_path)
            if platform.system() == "Windows":
                os.link(data_file_path, workspace_data_file_path)

    DEL_KEY = "__DEL__"  # 删除文件的特殊键值

    def inject_files(self, **files: str) -> None:
        """
        将代码注入到工作空间文件夹中
        
        文件字典格式：
        {
            <文件名1>: <代码内容>,  # 将 <代码内容> 写入 <文件名1>（创建新文件或替换现有文件）
            <文件名2>: "__DEL__"    # 删除文件名2。当我们要用新文件替换文件时通常使用此方式
        }
        
        Args:
            **files: 文件名到代码内容的映射字典
        """
        self.prepare()
        for k, v in files.items():
            target_file_path = self.workspace_path / k  # 在使用前定义 target_file_path
            if v == self.DEL_KEY:  # 使用 self.DEL_KEY 访问类变量
                if target_file_path.exists():
                    target_file_path.unlink()  # 如果文件存在则删除
                self.file_dict.pop(k, None)  # 安全地从 file_dict 中移除键
            else:
                self.file_dict[k] = v
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                target_file_path.write_text(v)

    def get_files(self) -> list[Path]:
        """
        获取工作空间中的文件列表
        
        为了保持通用性，只返回文件名列表。
        如何总结环境是开发者的责任。
        
        Returns:
            list[Path]: 工作空间中的文件路径列表
        """
        return list(self.workspace_path.iterdir())

    def inject_code_from_folder(self, folder_path: Path) -> None:
        """
        从文件夹加载工作空间
        
        Args:
            folder_path: 源文件夹路径
        """
        for file_path in folder_path.rglob("*"):
            if file_path.suffix in (".py", ".yaml", ".md"):
                relative_path = file_path.relative_to(folder_path)
                self.inject_files(**{str(relative_path): file_path.read_text()})

    def inject_code_from_file_dict(self, workspace: FBWorkspace) -> None:
        """
        从另一个工作空间的 file_dict 加载代码
        
        Args:
            workspace: 源工作空间实例
        """
        for name, code in workspace.file_dict.items():
            self.inject_files(**{name: code})

    def copy(self) -> FBWorkspace:
        """
        复制工作空间
        
        Returns:
            FBWorkspace: 工作空间的深拷贝
        """
        return deepcopy(self)

    def clear(self) -> None:
        """
        清空工作空间
        
        删除工作空间目录并清空文件字典
        """
        shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.file_dict = {}

    def before_execute(self) -> None:
        """
        执行代码前的准备工作
        
        需要准备工作空间并将代码注入到工作空间中
        """
        self.prepare()
        self.inject_files(**self.file_dict)

    def execute(self, env: Env, entry: str) -> str:
        """
        执行工作空间中的代码
        
        每次执行前确保准备和注入代码。
        
        Args:
            env: 执行环境
            entry: 入口点
            
        Returns:
            str: 截断后的标准输出（注意：截断只是为了与旧代码对齐）
        """
        result = self.run(env, entry)
        return result.get_truncated_stdout()  # 注意：截断只是为了与旧代码对齐

    def run(self, env: Env, entry: str) -> EnvResult:
        """
        在环境中执行代码并返回 EnvResult 对象（包含 stdout、exit_code、running_time）
        
        每次执行前确保准备和注入代码。
        
        Args:
            env: 执行环境
            entry: 入口点
            
        Returns:
            EnvResult: 包含执行结果的对象
        """
        self.prepare()
        self.inject_files(**self.file_dict)
        return env.run(entry, str(self.workspace_path), env={"PYTHONPATH": "./"})

    def create_ws_ckp(self) -> None:
        """
        将 ``workspace_path`` 的内容压缩并持久化到 ``self.ws_ckp`` 中，
        以便后续通过 :py:meth:`recover_ws_ckp` 恢复。
        
        检查点功能：
        - 只包含常规文件，最大 100 KB，保持检查点轻量
        - 更大的文件（如数据集）需要单独重新创建或挂载
        - 保留符号链接
        - 支持白名单文件（不受大小限制）
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.workspace_path.rglob("*"):
                # 只包含最大 100 KB 的常规文件，保持检查点轻量
                # 更大的文件（如数据集）需要单独重新创建或挂载
                if file_path.is_symlink():
                    # 在归档中保留符号链接
                    zi = zipfile.ZipInfo(str(file_path.relative_to(self.workspace_path)))
                    zi.create_system = 3  # 表示 Unix 系统
                    zi.external_attr = 0o120777 << 16  # 符号链接文件类型 + 0777 权限
                    zf.writestr(zi, str(file_path.readlink()))
                elif file_path.is_file():
                    size_limit = RD_AGENT_SETTINGS.workspace_ckp_size_limit
                    if (
                        RD_AGENT_SETTINGS.workspace_ckp_white_list_names is not None
                        and file_path.name in RD_AGENT_SETTINGS.workspace_ckp_white_list_names
                    ) or (size_limit <= 0 or file_path.stat().st_size <= size_limit):
                        zf.write(file_path, file_path.relative_to(self.workspace_path))
        self.ws_ckp = buf.getvalue()

    def recover_ws_ckp(self) -> None:
        """
        从由 :py:meth:`create_ws_ckp` 创建的内存检查点恢复工作空间目录
        
        Raises:
            RuntimeError: 如果检查点不存在
        """
        if self.ws_ckp is None:
            msg = "Workspace checkpoint doesn't exist. Call `create_ws_ckp` first."
            raise RuntimeError(msg)
        shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO(self.ws_ckp)
        with zipfile.ZipFile(buf, "r") as zf:
            for info in zf.infolist():
                dest_path = self.workspace_path / info.filename
                # 文件类型位（上4位）在 external_attr 的高16位中
                mode = (info.external_attr >> 16) & 0o170000
                symlink_mode = 0o120000  # Unix 中符号链接文件类型的常量
                if mode == symlink_mode:  # 符号链接
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    link_target = zf.read(info).decode()
                    dest_path.symlink_to(link_target)
                elif info.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with dest_path.open("wb") as f:
                        f.write(zf.read(info))
        # 注意：非常重要，减少对象大小
        self.ws_ckp = None

    def __str__(self) -> str:
        """
        返回工作空间的字符串表示
        
        Returns:
            str: 格式化的工作空间表示字符串
        """
        return f"Workspace[{self.workspace_path=}" + (
            "]" if self.target_task is None else f",{self.target_task.name=}]"
        )


# 实验相关的类型变量
ASpecificWSForExperiment = TypeVar("ASpecificWSForExperiment", bound=Workspace)  # 实验特定工作空间类型
ASpecificWSForSubTasks = TypeVar("ASpecificWSForSubTasks", bound=Workspace)  # 子任务特定工作空间类型


class ExperimentPlan(dict[str, Any]):
    """
    实验计划类
    
    继承自字典，包含每个阶段的计划信息。
    用于存储实验执行过程中的规划数据。
    """


class Experiment(
    ABC,
    Generic[ASpecificTask, ASpecificWSForExperiment, ASpecificWSForSubTasks],
):
    """
    实验抽象基类
    
    实验是一系列任务和由开发者生成的任务实现的序列。
    这是 RD-Agent 的核心概念，用于管理整个研发流程。
    
    实验包含：
    - 子任务列表
    - 基于的实验（可选）
    - 假设（可选）
    - 工作空间管理
    - 反馈机制
    - 运行信息
    """

    def __init__(
        self,
        sub_tasks: Sequence[ASpecificTask],
        based_experiments: Sequence[ASpecificWSForExperiment] = [],
        hypothesis: Hypothesis | None = None,
    ) -> None:
        """
        初始化实验
        
        Args:
            sub_tasks: 子任务序列
            based_experiments: 基于的实验序列，默认为空
            hypothesis: 假设，可选
        """
        self.hypothesis: Hypothesis | None = hypothesis  # 实验可能由假设生成
        self.sub_tasks: Sequence[ASpecificTask] = sub_tasks
        # None 表示：
        # - 实现前的初始化占位符
        # - 开发者主动跳过任务
        self.sub_workspace_list: list[ASpecificWSForSubTasks | None] = [None] * len(self.sub_tasks)
        # TODO: 将在运行器历史中使用
        # 如果实现整个工作流，就不需要使用它，然后可以移除
        self.based_experiments: Sequence[ASpecificWSForExperiment] = based_experiments

        self.experiment_workspace: ASpecificWSForExperiment | None = None

        # 实验可能由不同的开发者开发
        # 最后的反馈用于向下一个开发者传播信息
        # 生命周期：
        # - 开发者为下一个组件分配反馈
        # - 工作流控制清除反馈
        self.prop_dev_feedback: Feedback | None = None

        # TODO: (xiao) 我认为这太具体了；应该将其移入
        # 注意：假设
        # - 只有运行器会分配此变量
        # - 当进入下一个新循环时，我们总是创建一个新实验而不复制之前的结果
        self.running_info = RunningInfo()
        self.sub_results: dict[str, float] = (
            {}
        )  # TODO: 在 Kaggle 中，现在子结果都保存在 self.result 中，将来会移除这个

        # 支持并行多轨迹
        self.local_selection: tuple[int, ...] | None = None
        self.plan: ExperimentPlan | None = (
            None  # 存储此实验的规划信息，应该在 exp_gen.gen 内部生成
        )
        self.user_instructions: UserInstructions | None = None  # 存储此实验的用户指令

    def set_user_instructions(self, user_instructions: UserInstructions | None) -> None:
        """
        设置用户指令
        
        将用户指令设置到实验及其所有相关任务和工作空间中。
        
        Args:
            user_instructions: 用户指令列表，如果为 None 则不做任何操作
        """
        if user_instructions is None:
            return
        if not isinstance(user_instructions, UserInstructions) and isinstance(user_instructions, list):
            user_instructions = UserInstructions(user_instructions)
        self.user_instructions = user_instructions
        for ws in self.sub_workspace_list:
            if ws is not None:
                ws.target_task.user_instructions = user_instructions  # type: ignore[union-attr]
        for task in self.sub_tasks:
            task.user_instructions = user_instructions
        if self.experiment_workspace is not None and self.experiment_workspace.target_task is not None:
            self.experiment_workspace.target_task.user_instructions = user_instructions

    @property
    def result(self) -> object:
        """
        获取实验结果
        
        Returns:
            object: 实验结果
        """
        return self.running_info.result

    @result.setter
    def result(self, value: object) -> None:
        """
        设置实验结果
        
        Args:
            value: 实验结果
        """
        self.running_info.result = value

    def create_ws_ckp(self) -> None:
        """
        创建实验的工作空间检查点
        
        当工作空间可以就地修改时，提供创建检查点和恢复的支持。
        为实验工作空间和所有子工作空间创建检查点。
        """
        if self.experiment_workspace is not None:
            self.experiment_workspace.create_ws_ckp()
        for ws in self.sub_workspace_list:
            if ws is not None:
                ws.create_ws_ckp()

    def recover_ws_ckp(self) -> None:
        """
        从检查点恢复实验的工作空间
        
        恢复实验工作空间和所有子工作空间。
        处理工作空间共享的情况（避免重复恢复）。
        """
        if self.experiment_workspace is not None:
            self.experiment_workspace.recover_ws_ckp()
        for ws in self.sub_workspace_list:
            if ws is not None:
                try:
                    ws.recover_ws_ckp()
                except RuntimeError:
                    # FBWorkspace 在 experiment_workspace 和 sub_workspace_list 之间共享，
                    # 所以如果一个工作空间被恢复两次，recover_ws_ckp 会抛出 RuntimeError
                    print("recover_ws_ckp failed due to one workspace is recovered twice.")


# 更多类型变量定义
ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)  # 特定实验类型
ASpecificPlan = TypeVar("ASpecificPlan", bound=ExperimentPlan)  # 特定计划类型

TaskOrExperiment = TypeVar("TaskOrExperiment", Task, Experiment)  # 任务或实验的联合类型


class Loader(ABC, Generic[TaskOrExperiment]):
    """
    加载器抽象基类
    
    负责加载任务或实验实例的通用接口。
    """
    
    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> TaskOrExperiment:
        """
        加载任务或实验
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            TaskOrExperiment: 加载的任务或实验实例
            
        Raises:
            NotImplementedError: 如果子类没有实现此方法
        """
        err_msg = "load method is not implemented."
        raise NotImplementedError(err_msg)
