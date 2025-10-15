from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
)


class ExtendedBaseSettings(BaseSettings):
    """
    扩展的基础设置类，用于自定义配置源
    """

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        自定义配置源，支持从环境变量、.env文件和基类继承配置
        
        Args:
            settings_cls: 设置类
            init_settings: 初始化设置源
            env_settings: 环境变量设置源
            dotenv_settings: .env文件设置源
            file_secret_settings: 文件密钥设置源
            
        Returns:
            配置源元组
        """
        # 1) 从基类开始遍历
        def base_iter(settings_cls: type[ExtendedBaseSettings]) -> list[type[ExtendedBaseSettings]]:
            bases = []
            for cl in settings_cls.__bases__:
                if issubclass(cl, ExtendedBaseSettings) and cl is not ExtendedBaseSettings:
                    bases.append(cl)
                    bases.extend(base_iter(cl))
            return bases

        # 2) 从基类构建EnvSettingsSource，以便添加父级环境变量源
        parent_env_settings = [
            EnvSettingsSource(
                base_cls,
                case_sensitive=base_cls.model_config.get("case_sensitive"),
                env_prefix=base_cls.model_config.get("env_prefix"),
                env_nested_delimiter=base_cls.model_config.get("env_nested_delimiter"),
            )
            for base_cls in base_iter(cast("type[ExtendedBaseSettings]", settings_cls))
        ]
        return init_settings, env_settings, *parent_env_settings, dotenv_settings, file_secret_settings


class RDAgentSettings(ExtendedBaseSettings):
    """
    RD-Agent核心配置设置类
    """

    # Azure文档智能服务配置
    azure_document_intelligence_key: str = ""
    """Azure文档智能服务密钥"""
    azure_document_intelligence_endpoint: str = ""
    """Azure文档智能服务端点"""

    # 因子提取配置
    max_input_duplicate_factor_group: int = 300
    """输入重复因子组的最大数量"""
    max_output_duplicate_factor_group: int = 20
    """输出重复因子组的最大数量"""
    max_kmeans_group_number: int = 40
    """K-means聚类组的最大数量"""

    # 工作空间配置
    workspace_path: Path = Path.cwd() / "git_ignore_folder" / "RD-Agent_workspace"
    """工作空间路径"""
    workspace_ckp_size_limit: int = 0
    """工作空间检查点文件大小限制（字节），0表示无限制"""
    workspace_ckp_white_list_names: list[str] | None = None
    """
    工作空间的检查点是一个zip文件。
    0（或任何小于等于0的值）表示工作空间检查点中的文件没有大小限制
    """

    # 多进程配置
    multi_proc_n: int = 8
    """多进程数量"""

    # pickle缓存配置
    cache_with_pickle: bool = True  # whether to use pickle cache
    """是否使用pickle缓存"""
    pickle_cache_folder_path_str: str = str(
        Path.cwd() / "pickle_cache/",
    )  # the path of the folder to store the pickle cache
    """存储pickle缓存的文件夹路径"""
    use_file_lock: bool = (
        True  # when calling the function with same parameters, whether to use file lock to avoid
        # executing the function multiple times
    )
    """当使用相同参数调用函数时，是否使用文件锁来避免多次执行函数"""

    # 杂项配置
    """上下文stdout的限制"""
    stdout_context_len: int = 400
    """标准输出上下文长度"""
    stdout_line_len: int = 10000
    """标准输出行长度"""

    enable_mlflow: bool = False
    """是否启用MLflow"""

    initial_fator_library_size: int = 20
    """初始因子库大小"""

    # 并行循环配置
    step_semaphore: int | dict[str, int] = 1
    """每个步骤的信号量；可以指定整体信号量或步骤级信号量，如{"coding": 3, "running": 2}"""

    def get_max_parallel(self) -> int:
        """
        根据信号量设置，返回并行循环的最大数量
        
        Returns:
            并行循环的最大数量
        """
        if isinstance(self.step_semaphore, int):
            return self.step_semaphore
        return max(self.step_semaphore.values())

    # NOTE: 用于调试
    # 以下函数仅用于调试，在主逻辑中不是必需的
    subproc_step: bool = False
    """是否在子进程中执行步骤（用于调试）"""

    def is_force_subproc(self) -> bool:
        """
        是否强制在子进程中执行
        
        Returns:
            是否强制在子进程中执行
        """
        return self.subproc_step or self.get_max_parallel() > 1

    # 模板配置:
    app_tpl: str | None = None  # for application to override the default template, example: "app/fintune/tpl"
    """应用程序模板路径，用于覆盖默认模板，例如："app/fintune/tpl"""


RD_AGENT_SETTINGS = RDAgentSettings()
"""RD-Agent全局配置实例"""