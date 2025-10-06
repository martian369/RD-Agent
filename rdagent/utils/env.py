"""
环境管理工具模块

该模块的主要目标是为 RD-Agent 创建统一的执行环境：
- 所有代码和数据都包含在一个文件夹中
- 支持本地环境和 Docker 环境的统一接口
- 提供环境配置、准备和执行功能
"""

# TODO: 将特定场景的 Docker 环境移到其他文件夹中

from abc import abstractmethod
import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import re
import select
import shutil
import subprocess
import time
from types import MappingProxyType
from typing import Any, Generator, Generic, Mapping, Optional, TypeVar, cast
import uuid
import zipfile

import docker
import docker.models
import docker.models.containers
import docker.types
from pydantic import BaseModel, model_validator
from pydantic_settings import SettingsConfigDict
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm

from rdagent.core.conf import ExtendedBaseSettings
from rdagent.core.experiment import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils import filter_redundant_text
from rdagent.utils.agent.tpl import T
from rdagent.utils.fmt import shrink_text
from rdagent.utils.workflow import wait_retry


def get_docker_client() -> docker.DockerClient:
    """
    获取一个能够与各种 Docker 安装兼容的 Docker 客户端
    
    支持包括 Docker Desktop 和 Colima 在内的多种 Docker 安装方式
    
    Returns:
        docker.DockerClient: Docker 客户端实例
    """
    try:
        # 首先尝试使用默认方法连接
        client = docker.from_env()
        # 测试连接
        client.ping()
        return client
    except docker.errors.DockerException:
        # 如果默认方法失败，尝试常见的替代路径（如 Colima 等）
        possible_sockets = [
            os.path.expanduser("~/.colima/docker.sock"),
            os.path.expanduser("~/.colima/default/docker.sock"),
            "/var/run/docker.sock"
        ]
        
        for socket_path in possible_sockets:
            if os.path.exists(socket_path):
                try:
                    client = docker.DockerClient(base_url=f"unix://{socket_path}")
                    # 测试连接
                    client.ping()
                    return client
                except Exception:
                    continue
        
        # 如果所有路径都失败，则抛出异常
        raise docker.errors.DockerException("无法连接到 Docker 守护进程")


def cleanup_container(container: docker.models.containers.Container | None, context: str = "") -> None:  # type: ignore[no-any-unimported]
    """
    清理 Docker 容器的共享辅助函数
    总是在删除容器之前停止它

    Parameters
    ----------
    container : docker 容器对象或 None
        要清理的容器，如果没有容器需要清理则为 None
    context : str
        日志记录的附加上下文（例如，"健康检查"、"GPU 测试"）
    """
    if container is not None:
        try:
            # 总是先停止 - stop() 方法在容器已停止时不会抛出错误
            container.stop()
            container.remove()
        except Exception as cleanup_error:
            # 记录清理错误但不掩盖原始异常
            context_str = f" {context}" if context else ""
            logger.warning(f"清理{context_str}容器 {container.id} 失败: {cleanup_error}")


# 将卷中的所有绑定路径标准化为使用工作目录的绝对路径
def normalize_volumes(vols: dict[str, str | dict[str, str]], working_dir: str) -> dict:
    """
    标准化卷映射中的路径为绝对路径
    
    Parameters:
        vols: 卷映射字典
        working_dir: 工作目录路径
        
    Returns:
        标准化后的卷映射字典
    """
    abs_vols: dict[str, str | dict[str, str]] = {}

    def to_abs(path: str) -> str:
        # 将相对路径转换为使用工作目录的绝对路径
        return os.path.abspath(os.path.join(working_dir, path)) if not os.path.isabs(path) else path

    for lp, vinfo in vols.items():
        # 支持两种格式:
        # 1. {'host_path': {'bind': 'container_path', ...}}
        # 2. {'host_path': 'container_path'}
        if isinstance(vinfo, dict):
            # abs_vols = cast(dict[str, dict[str, str]], abs_vols)
            vinfo = vinfo.copy()
            vinfo["bind"] = to_abs(vinfo["bind"])
            abs_vols[lp] = vinfo
        else:
            # abs_vols = cast(dict[str, str], abs_vols)
            abs_vols[lp] = to_abs(vinfo)
    return abs_vols


def pull_image_with_progress(image: str) -> None:
    """
    带进度显示地拉取 Docker 镜像
    
    Parameters:
        image: 要拉取的镜像名称
    """
    client = docker.APIClient(base_url="unix://var/run/docker.sock")
    pull_logs = client.pull(image, stream=True, decode=True)
    progress_bars = {}

    for log in pull_logs:
        if "id" in log and log.get("progressDetail"):
            layer_id = log["id"]
            progress_detail = log["progressDetail"]
            current = progress_detail.get("current", 0)
            total = progress_detail.get("total", 0)

            if total:
                if layer_id not in progress_bars:
                    progress_bars[layer_id] = tqdm(total=total, desc=f"层 {layer_id}", unit="B", unit_scale=True)
                progress_bars[layer_id].n = current
                progress_bars[layer_id].refresh()

        elif "status" in log:
            print(log["status"])

    for pb in progress_bars.values():
        pb.close()


class EnvConf(ExtendedBaseSettings):
    """
    环境配置基类
    
    所有环境配置都继承自此类，提供基础的环境设置功能
    """
    default_entry: str  # 默认入口点
    extra_volumes: dict = {}  # 额外的卷映射
    running_timeout_period: int | None = 3600  # 运行超时时间（秒），默认10分钟
    # 辅助设置以支持透明化
    enable_cache: bool = True  # 是否启用缓存
    retry_count: int = 5  # Docker 运行的重试次数
    retry_wait_seconds: int = 10  # Docker 运行重试等待秒数

    model_config = SettingsConfigDict(
        # TODO: 添加前缀 ....
        env_parse_none_str="None",  # 这是接受 `RUNNING_TIMEOUT_PERIOD=None` 的关键
    )


ASpecificEnvConf = TypeVar("ASpecificEnvConf", bound=EnvConf)


@dataclass
class EnvResult:
    """
    环境运行结果数据类
    
    包含运行环境的结果信息：标准输出、退出码和运行时间
    """

    stdout: str  # 标准输出
    exit_code: int  # 退出码
    running_time: float  # 运行时间（秒）

    def get_truncated_stdout(self) -> str:
        """
        获取截断后的标准输出
        
        Returns:
            截断后的标准输出字符串
        """
        return shrink_text(
            filter_redundant_text(self.stdout),
            context_lines=RD_AGENT_SETTINGS.stdout_context_len,
            line_len=RD_AGENT_SETTINGS.stdout_line_len,
        )


class Env(Generic[ASpecificEnvConf]):
    """
    环境抽象基类
    
    使用 BaseModel 作为设置的原因是它提供的功能：
    - 提供基础类型检查功能
    - 加载和转储信息更容易：例如，可以使用 `pydantic-yaml` 等包
    """

    conf: ASpecificEnvConf  # 不同环境有不同的配置

    def __init__(self, conf: ASpecificEnvConf):
        """
        初始化环境
        
        Parameters:
            conf: 环境配置对象
        """
        self.conf = conf

    def zip_a_folder_into_a_file(self, folder_path: str, zip_file_path: str) -> None:
        """
        将文件夹压缩为文件，使用 zipfile 而不是 subprocess
        
        Parameters:
            folder_path: 要压缩的文件夹路径
            zip_file_path: 压缩文件输出路径
        """
        with zipfile.ZipFile(zip_file_path, "w") as z:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))

    def unzip_a_file_into_a_folder(self, zip_file_path: str, folder_path: str) -> None:
        """
        将文件解压到文件夹，使用 zipfile 而不是 subprocess
        
        Parameters:
            zip_file_path: 压缩文件路径
            folder_path: 解压目标文件夹路径
        """
        # 解压前清空目标文件夹
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extractall(folder_path)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        根据配置准备环境
        """

    def check_output(
        self, entry: str | None = None, local_path: str = ".", env: dict | None = None, **kwargs: dict
    ) -> str:
        """
        在环境中运行文件夹

        Parameters
        ----------
        entry : str | None
            运行时可能使用的入口点
            例如，运行和总结项目时可能有不同的入口点
        local_path : str | None
            本地路径（项目路径，主要用于代码）将被挂载到 Docker 中
            None 本地路径的一些示例：
            - 例如，运行 Docker 来更新 extra_volumes 中的数据
            - 简单地运行镜像。结果由输出或网络产生
        env : dict | None
            使用特定环境运行代码

        Returns
        -------
            标准输出
        """
        result = self.run(entry=entry, local_path=local_path, env=env, **kwargs)
        return result.stdout

    def __run_with_retry(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        """
        带重试机制的运行函数
        
        Parameters:
            entry: 入口点
            local_path: 本地路径
            env: 环境变量
            running_extra_volume: 运行时额外卷映射
            
        Returns:
            EnvResult: 运行结果
        """
        for retry_index in range(self.conf.retry_count + 1):
            try:
                start = time.time()
                log_output, return_code = self._run(
                    entry,
                    local_path,
                    env,
                    running_extra_volume=running_extra_volume,
                )
                end = time.time()
                logger.info(f"运行时间: {end - start} 秒")
                if self.conf.running_timeout_period is not None and end - start + 1 >= self.conf.running_timeout_period:
                    logger.warning(
                        f"运行时间超过 {self.conf.running_timeout_period} 秒，进程已被终止。"
                    )
                    log_output += f"\n\n运行时间超过 {self.conf.running_timeout_period} 秒，进程已被终止。"
                return EnvResult(log_output, return_code, end - start)
            except Exception as e:
                if retry_index == self.conf.retry_count:
                    raise
                logger.warning(
                    f"运行容器时出错: {e}, 当前尝试次数: {retry_index + 1}, 剩余 {self.conf.retry_count - retry_index - 1} 次重试。"
                )
                time.sleep(self.conf.retry_wait_seconds)
        raise RuntimeError  # 用于通过 CI

    def run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        **kwargs: dict,
    ) -> EnvResult:
        """
        在环境中运行文件夹并返回标准输出、退出码和运行时间

        Parameters
        ----------
        entry : str | None
            运行时可能使用的入口点
            例如，运行和总结项目时可能有不同的入口点
        local_path : str | None
            本地路径（项目路径，主要用于代码）将被挂载到 Docker 中
            None 本地路径的一些示例：
            - 例如，运行 Docker 来更新 extra_volumes 中的数据
            - 简单地运行镜像。结果由输出或网络产生
        env : dict | None
            使用特定环境运行代码

        Returns
        -------
            EnvResult: 包含标准输出、退出码和运行时间（秒）的对象
        """
        running_extra_volume = kwargs.get("running_extra_volume", {})
        if entry is None:
            entry = self.conf.default_entry

        if "|" in entry:
            logger.warning(
                "您正在使用带有 shell 管道的命令（即'|'）。"
                "退出码（$exit_code）将反映管道中最后一个命令的结果。"
            )

        # FIXME: 输入路径和缓存路径在这里是硬编码的
        # 我们不想更改输入和缓存路径中的内容
        # 否则，可能会产生大量警告
        def _get_chmod_cmd(workspace_path: str) -> str:
            def _get_path_stem(path: str) -> str | None:
                # 如果输入路径是相对路径，只保留第一个组件
                p = Path(path)
                if not p.is_absolute() and p.parts:
                    return p.parts[0]
                return None

            find_cmd = f"find {workspace_path} -mindepth 1 -maxdepth 1"
            for name in [
                _get_path_stem(T("scenarios.data_science.share:scen.cache_path").r()),
                _get_path_stem(T("scenarios.data_science.share:scen.input_path").r()),
            ]:
                find_cmd += f" ! -name {name}"
            chmod_cmd = f"{find_cmd} -exec chmod -R 777 {{}} +"
            return chmod_cmd

        if self.conf.running_timeout_period is None:
            timeout_cmd = entry
        else:
            timeout_cmd = f"timeout --kill-after=10 {self.conf.running_timeout_period} {entry}"
        entry_add_timeout = (
            f"/bin/sh -c '"  # shell 命令开始
            + f"{timeout_cmd}; entry_exit_code=$?; "
            + (
                f"{_get_chmod_cmd(self.conf.mount_path)}; "
                # 我们不必更改缓存和输入文件夹的权限来删除它
                # + f"if [ -d {self.conf.mount_path}/cache ]; then chmod 777 {self.conf.mount_path}/cache; fi; " +
                #     f"if [ -d {self.conf.mount_path}/input ]; then chmod 777 {self.conf.mount_path}/input; fi; "
                if isinstance(self.conf, DockerConf)
                else ""
            )
            + "exit $entry_exit_code"
            + "'"  # shell 命令结束
        )

        if self.conf.enable_cache:
            result = self.cached_run(entry_add_timeout, local_path, env, running_extra_volume)
        else:
            result = self.__run_with_retry(
                entry_add_timeout,
                local_path,
                env,
                running_extra_volume,
            )

        return result

    def cached_run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
    ) -> EnvResult:
        """
        在环境中运行文件夹
        将缓存输出和文件夹差异以供下一轮运行使用
        使用 Python 代码和参数（entry, running_extra_volume）作为键来哈希输入

        Parameters:
            entry: 入口点
            local_path: 本地路径
            env: 环境变量
            running_extra_volume: 运行时额外卷映射
            
        Returns:
            EnvResult: 运行结果
        """
        target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"utils.env.run"
        target_folder.mkdir(parents=True, exist_ok=True)

        # 我们必须将数据信息（超出代码）添加到键中
        # 否则，所有操作数据的命令都将失效（例如 rm -r submission.csv）
        # 因此，我们递归地遍历文件夹并将排序的相对文件名列表作为键的一部分
        # data_key = []
        # for path in Path(local_path).rglob("*"):
        #     p = str(path.relative_to(Path(local_path)))
        #     if p.startswith("__pycache__"):
        #         continue
        #     data_key.append(p)
        # data_key = sorted(data_key)

        key = md5_hash(
            json.dumps(
                [
                    [str(path.relative_to(Path(local_path))), path.read_text()]
                    for path in sorted(list(Path(local_path).rglob("*.py")) + list(Path(local_path).rglob("*.csv")))
                ]
            )
            + json.dumps({"entry": entry, "running_extra_volume": dict(running_extra_volume)})
            + json.dumps({"extra_volumes": self.conf.extra_volumes})
            # + json.dumps(data_key)
        )
        if Path(target_folder / f"{key}.pkl").exists() and Path(target_folder / f"{key}.zip").exists():
            with open(target_folder / f"{key}.pkl", "rb") as f:
                ret = pickle.load(f)
            self.unzip_a_file_into_a_folder(str(target_folder / f"{key}.zip"), local_path)
        else:
            ret = self.__run_with_retry(entry, local_path, env, running_extra_volume)
            with open(target_folder / f"{key}.pkl", "wb") as f:
                pickle.dump(ret, f)
            self.zip_a_folder_into_a_file(local_path, str(target_folder / f"{key}.zip"))
        return cast(EnvResult, ret)

    @abstractmethod
    def _run(
        self,
        entry: str | None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        在给定环境和本地路径中执行指定的入口点

        Parameters
        ----------
        entry : str | None
            要执行的入口点。如果为 None，则默认为配置的入口点
        local_path : str
            执行应该发生的本地目录路径
        env : dict | None
            执行期间要设置的环境变量
        kwargs : dict
            执行自定义的附加关键字参数

        Returns
        -------
        tuple[str, int]
            包含标准输出和退出码的元组
        """
        pass

    def dump_python_code_run_and_get_results(
        self,
        code: str,
        dump_file_names: list[str],
        local_path: str,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        code_dump_file_py_name: Optional[str] = None,
    ) -> tuple[str, list]:
        """
        将代码转储到本地路径并运行代码
        
        Parameters:
            code: 要运行的 Python 代码
            dump_file_names: 要提取的文件名列表
            local_path: 本地路径
            env: 环境变量
            running_extra_volume: 运行时额外卷映射
            code_dump_file_py_name: 代码转储文件的名称
            
        Returns:
            tuple[str, list]: 标准输出和结果列表
        """
        random_file_name = f"{uuid.uuid4()}.py" if code_dump_file_py_name is None else f"{code_dump_file_py_name}.py"
        with open(os.path.join(local_path, random_file_name), "w") as f:
            f.write(code)
        entry = f"python {random_file_name}"
        log_output = self.check_output(entry, local_path, env, running_extra_volume=dict(running_extra_volume))
        results = []
        os.remove(os.path.join(local_path, random_file_name))
        for name in dump_file_names:
            if os.path.exists(os.path.join(local_path, f"{name}")):
                results.append(pickle.load(open(os.path.join(local_path, f"{name}"), "rb")))
                os.remove(os.path.join(local_path, f"{name}"))
            else:
                return log_output, []
        return log_output, results


# class EnvWithCache
#

## 本地环境 -----


class LocalConf(EnvConf):
    """
    本地环境配置
    
    用于本地环境的配置设置
    """
    bin_path: str = ""
    """路径格式如 <path1>:<path2>:<path3>，将被前置到 bin 路径"""

    retry_count: int = 0  # 重试次数；运行 `retry_count + 1` 次
    live_output: bool = True  # 是否实时输出


ASpecificLocalConf = TypeVar("ASpecificLocalConf", bound=LocalConf)


class LocalEnv(Env[ASpecificLocalConf]):
    """
    本地环境类
    
    有时本地环境可能更适合测试
    """

    def prepare(self) -> None: 
        """
        准备本地环境
        """
        ...

    def _run(
        self,
        entry: str | None = None,
        local_path: str | None = None,
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: dict,
    ) -> tuple[str, int]:
        """
        在本地环境中运行代码
        
        Parameters:
            entry: 入口点
            local_path: 本地路径
            env: 环境变量
            running_extra_volume: 运行时额外卷映射
            **kwargs: 其他参数
            
        Returns:
            tuple[str, int]: 标准输出和退出码
        """

        # 处理卷链接
        volumes = {}
        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp["bind"] if isinstance(rp, dict) else rp
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = T("scenarios.data_science.share:scen.cache_path").r()
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp

        assert local_path is not None, "local_path 不应为 None"
        volumes = normalize_volumes(volumes, local_path)

        @contextlib.contextmanager
        def _symlink_ctx(vol_map: Mapping[str, str]) -> Generator[None, None, None]:
            """
            符号链接上下文管理器
            
            Parameters:
                vol_map: 卷映射
            """
            created_links: list[Path] = []
            try:
                for real, link in vol_map.items():
                    link_path = Path(link)
                    real_path = Path(real)
                    if not link_path.parent.exists():
                        link_path.parent.mkdir(parents=True, exist_ok=True)
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    link_path.symlink_to(real_path)
                    created_links.append(link_path)
                yield
            finally:
                for p in created_links:
                    try:
                        if p.is_symlink() or p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass

        with _symlink_ctx(volumes):
            # 设置环境
            if env is None:
                env = {}
            path = [*self.conf.bin_path.split(":"), "/bin/", "/usr/bin/", *env.get("PATH", "").split(":")]
            env["PATH"] = ":".join(path)

            if entry is None:
                entry = self.conf.default_entry

            print(Rule("[bold green]本地环境日志开始[/bold green]", style="dark_orange"))
            table = Table(title="运行信息", show_header=False)
            table.add_column("键", style="bold cyan")
            table.add_column("值", style="bold magenta")
            table.add_row("入口点", entry)
            table.add_row("本地路径", local_path or "")
            table.add_row("环境", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("卷", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)

            cwd = Path(local_path).resolve() if local_path else None
            env = {k: str(v) if isinstance(v, int) else v for k, v in env.items()}

            process = subprocess.Popen(
                entry,
                cwd=cwd,
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                universal_newlines=True,
            )

            # 设置轮询
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("子进程未正确创建 stdout/stderr 管道")

            if self.conf.live_output:
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()

                poller = select.poll()
                poller.register(stdout_fd, select.POLLIN)
                poller.register(stderr_fd, select.POLLIN)

                combined_output = ""
                while True:
                    if process.poll() is not None:
                        break
                    events = poller.poll(100)
                    for fd, event in events:
                        if event & select.POLLIN:
                            if fd == stdout_fd:
                                while True:
                                    output = process.stdout.readline()
                                    if output == "":
                                        break
                                    Console().print(output.strip(), markup=False)
                                    combined_output += output
                            elif fd == stderr_fd:
                                while True:
                                    error = process.stderr.readline()
                                    if error == "":
                                        break
                                    Console().print(error.strip(), markup=False)
                                    combined_output += error

                # 捕获任何最终输出
                remaining_output, remaining_error = process.communicate()
                if remaining_output:
                    Console().print(remaining_output.strip(), markup=False)
                    combined_output += remaining_output
                if remaining_error:
                    Console().print(remaining_error.strip(), markup=False)
                    combined_output += remaining_error
            else:
                # 牺牲实时输出以避免可能的标准 I/O 挂起
                out, err = process.communicate()
                Console().print(out, end="", markup=False)
                Console().print(err, end="", markup=False)
                combined_output = out + err

            return_code = process.returncode
            print(Rule("[bold green]本地环境日志结束[/bold green]", style="dark_orange"))

            return combined_output, return_code


class CondaConf(LocalConf):
    """
    Conda 环境配置
    
    用于 Conda 环境的配置设置
    """
    conda_env_name: str  # Conda 环境名称
    default_entry: str = "python main.py"  # 默认入口点

    @model_validator(mode="after")
    def change_bin_path(self, **data: Any) -> "CondaConf":
        """
        更改二进制路径
        
        Returns:
            CondaConf: 更新后的配置对象
        """
        conda_path_result = subprocess.run(
            f"conda run -n {self.conda_env_name} --no-capture-output env | grep '^PATH='",
            capture_output=True,
            text=True,
            shell=True,
        )
        self.bin_path = conda_path_result.stdout.strip().split("=")[1] if conda_path_result.returncode == 0 else ""
        return self


class MLECondaConf(CondaConf):
    """
    MLE Conda 环境配置
    
    与 Docker 设置对齐的配置
    """
    enable_cache: bool = False  # 与 Docker 设置对齐


## Docker 环境 -----
class DockerConf(EnvConf):
    """
    Docker 环境配置
    
    用于 Docker 环境的配置设置
    """
    build_from_dockerfile: bool = False  # 是否从 Dockerfile 构建
    dockerfile_folder_path: Optional[Path] = (
        None  # Dockerfile 路径，当 build_from_dockerfile 为 False 时提供的可选路径
    )
    image: str  # 要构建的镜像名称
    mount_path: str  # Docker 镜像中挂载文件夹的路径
    default_entry: str  # 镜像的入口点

    extra_volumes: dict = {}
    """接受卷字典，可以是以下格式之一:
    {<host_path>: <container_path>} 或
    {<host_path>: {"bind": <container_path>, "mode": <mode, ro/rw/默认为 extra_volume_mode>}}
    """
    extra_volume_mode: str = "ro"  # 默认情况下，只有 mount_path 应该是可写的，其他更改为只读
    # 有时，我们需要为工作区维护一些额外数据
    # 额外数据可能是共享的，下载可能耗时
    # 所以我们只想下载一次
    network: str | None = "bridge"  # Docker 的网络模式
    shm_size: str | None = None  # 共享内存大小
    enable_gpu: bool = True  # 因为我们会在 GPU 不可用时自动禁用，所以默认启用
    mem_limit: str | None = "48g"  # 添加内存限制属性
    cpu_count: int | None = None  # 添加 CPU 限制属性

    running_timeout_period: int | None = 3600  # 1 小时

    enable_cache: bool = True  # 启用缓存机制

    retry_count: int = 5  # Docker 运行的重试次数
    retry_wait_seconds: int = 10  # Docker 运行重试等待秒数


class QlibCondaConf(CondaConf):
    """
    Qlib Conda 环境配置
    
    用于 Qlib Conda 环境的配置设置
    """
    conda_env_name: str = "rdagent4qlib"  # Conda 环境名称
    enable_cache: bool = False  # 是否启用缓存
    default_entry: str = "qrun conf.yaml"  # 默认入口点
    # extra_volumes: dict = {str(Path("~/.qlib/").expanduser().resolve().absolute()): "/root/.qlib/"}


class QlibCondaEnv(LocalEnv[QlibCondaConf]):
    """
    Qlib Conda 环境
    
    用于 Qlib Conda 环境的实现
    """
    def prepare(self) -> None:
        """如果环境尚未创建，则准备 Conda 环境"""
        try:
            envs = subprocess.run("conda env list", capture_output=True, text=True, shell=True)
            if self.conf.conda_env_name not in envs.stdout:
                print(f"[yellow]未找到 Conda 环境 '{self.conf.conda_env_name}'，正在创建...[/yellow]")
                subprocess.check_call(
                    f"conda create -y -n {self.conf.conda_env_name} python=3.10",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install --upgrade pip cython",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install git+https://github.com/microsoft/qlib.git@3e72593b8c985f01979bebcf646658002ac43b00",
                    shell=True,
                )
                subprocess.check_call(
                    f"conda run -n {self.conf.conda_env_name} pip install catboost xgboost scipy==1.11.4 tables torch",
                    shell=True,
                )
        except Exception as e:
            print(f"[red]准备 Conda 环境失败: {e}[/red]")


class QlibDockerConf(DockerConf):
    """
    Qlib Docker 环境配置
    
    用于 Qlib Docker 环境的配置设置
    """
    model_config = SettingsConfigDict(
        env_prefix="QLIB_DOCKER_",
        env_parse_none_str="None",  # 这是接受 `RUNNING_TIMEOUT_PERIOD=None` 的关键
    )

    build_from_dockerfile: bool = True  # 是否从 Dockerfile 构建
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "qlib" / "docker"  # Dockerfile 文件夹路径
    image: str = "local_qlib:latest"  # 镜像名称
    mount_path: str = "/workspace/qlib_workspace/"  # 挂载路径
    default_entry: str = "qrun conf.yaml"  # 默认入口点
    extra_volumes: dict = {
        str(Path("~/.qlib/").expanduser().resolve().absolute()): {"bind": "/root/.qlib/", "mode": "rw"}
    }  # 额外卷映射
    shm_size: str | None = "16g"  # 共享内存大小
    enable_gpu: bool = True  # 是否启用 GPU
    enable_cache: bool = False  # 是否启用缓存


class KGDockerConf(DockerConf):
    """
    Kaggle Docker 环境配置
    
    用于 Kaggle Docker 环境的配置设置
    """
    model_config = SettingsConfigDict(env_prefix="KG_DOCKER_")

    build_from_dockerfile: bool = True  # 是否从 Dockerfile 构建
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "kaggle_docker"  # Dockerfile 文件夹路径
    image: str = "local_kg:latest"  # 镜像名称
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/kg_workspace/"  # 挂载路径
    default_entry: str = "python train.py"  # 默认入口点
    # extra_volumes: dict = {
    #     # TODO 连接到数据存储位置
    #     Path("git_ignore_folder/data").resolve(): "/root/.data/"
    # }

    running_timeout_period: int | None = 600  # 运行超时时间（秒）
    mem_limit: str | None = (
        "48g"  # 添加内存限制属性 # new-york-city-taxi-fare-prediction 可能需要更多内存
    )


class DSDockerConf(DockerConf):
    """
    数据科学 Docker 环境配置
    
    用于数据科学 Docker 环境的配置设置
    """
    model_config = SettingsConfigDict(env_prefix="DS_DOCKER_")

    build_from_dockerfile: bool = True  # 是否从 Dockerfile 构建
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "DS_docker"  # Dockerfile 文件夹路径
    image: str = "local_ds:latest"  # 镜像名称
    mount_path: str = "/kaggle/workspace"  # 挂载路径
    default_entry: str = "python main.py"  # 默认入口点

    running_timeout_period: int | None = 600  # 运行超时时间（秒）
    mem_limit: str | None = (
        "48g"  # 添加内存限制属性 # new-york-city-taxi-fare-prediction 可能需要更多内存
    )


class MLEBDockerConf(DockerConf):
    """
    MLEBench Docker 环境配置
    
    用于 MLEBench Docker 环境的配置设置
    """
    model_config = SettingsConfigDict(env_prefix="MLEB_DOCKER_")

    build_from_dockerfile: bool = True  # 是否从 Dockerfile 构建
    dockerfile_folder_path: Path = Path(__file__).parent.parent / "scenarios" / "kaggle" / "docker" / "mle_bench_docker"  # Dockerfile 文件夹路径
    image: str = "local_mle:latest"  # 镜像名称
    # image: str = "gcr.io/kaggle-gpu-images/python:latest"
    mount_path: str = "/workspace/data_folder/"  # 挂载路径
    default_entry: str = "mlebench prepare --all"  # 默认入口点
    # extra_volumes: dict = {
    #     # TODO 连接到数据存储位置
    #     Path("git_ignore_folder/data").resolve(): "/root/.data/"
    # }
    mem_limit: str | None = (
        "48g"  # 添加内存限制属性 # new-york-city-taxi-fare-prediction 可能需要更多内存
    )
    enable_cache: bool = False  # 是否启用缓存


# physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/FIDDLE_mimic3
class DockerEnv(Env[DockerConf]):
    """
    Docker 环境类
    
    用于 Docker 环境的实现
    """
    # TODO: 将输出保存到特定文件

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        下载镜像（如果不存在）
        """
        client = get_docker_client()
        if (
            self.conf.build_from_dockerfile
            and self.conf.dockerfile_folder_path is not None
            and self.conf.dockerfile_folder_path.exists()
        ):
            logger.info(f"正在从 Dockerfile 构建镜像: {self.conf.dockerfile_folder_path}")
            resp_stream = client.api.build(
                path=str(self.conf.dockerfile_folder_path), tag=self.conf.image, network_mode=self.conf.network
            )
            if isinstance(resp_stream, str):
                logger.info(resp_stream)
            with Progress(SpinnerColumn(), TextColumn("{task.description}")) as p:
                task = p.add_task("[cyan]正在构建镜像...")
                for part in resp_stream:
                    lines = part.decode("utf-8").split("\r\n")
                    for line in lines:
                        if line.strip():
                            status_dict = json.loads(line)
                            if "error" in status_dict:
                                p.update(task, description=f"[red]错误: {status_dict['error']}")
                                raise docker.errors.BuildError(status_dict["error"], "")
                            if "stream" in status_dict:
                                p.update(task, description=status_dict["stream"])
            logger.info(f"完成从 Dockerfile 构建镜像: {self.conf.dockerfile_folder_path}")
        try:
            client.images.get(self.conf.image)
        except docker.errors.ImageNotFound:
            image_pull = client.api.pull(self.conf.image, stream=True, decode=True)
            current_status = ""
            layer_set = set()
            completed_layers = 0
            with Progress(TextColumn("{task.description}"), TextColumn("{task.fields[progress]}")) as sp:
                main_task = sp.add_task("[cyan]正在拉取镜像...", progress="")
                status_task = sp.add_task("[bright_magenta]层状态", progress="")
                for line in image_pull:
                    if "error" in line:
                        sp.update(status_task, description=f"[red]错误", progress=line["error"])
                        raise docker.errors.APIError(line["error"])

                    layer_id = line["id"]
                    status = line["status"]
                    p_text = line.get("progress", None)

                    if layer_id not in layer_set:
                        layer_set.add(layer_id)

                    if p_text:
                        current_status = p_text

                    if status == "Pull complete" or status == "Already exists":
                        completed_layers += 1

                    sp.update(main_task, progress=f"[green]{completed_layers}[white]/{len(layer_set)} 层完成")
                    sp.update(
                        status_task,
                        description=f"[bright_magenta]层 {layer_id} [yellow]{status}",
                        progress=current_status,
                    )
        except docker.errors.APIError as e:
            raise RuntimeError(f"拉取镜像时出错: {e}")

    def _gpu_kwargs(self, client: docker.DockerClient) -> dict:  # type: ignore[no-any-unimported]
        """
        根据 GPU 可用性获取 GPU 参数
        
        Parameters:
            client: Docker 客户端
            
        Returns:
            dict: GPU 参数字典
        """
        if not self.conf.enable_gpu:
            return {}
        gpu_kwargs = {
            "device_requests": (
                [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if self.conf.enable_gpu else None
            ),
        }

        def get_image(image_name: str) -> None:
            """
            获取镜像
            
            Parameters:
                image_name: 镜像名称
            """
            try:
                client.images.get(image_name)
            except docker.errors.ImageNotFound:
                pull_image_with_progress(image_name)

        @wait_retry(5, 10)
        def _f() -> dict:
            """
            获取 GPU 参数的内部函数
            
            Returns:
                dict: GPU 参数字典
            """
            container = None
            try:
                get_image(self.conf.image)
                container = client.containers.run(self.conf.image, "nvidia-smi", detach=True, **gpu_kwargs)
                # 等待容器完成
                container.wait()
                logger.info("GPU 设备可用。")
            except docker.errors.APIError:
                return {}
            finally:
                cleanup_container(container, context="GPU 测试")
            return gpu_kwargs

        return _f()

    def _run(
        self,
        entry: str | None = None,
        local_path: str = ".",
        env: dict | None = None,
        running_extra_volume: Mapping = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[str, int]:
        """
        在 Docker 环境中运行代码
        
        Parameters:
            entry: 入口点
            local_path: 本地路径
            env: 环境变量
            running_extra_volume: 运行时额外卷映射
            **kwargs: 其他参数
            
        Returns:
            tuple[str, int]: 标准输出和退出码
        """
        if env is None:
            env = {}
        env["PYTHONWARNINGS"] = "ignore"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"
        client = get_docker_client()

        volumes = {}
        if local_path is not None:
            local_path = os.path.abspath(local_path)
            volumes[local_path] = {"bind": self.conf.mount_path, "mode": "rw"}

        if self.conf.extra_volumes is not None:
            for lp, rp in self.conf.extra_volumes.items():
                volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}
            cache_path = "/tmp/sample" if "/sample/" in "".join(self.conf.extra_volumes.keys()) else "/tmp/full"
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            volumes[cache_path] = {"bind": T("scenarios.data_science.share:scen.cache_path").r(), "mode": "rw"}
        for lp, rp in running_extra_volume.items():
            volumes[lp] = rp if isinstance(rp, dict) else {"bind": rp, "mode": self.conf.extra_volume_mode}

        volumes = normalize_volumes(cast(dict[str, str | dict[str, str]], volumes), self.conf.mount_path)

        log_output = ""
        container: docker.models.containers.Container | None = None  # type: ignore[no-any-unimported]

        try:
            container = client.containers.run(
                image=self.conf.image,
                command=entry,
                volumes=volumes,
                environment=env,
                detach=True,
                working_dir=self.conf.mount_path,
                # auto_remove=True, # 删除太快可能导致无法获取日志
                network=self.conf.network,
                shm_size=self.conf.shm_size,
                mem_limit=self.conf.mem_limit,  # 设置内存限制
                cpu_count=self.conf.cpu_count,  # 设置 CPU 限制
                **self._gpu_kwargs(client),
            )
            assert container is not None  # 确保容器创建成功
            logs = container.logs(stream=True)
            print(Rule("[bold green]Docker 日志开始[/bold green]", style="dark_orange"))
            table = Table(title="运行信息", show_header=False)
            table.add_column("键", style="bold cyan")
            table.add_column("值", style="bold magenta")
            table.add_row("镜像", self.conf.image)
            table.add_row("容器 ID", container.id)
            table.add_row("容器名称", container.name)
            table.add_row("入口点", entry)
            table.add_row("环境", "\n".join(f"{k}:{v}" for k, v in env.items()))
            table.add_row("卷", "\n".join(f"{k}:\n  {v}" for k, v in volumes.items()))
            print(table)
            for log in logs:
                decoded_log = log.strip().decode()
                Console().print(decoded_log, markup=False)
                log_output += decoded_log + "\n"
            exit_status = container.wait()["StatusCode"]
            print(Rule("[bold green]Docker 日志结束[/bold green]", style="dark_orange"))
            return log_output, exit_status
        except docker.errors.ContainerError as e:
            raise RuntimeError(f"运行容器时出错: {e}")
        except docker.errors.ImageNotFound:
            raise RuntimeError("未找到 Docker 镜像。")
        except docker.errors.APIError as e:
            raise RuntimeError(f"运行容器时出错: {e}")
        finally:
            cleanup_container(container)


class QTDockerEnv(DockerEnv):
    """
    Qlib Torch Docker 环境
    
    用于 Qlib Torch Docker 环境的实现
    """

    def __init__(self, conf: DockerConf = QlibDockerConf()):
        """
        初始化 Qlib Torch Docker 环境
        
        Parameters:
            conf: Docker 配置对象
        """
        super().__init__(conf)

    def prepare(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        下载镜像和数据（如果不存在）
        """
        super().prepare()
        qlib_data_path = next(iter(self.conf.extra_volumes.keys()))
        if not (Path(qlib_data_path) / "qlib_data" / "cn_data").exists():
            logger.info("正在下载数据!")
            cmd = "python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --interval 1d --delete_old False"
            self.check_output(entry=cmd)
        else:
            logger.info("数据已存在。跳过下载。")


class KGDockerEnv(DockerEnv):
    """
    Kaggle 竞赛 Docker 环境
    
    用于 Kaggle 竞赛 Docker 环境的实现
    """

    def __init__(self, competition: str | None = None, conf: DockerConf = KGDockerConf()):
        """
        初始化 Kaggle 竞赛 Docker 环境
        
        Parameters:
            competition: 竞赛名称
            conf: Docker 配置对象
        """
        super().__init__(conf)


class MLEBDockerEnv(DockerEnv):
    """
    MLEBench Docker 环境
    
    用于 MLEBench Docker 环境的实现
    """

    def __init__(self, conf: DockerConf = MLEBDockerConf()):
        """
        初始化 MLEBench Docker 环境
        
        Parameters:
            conf: Docker 配置对象
        """
        super().__init__(conf)