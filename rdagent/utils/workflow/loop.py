"""
这是一个尝试存储/恢复/追溯工作流会话的类


附注:
- 最初，我想用Python生成器以更通用的方式实现。
  但是，Python生成器不可pickle化（dill也不支持pickle）

"""

import asyncio
import concurrent.futures
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import psutil
from tqdm.auto import tqdm

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.utils.workflow.tracking import WorkflowTracker


class LoopMeta(type):
    """循环元类，用于收集所有步骤方法"""

    @staticmethod
    def _get_steps(bases: tuple[type, ...]) -> list[str]:
        """
        递归地从基类中获取所有`steps`并将它们组合成一个列表。

        参数:
            bases (tuple): 基类元组。

        返回:
            List[Callable]: 从所有基类组合的步骤列表。
        """
        steps = []
        for base in bases:
            for step in LoopMeta._get_steps(base.__bases__) + getattr(base, "steps", []):
                if step not in steps and step not in ["load", "dump"]:  # 防止用户覆盖load/dump方法
                    steps.append(step)
        return steps

    def __new__(mcs, clsname: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> Any:
        """
        创建一个新类，其中包含从基类和当前类组合的步骤。

        参数:
            clsname (str): 新类的名称。
            bases (tuple): 基类。
            attrs (dict): 新类的属性。

        返回:
            LoopMeta: LoopMeta的新实例。
        """
        steps = LoopMeta._get_steps(bases)  # 所有父类的基类
        for name, attr in attrs.items():
            if not name.startswith("_") and callable(attr) and not isinstance(attr, type):
                # NOTE: `not isinstance(attr, type)` 试图排除类类型属性
                if name not in steps and name not in ["load", "dump"]:  # 防止用户覆盖load/dump方法
                    # NOTE: 如果我们在子类中覆盖了步骤
                    # 那么它就不是新步骤。所以我们跳过它。
                    steps.append(name)
        attrs["steps"] = steps
        return super().__new__(mcs, clsname, bases, attrs)


@dataclass
class LoopTrace:
    """循环跟踪信息数据类"""
    start: datetime  # 跟踪的开始时间
    end: datetime  # 跟踪的结束时间
    step_idx: int
    # TODO: 更多关于跟踪的信息


class LoopBase:
    """
    循环基类，用于管理工作流循环
    
    假设:
    - 最后一步负责记录信息!!!!

    未解决问题:
    - 当`force_subproc`为True时的全局变量同步
        - 计时器
    """

    steps: list[str]  # 要处理的步骤列表
    loop_trace: dict[int, list[LoopTrace]]  # 循环跟踪信息

    # 可以定义一个错误列表，当发生这些错误时将跳过当前循环
    skip_loop_error: tuple[type[BaseException], ...] = ()
    # 可以定义一个错误列表，当发生这些错误时将撤回当前循环
    withdraw_loop_error: tuple[
        type[BaseException], ...
    ] = ()

    EXCEPTION_KEY = "_EXCEPTION"  # 异常键
    LOOP_IDX_KEY = "_LOOP_IDX"    # 循环索引键
    SENTINEL = -1                 # 哨兵值，用于标记队列结束

    _pbar: tqdm  # 进度条实例

    class LoopTerminationError(Exception):
        """当循环条件指示循环应终止时引发的异常"""

    class LoopResumeError(Exception):
        """当循环条件指示循环应停止所有协程并恢复时引发的异常"""

    def __init__(self) -> None:
        """初始化循环基类"""
        # 进度控制
        self.loop_idx: int = 0  # 当前循环索引 / 下一个要启动的循环索引
        self.step_idx: defaultdict[int, int] = defaultdict(int)  # 从循环索引到下一步索引的字典
        self.queue: asyncio.Queue[Any] = asyncio.Queue()

        # 为所有循环存储步骤结果的嵌套字典，将存储以下信息:
        # - loop_prev_out[loop_index][step_name]: 步骤函数的输出
        # - loop_prev_out[loop_index][<特殊键如LOOP_IDX_KEY或EXCEPTION_KEY>]: 特殊键
        self.loop_prev_out: dict[int, dict[str, Any]] = defaultdict(dict)
        self.loop_trace = defaultdict(list[LoopTrace])  # 键是循环编号
        self.session_folder = Path(LOG_SETTINGS.trace_path) / "__session__"
        self.timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        self.tracker = WorkflowTracker(self)  # 使用此LoopBase实例初始化跟踪器

        # 进度控制
        self.loop_n: Optional[int] = None  # 剩余循环次数
        self.step_n: Optional[int] = None  # 剩余步骤次数

        self.semaphores: dict[str, asyncio.Semaphore] = {}  # 信号量字典

    def get_unfinished_loop_cnt(self, next_loop: int) -> int:
        """
        获取未完成的循环数量
        
        参数:
            next_loop (int): 下一个循环索引
            
        返回:
            int: 未完成的循环数量
        """
        n = 0
        for li in range(next_loop):
            if self.step_idx[li] < len(self.steps):  # 未完成的循环
                n += 1
        return n

    def get_semaphore(self, step_name: str) -> asyncio.Semaphore:
        """
        获取步骤的信号量，用于控制并发执行
        
        参数:
            step_name (str): 步骤名称
            
        返回:
            asyncio.Semaphore: 信号量对象
        """
        if isinstance(limit := RD_AGENT_SETTINGS.step_semaphore, dict):
            limit = limit.get(step_name, 1)  # 如果未指定，默认为1

        # NOTE:
        # (1) 我们假设record步骤总是最后一个修改全局环境的步骤，
        #     所以我们将限制设置为1以避免竞争条件
        # (2) 因为我们支持(-1,)作为局部选择；所以很难对齐a) `feedback`中的比较目标
        #     和b) `record`中的父节点；所以我们防止`feedback`和`record`中的并行性以避免不一致
        if step_name in ("record", "feedback"):
            limit = 1

        if step_name not in self.semaphores:
            self.semaphores[step_name] = asyncio.Semaphore(limit)
        return self.semaphores[step_name]

    @property
    def pbar(self) -> tqdm:
        """进度条属性，如果不存在则初始化"""
        if getattr(self, "_pbar", None) is None:
            self._pbar = tqdm(total=len(self.steps), desc="工作流进度", unit="步")
        return self._pbar

    def close_pbar(self) -> None:
        """关闭进度条"""
        if getattr(self, "_pbar", None) is not None:
            self._pbar.close()
            del self._pbar

    def _check_exit_conditions_on_step(self, loop_id: Optional[int] = None, step_id: Optional[int] = None) -> None:
        """
        检查循环是否应继续或终止。

        异常:
        ------
        LoopTerminationException
            当条件指示循环应终止时
        """
        # 检查步骤计数限制
        if self.step_n is not None:
            if self.step_n <= 0:
                raise self.LoopTerminationError("已达到步骤计数限制")
            self.step_n -= 1

        # 检查计时器超时
        if self.timer.started:
            if self.timer.is_timeout():
                logger.warning("超时，退出循环。")
                raise self.LoopTerminationError("计时器超时")
            else:
                logger.info(f"计时器剩余时间: {self.timer.remain_time()}")

    async def _run_step(self, li: int, force_subproc: bool = False) -> None:
        """
        执行工作流中的单个步骤（下一个未运行的步骤）（异步版本，支持强制子进程选项）。

        参数:
        ----------
        li : int
            循环索引

        force_subproc : bool
            是否强制步骤在asyncio中的子进程中运行

        返回:
        -------
        Any
            步骤函数的结果
        """
        si = self.step_idx[li]
        name = self.steps[si]

        async with self.get_semaphore(name):

            logger.info(f"开始循环 {li}, 步骤 {si}: {name}")
            self.tracker.log_workflow_state()

            with logger.tag(f"Loop_{li}.{name}"):
                start = datetime.now(timezone.utc)
                func: Callable[..., Any] = cast(Callable[..., Any], getattr(self, name))

                next_step_idx = si + 1
                step_forward = True
                # NOTE: 每个步骤都知道当前的循环索引
                # 在调用步骤函数之前设置它非常重要！
                self.loop_prev_out[li][self.LOOP_IDX_KEY] = li

                try:
                    # 使用当前循环的输出调用函数，如果需要则等待协程或使用ProcessPoolExecutor执行同步函数
                    if force_subproc:
                        curr_loop = asyncio.get_running_loop()
                        with concurrent.futures.ProcessPoolExecutor() as pool:
                            result = await curr_loop.run_in_executor(pool, func, self.loop_prev_out[li])
                    else:
                        # 自动确定是运行异步还是同步
                        if asyncio.iscoroutinefunction(func):
                            result = await func(self.loop_prev_out[li])
                        else:
                            # 默认: 直接运行同步函数
                            result = func(self.loop_prev_out[li])
                    # 将结果存储在嵌套字典中
                    self.loop_prev_out[li][name] = result
                except Exception as e:
                    if isinstance(e, self.skip_loop_error):
                        logger.warning(f"由于 {e} 跳过循环 {li}")
                        # 跳转到最后一步（假设最后一步用于记录）
                        next_step_idx = len(self.steps) - 1
                        self.loop_prev_out[li][name] = None
                        self.loop_prev_out[li][self.EXCEPTION_KEY] = e
                    elif isinstance(e, self.withdraw_loop_error):
                        logger.warning(f"由于 {e} 撤回循环 {li}")
                        # 返回到上一个循环
                        self.withdraw_loop(li)
                        step_forward = False

                        msg = "我们已重置循环实例，停止所有例程并恢复。"
                        raise self.LoopResumeError(msg) from e
                    else:
                        raise  # 重新抛出未处理的异常
                finally:
                    # 无论执行成功与否，都必须完成以下步骤

                    # 记录跟踪信息
                    end = datetime.now(timezone.utc)
                    self.loop_trace[li].append(LoopTrace(start, end, step_idx=si))
                    logger.log_object(
                        {
                            "start_time": start,
                            "end_time": end,
                        },
                        tag="time_info",
                    )
                    if step_forward:
                        # 增加步骤索引
                        self.step_idx[li] = next_step_idx

                        # 更新进度条
                        current_step = self.step_idx[li]
                        self.pbar.n = current_step
                        next_step = self.step_idx[li] % len(self.steps)
                        self.pbar.set_postfix(
                            loop_index=li + next_step_idx // len(self.steps),
                            step_index=next_step,
                            step_name=self.steps[next_step],
                        )

                        # 完成步骤后保存快照;
                        # 1) 必须在更新step_idx之后，这样加载快照时才会在正确的步骤上。
                        # 2) 只有在步骤前进时才保存，撤回不值得保存。
                        if name in self.loop_prev_out[li]:
                            # 3) 只有在步骤成功执行时才转储（这样我们在加载会话时就不必重新执行该步骤）
                            self.dump(self.session_folder / f"{li}" / f"{si}_{name}")

                        self._check_exit_conditions_on_step(loop_id=li, step_id=si)
                    else:
                        logger.warning(f"循环 {li} 的步骤前进 {si} 被跳过。")

    async def kickoff_loop(self) -> None:
        """
        启动循环，将循环放入队列中等待执行
        """
        while True:
            li = self.loop_idx

            # 根据循环限制退出
            # 检查是否设置了循环次数限制
            if self.loop_n is not None:
                # 如果剩余循环次数小于等于0，说明已达到循环次数限制
                if self.loop_n <= 0:
                    # 向队列中添加哨兵值，通知所有执行循环的协程退出
                    # 添加的哨兵值数量等于最大并行数，确保每个执行协程都能收到退出信号
                    for _ in range(RD_AGENT_SETTINGS.get_max_parallel()):
                        self.queue.put_nowait(self.SENTINEL)
                    # 退出循环
                    break
                # 减少剩余循环次数
                self.loop_n -= 1

            # NOTE:
            # 尽力启动第一步；第一步总是ExpGen；
            # 它有权决定何时停止产生新实验
            if self.step_idx[li] == 0:
                # 假设第一步是ExpGen
                # 只有在从未启动过ExpGen时才启动它
                await self._run_step(li)
            self.queue.put_nowait(li)  # 循环`li`已启动，等待工作器获取
            self.loop_idx += 1
            await asyncio.sleep(0)

    async def execute_loop(self) -> None:
        """
        执行循环中的步骤
        """
        while True:
            # 1) 获取要继续的循环`li`
            li = await self.queue.get()
            if li == self.SENTINEL:
                break
            # 2) 运行未完成的步骤
            while self.step_idx[li] < len(self.steps):
                if self.step_idx[li] == len(self.steps) - 1:
                    # NOTE: 假设最后一步是record，它会很快并影响全局环境
                    # 如果是最后一步，直接运行()
                    await self._run_step(li)
                else:
                    # 等待步骤；并行运行在此发生！
                    # 只有当我们有多个进程时才触发子进程。
                    await self._run_step(li, force_subproc=RD_AGENT_SETTINGS.is_force_subproc())

    async def run(self, step_n: int | None = None, loop_n: int | None = None, all_duration: str | None = None) -> None:
        """
        运行工作流循环。

        参数:
        ----------
        step_n: int | None
            要运行的步骤数；如果当前循环不完整，它将被计为第一个完成的循环
            `None` 表示永远运行直到错误或KeyboardInterrupt
        loop_n: int | None
            要运行的循环数；如果当前循环不完整，它将被计为第一个完成的循环
            `None` 表示永远运行直到错误或KeyboardInterrupt
        all_duration : str | None
            运行的最大持续时间，格式为计时器接受的格式
        """
        # 如果提供了持续时间，则初始化计时器
        if all_duration is not None and not self.timer.started:
            self.timer.reset(all_duration=all_duration)

        if step_n is not None:
            self.step_n = step_n
        if loop_n is not None:
            self.loop_n = loop_n

        # 重启时清空队列
        while not self.queue.empty():
            self.queue.get_nowait()
        self.loop_idx = (
            0  # 如果我们重新运行循环，应该将循环索引重置为0，以确保每个循环都正确启动
        )

        tasks: list[asyncio.Task] = []
        while True:
            try:
                # 运行一个kickoff_loop和execute_loop
                tasks = [
                    asyncio.create_task(t)
                    for t in [
                        self.kickoff_loop(),
                        *[self.execute_loop() for _ in range(RD_AGENT_SETTINGS.get_max_parallel())],
                    ]
                ]
                await asyncio.gather(*tasks)
                break
            except self.LoopResumeError as e:
                logger.warning(f"停止所有例程并恢复循环: {e}")
                self.loop_idx = 0
            except self.LoopTerminationError as e:
                logger.warning(f"达到停止条件并停止循环: {e}")
                kill_subprocesses()  # NOTE: 基于协程的工作流无法自动停止子进程。
                break
            finally:
                # 在恢复所有循环或退出之前取消所有先前的任务
                for t in tasks:
                    t.cancel()
                self.close_pbar()

    def withdraw_loop(self, loop_idx: int) -> None:
        """
        撤回循环，加载上一个循环的状态
        
        参数:
            loop_idx (int): 要撤回的循环索引
        """
        prev_session_dir = self.session_folder / str(loop_idx - 1)
        prev_path = min(
            (p for p in prev_session_dir.glob("*_*") if p.is_file()),
            key=lambda item: int(item.name.split("_", 1)[0]),
            default=None,
        )
        if prev_path:
            loaded = type(self).load(
                prev_path,
                checkout=True,
                replace_timer=True,
            )
            logger.info(f"从 {prev_path} 加载上一个会话")
            # 覆盖当前实例状态
            self.__dict__ = loaded.__dict__
        else:
            logger.error(f"在 {prev_session_dir} 找不到上一个转储，无法撤回循环 {loop_idx}")
            raise

    def dump(self, path: str | Path) -> None:
        """
        将当前状态转储到文件
        
        参数:
            path (str | Path): 转储文件路径
        """
        if RD_Agent_TIMER_wrapper.timer.started:
            RD_Agent_TIMER_wrapper.timer.update_remain_time()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def truncate_session_folder(self, li: int, si: int) -> None:
        """
        通过删除给定循环索引(li)和步骤索引(si)之后的所有会话对象来清除会话文件夹。
        
        参数:
            li (int): 循环索引
            si (int): 步骤索引
        """
        # 清除li之后的会话文件夹
        for sf in self.session_folder.iterdir():
            if sf.is_dir() and int(sf.name) > li:
                for file in sf.iterdir():
                    file.unlink()
                sf.rmdir()

        # 清除li中的步骤会话对象
        final_loop_session_folder = self.session_folder / str(li)
        for step_session in final_loop_session_folder.glob("*_*"):
            if step_session.is_file():
                step_id = int(step_session.name.split("_", 1)[0])
                if step_id > si:
                    step_session.unlink()

    @classmethod
    def load(
        cls,
        path: str | Path,
        checkout: bool | Path | str = False,
        replace_timer: bool = True,
    ) -> "LoopBase":
        """
        从给定路径加载会话。
        
        参数:
        ----------
        path : str | Path
            会话文件的路径。
        checkout : bool | Path | str
            如果为True，新循环将使用现有文件夹并清除给定路径对应会话之后的日志。
            如果为False，新循环将使用现有文件夹但保留给定路径对应会话之后的日志。
            如果提供了路径（或类似Path的str），新循环将保存到该路径，保持原始路径不变。
        replace_timer : bool
            如果加载了会话，确定是否用session.timer替换计时器。
            默认为True，表示会话计时器将被当前计时器替换。
            如果为False，会话计时器将不会被替换。
            
        返回:
        -------
        LoopBase
            具有加载会话的LoopBase实例。
        """
        path = Path(path)
        # 如果路径是目录，加载最新的会话
        if path.is_dir():
            if path.name != "__session__":
                path = path / "__session__"

            if not path.exists():
                raise FileNotFoundError(f"在 {path} 中找不到会话文件")

            # 按递增顺序迭代转储步骤
            files = sorted(path.glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))
            path = files[-1]
            logger.info(f"从 {path} 加载最新会话")
        with path.open("rb") as f:
            session = cast(LoopBase, pickle.load(f))

        # 设置会话文件夹
        if checkout:
            if checkout is True:
                logger.set_storages_path(session.session_folder.parent)
                max_loop = max(session.loop_trace.keys())

                # 截断最大循环后的日志存储
                session.truncate_session_folder(max_loop, len(session.loop_trace[max_loop]) - 1)
                logger.truncate_storages(session.loop_trace[max_loop][-1].end)
            else:
                checkout = Path(checkout)
                checkout.mkdir(parents=True, exist_ok=True)
                session.session_folder = checkout / "__session__"
                logger.set_storages_path(checkout)

        if session.timer.started:
            if replace_timer:
                RD_Agent_TIMER_wrapper.replace_timer(session.timer)
                RD_Agent_TIMER_wrapper.timer.restart_by_remain_time()
            else:
                # 使用默认计时器替换会话计时器
                session.timer = RD_Agent_TIMER_wrapper.timer

        return session

    def __getstate__(self) -> dict[str, Any]:
        """
        获取对象状态，用于pickle序列化
        
        返回:
            dict: 对象状态字典
        """
        res = {}
        for k, v in self.__dict__.items():
            if k not in ["queue", "semaphores", "_pbar"]:
                res[k] = v
        return res

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        设置对象状态，用于pickle反序列化
        
        参数:
            state (dict): 对象状态字典
        """
        self.__dict__.update(state)
        self.queue = asyncio.Queue()
        self.semaphores = {}


def kill_subprocesses() -> None:
    """
    由于工作流基于协程的特性，主进程的事件循环无法
    停止由`curr_loop.run_in_executor`启动的所有子进程。所以我们需要手动杀死它们。
    否则，子进程将在后台继续运行，主进程会一直等待。
    """
    current_proc = psutil.Process(os.getpid())
    for child in current_proc.children(recursive=True):
        try:
            print(f"终止子进程 PID {child.pid} ({child.name()})")
            child.terminate()
        except Exception as ex:
            print(f"无法终止子进程 {child.pid}: {ex}")
    print("已完成终止子进程。然后强制杀死仍然存活的子进程。")
    _, alive = psutil.wait_procs(current_proc.children(recursive=True), timeout=3)
    for p in alive:
        try:
            print(f"杀死仍然存活的子进程 PID {p.pid} ({p.name()})")
            p.kill()
        except Exception as ex:
            print(f"无法杀死子进程 {p.pid}: {ex}")
    print("已完成杀死子进程。")