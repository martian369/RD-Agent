"""
RD-Agent Streamlit UI 应用

这是一个基于 Streamlit 的可视化界面，用于展示 RD-Agent 的运行日志和实验结果。
主要功能包括：
1. 显示实验场景描述
2. 展示研究假设和反馈
3. 可视化指标变化
4. 显示代码演化过程
5. 提供任务完成情况分析

作者：RD-Agent 团队
"""

import argparse
import re
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from importlib.resources import files as rfiles
from pathlib import Path
from typing import Callable, Type

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit import session_state as state
from streamlit_theme import st_theme

from rdagent.components.coder.factor_coder.evaluators import FactorSingleFeedback
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.components.coder.model_coder.evaluators import ModelSingleFeedback
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.core.scenario import Scenario
from rdagent.log.base import Message
from rdagent.log.storage import FileStorage
from rdagent.log.ui.qlib_report_figure import report_figure
from rdagent.scenarios.general_model.scenario import GeneralModelScenario
from rdagent.scenarios.kaggle.experiment.scenario import KGScenario
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.factor_from_report_experiment import (
    QlibFactorFromReportScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import (
    QlibModelExperiment,
    QlibModelScenario,
)
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario

# 设置Streamlit页面配置
st.set_page_config(layout="wide", page_title="RD-Agent", page_icon="🎓", initial_sidebar_state="expanded")


# 获取命令行参数
parser = argparse.ArgumentParser(description="RD-Agent Streamlit App")
parser.add_argument("--log_dir", type=str, help="日志目录路径")
parser.add_argument("--debug", action="store_true", help="启用调试模式")
args = parser.parse_args()
if args.log_dir:
    main_log_path = Path(args.log_dir)
    if not main_log_path.exists():
        st.error(f"日志目录 `{main_log_path}` 不存在!")
        st.stop()
else:
    main_log_path = None


# Qlib选定的指标
QLIB_SELECTED_METRICS = [
    "IC",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_with_cost.max_drawdown",
]

# 相似的场景类型
SIMILAR_SCENARIOS = (
    QlibModelScenario,
    QlibFactorScenario,
    QlibFactorFromReportScenario,
    QlibQuantScenario,
    KGScenario,
)


def filter_log_folders(main_log_path):
    """
    过滤并返回相对于主日志路径的日志文件夹
    
    该函数用于筛选指定目录下的所有子目录，并按名称排序返回相对路径列表。
    主要用于在UI中显示可用的日志目录供用户选择。
    
    Parameters:
        main_log_path: 主日志路径
        
    Returns:
        list: 排序后的日志文件夹列表
    """
    folders = [folder.relative_to(main_log_path) for folder in main_log_path.iterdir() if folder.is_dir()]
    folders = sorted(folders, key=lambda x: x.name)
    return folders


# 初始化Streamlit会话状态
# 会话状态用于在用户与应用交互时保持数据
if "log_path" not in state:
    if main_log_path:
        state.log_path = filter_log_folders(main_log_path)[0]
    else:
        state.log_path = None
        st.toast(":red[**请设置日志路径!**]", icon="⚠️")

if "scenario" not in state:
    state.scenario = None

if "fs" not in state:
    state.fs = None

if "msgs" not in state:
    state.msgs = defaultdict(lambda: defaultdict(list))

if "last_msg" not in state:
    state.last_msg = None

if "current_tags" not in state:
    state.current_tags = []

if "lround" not in state:
    state.lround = 0  # RD循环轮次

if "erounds" not in state:
    state.erounds = defaultdict(int)  # 每个RD循环中的演化轮次

if "e_decisions" not in state:
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))

# 摘要信息
if "hypotheses" not in state:
    # 每个RD循环中的假设
    state.hypotheses = defaultdict(None)

if "h_decisions" not in state:
    state.h_decisions = defaultdict(bool)

if "metric_series" not in state:
    state.metric_series = []

if "all_metric_series" not in state:
    state.all_metric_series = []

# 因子任务基线
if "alpha_baseline_metrics" not in state:
    state.alpha_baseline_metrics = None

# 排除标签和类型
if "excluded_tags" not in state:
    state.excluded_tags = ["llm_messages"]  # 默认排除llm_messages标签

if "excluded_types" not in state:
    state.excluded_types = []


def should_display(msg: Message):
    """
    判断消息是否应该显示
    
    Parameters:
        msg: 消息对象
        
    Returns:
        bool: 是否应该显示
    """
    for t in state.excluded_tags + ["debug_tpl", "debug_llm"]:
        if t in msg.tag.split("."):
            return False

    if type(msg.content).__name__ in state.excluded_types:
        return False

    return True


def get_msgs_until(end_func: Callable[[Message], bool] = lambda _: True):
    """
    获取消息直到满足结束条件
    
    Parameters:
        end_func: 结束条件函数
    """
    if state.fs:
        while True:
            try:
                msg = next(state.fs)
                if should_display(msg):
                    tags = msg.tag.split(".")
                    if "hypothesis generation" in msg.tag:
                        state.lround += 1

                    # 新场景生成这些标签，旧版本UI没有这些标签
                    msg.tag = re.sub(r"\.evo_loop_\d+", "", msg.tag)
                    msg.tag = re.sub(r"Loop_\d+\.[^.]+", "", msg.tag)
                    msg.tag = re.sub(r"\.\.", ".", msg.tag)

                    # 移除旧的冗余标签
                    msg.tag = re.sub(r"init\.", "", msg.tag)
                    msg.tag = re.sub(r"r\.", "", msg.tag)
                    msg.tag = re.sub(r"d\.", "", msg.tag)
                    msg.tag = re.sub(r"ef\.", "", msg.tag)

                    msg.tag = msg.tag.strip(".")

                    if "evolving code" not in state.current_tags and "evolving code" in tags:
                        state.erounds[state.lround] += 1

                    state.current_tags = tags
                    state.last_msg = msg

                    # 更新摘要信息
                    if "runner result" in tags:
                        # 因子基线实验指标
                        if (
                            isinstance(state.scenario, (QlibFactorScenario, QlibQuantScenario))
                            and state.alpha_baseline_metrics is None
                        ):
                            try:
                                sms = msg.content.based_experiments[0].result
                            except AttributeError:
                                sms = msg.content.based_experiments[0].__dict__["result"]
                            sms = sms.loc[QLIB_SELECTED_METRICS]
                            sms.name = "Alpha Base"
                            state.alpha_baseline_metrics = sms

                        if state.lround == 1 and len(msg.content.based_experiments) > 0:
                            try:
                                sms = msg.content.based_experiments[-1].result
                            except AttributeError:
                                sms = msg.content.based_experiments[-1].__dict__["result"]
                            if sms is not None:
                                if isinstance(
                                    state.scenario,
                                    (
                                        QlibModelScenario,
                                        QlibFactorFromReportScenario,
                                        QlibFactorScenario,
                                        QlibQuantScenario,
                                    ),
                                ):
                                    sms_all = sms
                                    sms = sms.loc[QLIB_SELECTED_METRICS]
                                sms.name = f"Baseline"
                                state.metric_series.append(sms)
                                state.all_metric_series.append(sms_all)

                        # 通用指标
                        try:
                            sms = msg.content.result
                        except AttributeError:
                            sms = msg.content.__dict__["result"]
                        if isinstance(
                            state.scenario,
                            (
                                QlibModelScenario,
                                QlibFactorFromReportScenario,
                                QlibFactorScenario,
                                QlibQuantScenario,
                            ),
                        ):
                            sms_all = sms
                            sms = sms.loc[QLIB_SELECTED_METRICS]

                        sms.name = f"Round {state.lround}"
                        sms_all.name = f"Round {state.lround}"
                        state.metric_series.append(sms)
                        state.all_metric_series.append(sms_all)
                    elif "hypothesis generation" in tags:
                        state.hypotheses[state.lround] = msg.content
                    elif "evolving code" in tags:
                        msg.content = [i for i in msg.content if i]
                    elif "evolving feedback" in tags:
                        total_len = len(msg.content)
                        none_num = total_len - len(msg.content)
                        right_num = 0
                        for wsf in msg.content:
                            if wsf.final_decision:
                                right_num += 1
                        wrong_num = len(msg.content) - right_num
                        state.e_decisions[state.lround][state.erounds[state.lround]] = (
                            right_num,
                            wrong_num,
                            none_num,
                        )
                    elif "feedback" in tags and isinstance(msg.content, HypothesisFeedback):
                        state.h_decisions[state.lround] = msg.content.decision

                    state.msgs[state.lround][msg.tag].append(msg)

                    # 停止获取日志
                    if end_func(msg):
                        break
            except StopIteration:
                st.toast(":red[**没有更多日志可显示!**]", icon="🛑")
                break


def refresh(same_trace: bool = False):
    """
    刷新日志数据
    
    Parameters:
        same_trace: 是否使用相同轨迹
    """
    if state.log_path is None:
        st.toast(":red[**请设置日志路径!**]", icon="⚠️")
        return

    if main_log_path:
        state.fs = FileStorage(main_log_path / state.log_path).iter_msg()
    else:
        state.fs = FileStorage(state.log_path).iter_msg()

    # 检测场景
    if not same_trace:
        get_msgs_until(lambda m: isinstance(m.content, Scenario))
        if state.last_msg is None or not isinstance(state.last_msg.content, Scenario):
            st.write(state.msgs)
            st.toast(":red[**未检测到场景信息**]", icon="❗")
            state.scenario = None
        else:
            state.scenario = state.last_msg.content
            st.toast(f":green[**检测到场景信息**] *{type(state.scenario).__name__}*", icon="✅")

    state.msgs = defaultdict(lambda: defaultdict(list))
    state.lround = 0
    state.erounds = defaultdict(int)
    state.e_decisions = defaultdict(lambda: defaultdict(tuple))
    state.hypotheses = defaultdict(None)
    state.h_decisions = defaultdict(bool)
    state.metric_series = []
    state.all_metric_series = []
    state.last_msg = None
    state.current_tags = []
    state.alpha_baseline_metrics = None


def evolving_feedback_window(wsf: FactorSingleFeedback | ModelSingleFeedback):
    """
    显示演化反馈窗口
    
    Parameters:
        wsf: 因子或模型的单一反馈对象
    """
    if isinstance(wsf, FactorSingleFeedback):
        ffc, efc, cfc, vfc = st.tabs(
            ["**最终反馈🏁**", "执行反馈🖥️", "代码反馈📄", "数值反馈🔢"]
        )
        with ffc:
            st.markdown(wsf.final_feedback)
        with efc:
            st.code(wsf.execution_feedback, language="log")
        with cfc:
            st.markdown(wsf.code_feedback)
        with vfc:
            st.markdown(wsf.value_feedback)
    elif isinstance(wsf, ModelSingleFeedback):
        ffc, efc, cfc, msfc, vfc = st.tabs(
            [
                "**最终反馈🏁**",
                "执行反馈🖥️",
                "代码反馈📄",
                "模型形状反馈📐",
                "数值反馈🔢",
            ]
        )
        with ffc:
            st.markdown(wsf.final_feedback)
        with efc:
            st.code(wsf.execution_feedback, language="log")
        with cfc:
            st.markdown(wsf.code_feedback)
        with msfc:
            st.markdown(wsf.shape_feedback)
        with vfc:
            st.markdown(wsf.value_feedback)


def display_hypotheses(hypotheses: dict[int, Hypothesis], decisions: dict[int, bool], success_only: bool = False):
    """
    显示假设信息
    
    Parameters:
        hypotheses: 假设字典
        decisions: 决策字典
        success_only: 是否只显示成功的假设
    """
    name_dict = {
        "hypothesis": "RD-Agent提出的假设⬇️",
        "concise_justification": "因为原因⬇️",
        "concise_observation": "基于观察⬇️",
        "concise_knowledge": "实践后获得的知识⬇️",
    }
    if success_only:
        shd = {k: v.__dict__ for k, v in hypotheses.items() if decisions[k]}
    else:
        shd = {k: v.__dict__ for k, v in hypotheses.items()}
    df = pd.DataFrame(shd).T

    if "concise_observation" in df.columns and "concise_justification" in df.columns:
        df["concise_observation"], df["concise_justification"] = df["concise_justification"], df["concise_observation"]
        df.rename(
            columns={"concise_observation": "concise_justification", "concise_justification": "concise_observation"},
            inplace=True,
        )
    if "reason" in df.columns:
        df.drop(["reason"], axis=1, inplace=True)
    if "concise_reason" in df.columns:
        df.drop(["concise_reason"], axis=1, inplace=True)

    df.columns = df.columns.map(lambda x: name_dict.get(x, x))
    for col in list(df.columns):
        if all([value is None for value in df[col]]):
            df.drop([col], axis=1, inplace=True)

    def style_rows(row):
        if decisions[row.name]:
            return ["color: green;"] * len(row)
        return [""] * len(row)

    def style_columns(col):
        if col.name != name_dict.get("hypothesis", "hypothesis"):
            return ["font-style: italic;"] * len(col)
        return ["font-weight: bold;"] * len(col)

    # st.dataframe(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0))
    st.markdown(df.style.apply(style_rows, axis=1).apply(style_columns, axis=0).to_html(), unsafe_allow_html=True)


def metrics_window(df: pd.DataFrame, R: int, C: int, *, height: int = 300, colors: list[str] = None):
    """
    显示指标窗口
    
    Parameters:
        df: 数据框
        R: 行数
        C: 列数
        height: 图表高度
        colors: 颜色列表
    """
    fig = make_subplots(rows=R, cols=C, subplot_titles=df.columns)

    def hypothesis_hover_text(h: Hypothesis, d: bool = False):
        color = "green" if d else "black"
        text = h.hypothesis
        lines = textwrap.wrap(text, width=60)
        return f"<span style='color: {color};'>{'<br>'.join(lines)}</span>"

    hover_texts = [
        hypothesis_hover_text(state.hypotheses[int(i[6:])], state.h_decisions[int(i[6:])])
        for i in df.index
        if i != "Alpha Base" and i != "Baseline"
    ]
    if state.alpha_baseline_metrics is not None:
        hover_texts = ["Baseline"] + hover_texts
    for ci, col in enumerate(df.columns):
        row = ci // C + 1
        col_num = ci % C + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                mode="lines+markers",
                connectgaps=True,
                marker=dict(size=10, color=colors[ci]) if colors else dict(size=10),
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<br><br><span style='color: black'>%{x} 值:</span> <span style='color: blue'>%{y}</span><extra></extra>",
            ),
            row=row,
            col=col_num,
        )
    fig.update_layout(showlegend=False, height=height)

    if state.alpha_baseline_metrics is not None:
        for i in range(1, R + 1):  # 行
            for j in range(1, C + 1):  # 列
                fig.update_xaxes(
                    tickvals=[df.index[0]] + list(df.index[1:]),
                    ticktext=[f'<span style="color:blue; font-weight:bold">{df.index[0]}</span>'] + list(df.index[1:]),
                    row=i,
                    col=j,
                )
    st.plotly_chart(fig)

    from io import BytesIO

    buffer = BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    st.download_button(label="下载指标 (csv)", data=buffer, file_name="metrics.csv", mime="text/csv")


def summary_window():
    """
    显示摘要窗口
    """
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        st.header("摘要📊", divider="rainbow", anchor="_summary")
        if state.lround == 0:
            return
        with st.container():
            # TODO: not fixed height
            with st.container():
                bc, cc = st.columns([2, 2], vertical_alignment="center")
                with bc:
                    st.subheader("指标📈", anchor="_metrics")
                with cc:
                    show_true_only = st.toggle("成功的假设", value=False)

            # hypotheses_c, chart_c = st.columns([2, 3])
            chart_c = st.container()
            hypotheses_c = st.container()

            with hypotheses_c:
                st.subheader("假设🏅", anchor="_hypotheses")
                display_hypotheses(state.hypotheses, state.h_decisions, show_true_only)

            with chart_c:
                if isinstance(state.scenario, QlibFactorScenario) and state.alpha_baseline_metrics is not None:
                    df = pd.DataFrame([state.alpha_baseline_metrics] + state.metric_series[1:])
                elif isinstance(state.scenario, QlibQuantScenario) and state.alpha_baseline_metrics is not None:
                    df = pd.DataFrame([state.alpha_baseline_metrics] + state.metric_series[1:])
                else:
                    df = pd.DataFrame(state.metric_series)
                if show_true_only and len(state.hypotheses) >= len(state.metric_series):
                    if state.alpha_baseline_metrics is not None:
                        selected = ["Alpha Base"] + [
                            i for i in df.index if i == "Baseline" or (i.startswith("Round ") and state.h_decisions[int(i[6:])])
                        ]
                    else:
                        selected = [i for i in df.index if i == "Baseline" or (i.startswith("Round ") and state.h_decisions[int(i[6:])])]
                    df = df.loc[selected]
                if df.shape[0] == 1:
                    st.table(df.iloc[0])
                elif df.shape[0] > 1:
                    if df.shape[1] == 1:
                        fig = px.line(df, x=df.index, y=df.columns, markers=True)
                        fig.update_layout(xaxis_title="循环轮次", yaxis_title=None)
                        st.plotly_chart(fig)
                    else:
                        metrics_window(df, 1, 4, height=300, colors=["red", "blue", "orange", "green"])

    elif isinstance(state.scenario, GeneralModelScenario):
        with st.container(border=True):
            st.subheader("摘要📊", divider="rainbow", anchor="_summary")
            if len(state.msgs[state.lround]["evolving code"]) > 0:
                # pass
                ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[state.lround]["evolving code"][-1].content
                # 所有任务

                tab_names = [
                    w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name
                    for w in ws
                ]
                for j in range(len(ws)):
                    if state.msgs[state.lround]["evolving feedback"][-1].content[j].final_decision:
                        tab_names[j] += "✔️"
                    else:
                        tab_names[j] += "❌"

                wtabs = st.tabs(tab_names)
                for j, w in enumerate(ws):
                    with wtabs[j]:
                        # 演化代码
                        for k, v in w.file_dict.items():
                            with st.expander(f":green[`{k}`]", expanded=False):
                                st.code(v, language="python")

                        # 演化反馈
                        evolving_feedback_window(state.msgs[state.lround]["evolving feedback"][-1].content[j])


def tabs_hint():
    """
    显示标签页提示
    """
    st.markdown(
        "<p style='font-size: small; color: #888888;'>您可以使用 ⬅️ ➡️ 或按住Shift并用鼠标滚轮🖱️在标签页间导航。</p>",
        unsafe_allow_html=True,
    )


def tasks_window(tasks: list[FactorTask | ModelTask]):
    """
    显示任务窗口
    
    Parameters:
        tasks: 任务列表
    """
    if isinstance(tasks[0], FactorTask):
        st.markdown("**因子任务🚩**")
        tnames = [f.factor_name for f in tasks]
        if sum(len(tn) for tn in tnames) > 100:
            tabs_hint()
        tabs = st.tabs(tnames)
        for i, ft in enumerate(tasks):
            with tabs[i]:
                # st.markdown(f"**因子名称**: {ft.factor_name}")
                st.markdown(f"**描述**: {ft.factor_description}")
                st.latex("公式")
                st.latex(ft.factor_formulation)

                mks = "| 变量 | 描述 |\n| --- | --- |\n"
                if isinstance(ft.variables, dict):
                    for v, d in ft.variables.items():
                        mks += f"| ${v}$ | {d} |\n"
                    st.markdown(mks)

    elif isinstance(tasks[0], ModelTask):
        st.markdown("**模型任务🚩**")
        tnames = [m.name for m in tasks]
        if sum(len(tn) for tn in tnames) > 100:
            tabs_hint()
        tabs = st.tabs(tnames)
        for i, mt in enumerate(tasks):
            with tabs[i]:
                # st.markdown(f"**模型名称**: {mt.name}")
                st.markdown(f"**模型类型**: {mt.model_type}")
                st.markdown(f"**描述**: {mt.description}")
                st.latex("公式")
                st.latex(mt.formulation)

                mks = "| 变量 | 描述 |\n| --- | --- |\n"
                if mt.variables:
                    for v, d in mt.variables.items():
                        mks += f"| ${v}$ | {d} |\n"
                    st.markdown(mks)
                st.markdown(f"**训练参数**: {mt.training_hyperparameters}")


def research_window():
    """
    显示研究窗口
    """
    with st.container(border=True):
        title = "研究🔍" if isinstance(state.scenario, SIMILAR_SCENARIOS) else "研究🔍 (阅读器)"
        st.subheader(title, divider="blue", anchor="_research")
        if isinstance(state.scenario, SIMILAR_SCENARIOS):
            # pdf图像
            if pim := state.msgs[round]["load_pdf_screenshot"]:
                for i in range(min(2, len(pim))):
                    st.image(pim[i].content, use_container_width=True)

            # 假设
            if hg := state.msgs[round]["hypothesis generation"]:
                st.markdown("**假设💡**")  # 🧠
                h: Hypothesis = hg[0].content
                st.markdown(
                    f"""
- **假设**: {h.hypothesis}
- **原因**: {h.reason}"""
                )

            if eg := state.msgs[round]["experiment generation"]:
                tasks_window(eg[0].content)

        elif isinstance(state.scenario, GeneralModelScenario):
            # pdf图像
            c1, c2 = st.columns([2, 3])
            with c1:
                if pim := state.msgs[0]["pdf_image"]:
                    for i in range(len(pim)):
                        st.image(pim[i].content, use_container_width=True)

            # 加载的模型实验
            with c2:
                if mem := state.msgs[0]["load_experiment"]:
                    me: QlibModelExperiment = mem[0].content
                    tasks_window(me.sub_tasks)


def feedback_window():
    """
    显示反馈窗口
    """
    # st.write(round)
    # # 检查指标序列是否存在且有匹配的轮次
    # if state.all_metric_series:
    #     for metric in state.all_metric_series:
    #         if metric.name == f"Round {round}":
    #             # 选择特定的含成本指标
    #             selected_metrics_with_cost = {
    #                 'IC': float(f"{metric['IC']:.4f}"),
    #                 'ICIR': float(f"{metric['ICIR']:.4f}"),
    #                 'Rank IC': float(f"{metric['Rank IC']:.4f}"),
    #                 'Rank ICIR': float(f"{metric['Rank ICIR']:.4f}"),
    #                 'ARR': float(f"{metric['1day.excess_return_with_cost.annualized_return']:.4f}"),
    #                 'IR': float(f"{metric['1day.excess_return_with_cost.information_ratio']:.4f}"),
    #                 'MDD': float(f"{metric['1day.excess_return_with_cost.max_drawdown']:.4f}"),
    #                 'Sharpe': float(f"{metric['1day.excess_return_with_cost.annualized_return'] / abs(metric['1day.excess_return_with_cost.max_drawdown']):.4f}")
    #             }
    #             st.write("含成本指标:")
    #             st.write(pd.Series(selected_metrics_with_cost))

    #             # 选择特定的不含成本指标
    #             selected_metrics_without_cost = {
    #                 'IC': float(f"{metric['IC']:.4f}"),
    #                 'ICIR': float(f"{metric['ICIR']:.4f}"),
    #                 'Rank IC': float(f"{metric['Rank IC']:.4f}"),
    #                 'Rank ICIR': float(f"{metric['Rank ICIR']:.4f}"),
    #                 'ARR': float(f"{metric['1day.excess_return_without_cost.annualized_return']:.4f}"),
    #                 'IR': float(f"{metric['1day.excess_return_without_cost.information_ratio']:.4f}"),
    #                 'MDD': float(f"{metric['1day.excess_return_without_cost.max_drawdown']:.4f}"),
    #                 'Sharpe': float(f"{metric['1day.excess_return_without_cost.annualized_return'] / abs(metric['1day.excess_return_without_cost.max_drawdown']):.4f}")
    #             }
    #             st.write("不含成本指标:")
    #             st.write(pd.Series(selected_metrics_without_cost))
    #             break
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        with st.container(border=True):
            st.subheader("反馈📝", divider="orange", anchor="_feedback")

            if state.lround > 0 and isinstance(
                state.scenario,
                (QlibModelScenario, QlibFactorScenario, QlibFactorFromReportScenario, QlibQuantScenario, KGScenario),
            ):
                if fbr := state.msgs[round]["runner result"]:
                    try:
                        st.write("工作空间")
                        st.write(fbr[0].content.experiment_workspace.workspace_path)
                        st.write(fbr[0].content.stdout)
                    except Exception as e:
                        st.error(f"显示工作空间路径时出错: {str(e)}")
                with st.expander("**配置⚙️**", expanded=True):
                    st.markdown(state.scenario.experiment_setting, unsafe_allow_html=True)

            if fb := state.msgs[round]["feedback"]:
                if fbr := state.msgs[round]["Quantitative Backtesting Chart"]:
                    st.markdown("**收益📈**")
                    fig = report_figure(fbr[0].content)
                    st.plotly_chart(fig)
                st.markdown("**假设反馈🔍**")
                h: HypothesisFeedback = fb[0].content
                st.markdown(
                    f"""
- **观察**: {h.observations}
- **假设评估**: {h.hypothesis_evaluation}
- **新假设**: {h.new_hypothesis}
- **决策**: {h.decision}
- **原因**: {h.reason}"""
                )

            if isinstance(state.scenario, KGScenario):
                if fbe := state.msgs[round]["runner result"]:
                    submission_path = fbe[0].content.experiment_workspace.workspace_path / "submission.csv"
                    st.markdown(
                        f":green[**实验工作空间**]: {str(fbe[0].content.experiment_workspace.workspace_path.absolute())}"
                    )
                    try:
                        data = submission_path.read_bytes()
                        st.download_button(
                            label="**下载** submission.csv",
                            data=data,
                            file_name="submission.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.markdown(f":red[**下载按钮错误**]: {e}")


def evolving_window():
    """
    显示演化窗口
    """
    title = "开发🛠️" if isinstance(state.scenario, SIMILAR_SCENARIOS) else "开发🛠️ (演化编码器)"
    st.subheader(title, divider="green", anchor="_development")

    # 演化状态
    if state.erounds[round] > 0:
        st.markdown("**☑️ 演化状态**")
        es = state.e_decisions[round]
        e_status_mks = "".join(f"| {ei} " for ei in range(1, state.erounds[round] + 1)) + "|\n"
        e_status_mks += "|--" * state.erounds[round] + "|\n"
        for ei, estatus in es.items():
            if not estatus:
                estatus = (0, 0, 0)
            e_status_mks += "| " + "🕙<br>" * estatus[2] + "✔️<br>" * estatus[0] + "❌<br>" * estatus[1] + " "
        e_status_mks += "|\n"
        st.markdown(e_status_mks, unsafe_allow_html=True)

    # 演化标签页
    if state.erounds[round] > 0:
        if state.erounds[round] > 1:
            evolving_round = st.radio(
                "**🔄️演化轮次**",
                horizontal=True,
                options=range(1, state.erounds[round] + 1),
                index=state.erounds[round] - 1,
                key="show_eround",
            )
        else:
            evolving_round = 1

        ws: list[FactorFBWorkspace | ModelFBWorkspace] = state.msgs[round]["evolving code"][evolving_round - 1].content
        # 所有任务

        tab_names = [
            w.target_task.factor_name if isinstance(w.target_task, FactorTask) else w.target_task.name for w in ws
        ]
        if len(state.msgs[round]["evolving feedback"]) >= evolving_round:
            for j in range(len(ws)):
                if state.msgs[round]["evolving feedback"][evolving_round - 1].content[j].final_decision:
                    tab_names[j] += "✔️"
                else:
                    tab_names[j] += "❌"
        if sum(len(tn) for tn in tab_names) > 100:
            tabs_hint()
        wtabs = st.tabs(tab_names)
        for j, w in enumerate(ws):
            with wtabs[j]:
                # 演化代码
                st.markdown(f"**工作空间路径**: {w.workspace_path}")
                for k, v in w.file_dict.items():
                    with st.expander(f":green[`{k}`]", expanded=True):
                        st.code(v, language="python")

                # 演化反馈
                if len(state.msgs[round]["evolving feedback"]) >= evolving_round:
                    evolving_feedback_window(state.msgs[round]["evolving feedback"][evolving_round - 1].content[j])


toc = """
## [场景描述📖](#_scenario)
## [摘要📊](#_summary)
- [**指标📈**](#_metrics)
- [**假设🏅**](#_hypotheses)
## [R&D循环♾️](#_rdloops)
- [**研究🔍**](#_research)
- [**开发🛠️**](#_development)
- [**反馈📝**](#_feedback)
"""
if isinstance(state.scenario, GeneralModelScenario):
    toc = """
## [场景描述📖](#_scenario)
### [摘要📊](#_summary)
### [研究🔍](#_research)
### [开发🛠️](#_development)
"""
# 配置侧边栏
with st.sidebar:
    st.markdown("# RD-Agent🤖  [:grey[@GitHub]](https://github.com/microsoft/RD-Agent)")
    st.subheader(":blue[目录]", divider="blue")
    st.markdown(toc)
    st.subheader(":orange[控制面板]", divider="red")

    with st.container(border=True):
        if main_log_path:
            lc1, lc2 = st.columns([1, 2], vertical_alignment="center")
            with lc1:
                st.markdown(":blue[**日志路径**]")
            with lc2:
                manually = st.toggle("手动输入")
            if manually:
                st.text_input("日志路径", key="log_path", on_change=refresh, label_visibility="collapsed")
            else:
                folders = filter_log_folders(main_log_path)
                st.selectbox(f"**从 `{main_log_path}` 中选择**", folders, key="log_path", on_change=refresh)
        else:
            st.text_input(":blue[**日志路径**]", key="log_path", on_change=refresh)

    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        if st.button(":green[**所有循环**]", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: False)
        if st.button("**重置**", use_container_width=True):
            refresh(same_trace=True)
    with c2:
        if st.button(":green[下一循环]", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: "feedback" in m.tag and "evolving feedback" not in m.tag)

        if st.button("下一步", use_container_width=True):
            if not state.fs:
                refresh()
            get_msgs_until(lambda m: "evolving feedback" in m.tag)

    with st.popover(":orange[**配置⚙️**]", use_container_width=True):
        st.multiselect("排除的日志标签", ["llm_messages"], ["llm_messages"], key="excluded_tags")
        st.multiselect("排除的日志类型", ["str", "dict", "list"], ["str"], key="excluded_types")

    if args.debug:
        debug = st.toggle("调试", value=False)

        if debug:
            if st.button("单步运行", use_container_width=True):
                get_msgs_until()
    else:
        debug = False


# 调试信息窗口
if debug:
    with st.expander(":red[**调试信息**]", expanded=True):
        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            st.markdown(
                f"**日志路径**: {state.log_path}\n\n"
                f"**排除标签**: {state.excluded_tags}\n\n"
                f"**排除类型**: {state.excluded_types}\n\n"
                f":blue[**消息ID**]: {sum(sum(len(tmsgs) for tmsgs in rmsgs.values()) for rmsgs in state.msgs.values())}\n\n"
                f":blue[**轮次**]: {state.lround}\n\n"
                f":blue[**演化轮次**]: {state.erounds[state.lround]}\n\n"
            )
        with dcol2:
            if state.last_msg:
                st.write(state.last_msg)
                if isinstance(state.last_msg.content, list):
                    st.write(state.last_msg.content[0])
                elif isinstance(state.last_msg.content, dict):
                    st.write(state.last_msg.content)
                elif not isinstance(state.last_msg.content, str):
                    try:
                        st.write(state.last_msg.content.__dict__)
                    except:
                        st.write(type(state.last_msg.content))

if state.log_path and state.fs is None:
    refresh()

# 主窗口
header_c1, header_c3 = st.columns([1, 6], vertical_alignment="center")
with st.container():
    with header_c1:
        st.image("https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31")
    with header_c3:
        st.markdown(
            """
        <h1>
            RD-Agent:<br>基于LLM的工业数据驱动R&D自主演化代理
        </h1>
        """,
            unsafe_allow_html=True,
        )

# 项目信息
with st.container():
    image_c, scen_c = st.columns([3, 3], vertical_alignment="center")
    with image_c:
        img_path = rfiles("rdagent.log.ui").joinpath("flow.png")
        st.image(str(img_path), use_container_width=True)
    with scen_c:
        st.header("场景描述📖", divider="violet", anchor="_scenario")
        if state.scenario is not None:
            theme = st_theme()
            if theme:
                theme = theme.get("base", "light")
            css = f"""
<style>
    a[href="#_rdloops"], a[href="#_research"], a[href="#_development"], a[href="#_feedback"], a[href="#_scenario"], a[href="#_summary"], a[href="#_hypotheses"], a[href="#_metrics"] {{
        color: {"black" if theme == "light" else "white"};
    }}
</style>
"""
            st.markdown(state.scenario.rich_style_description + css, unsafe_allow_html=True)


def analyze_task_completion():
    """
    分析任务完成情况
    """
    st.header("任务完成情况分析", divider="orange")

    # 存储所有循环结果的字典
    completion_stats = {}

    # 遍历所有循环
    for loop_round in state.msgs.keys():
        if loop_round == 0:  # 跳过初始化循环
            continue

        max_evolving_round = state.erounds[loop_round]
        if max_evolving_round == 0:
            continue

        # 跟踪每轮演化中通过的任务
        tasks_passed_by_round = {}
        cumulative_passed = set()

        # 对于此循环中的每轮演化
        for e_round in range(1, max_evolving_round + 1):
            if len(state.msgs[loop_round]["evolving feedback"]) >= e_round:
                # 获取此演化轮次的反馈
                feedback = state.msgs[loop_round]["evolving feedback"][e_round - 1].content

                # 计算通过的任务并跟踪其索引
                passed_tasks = set()
                for j, task_feedback in enumerate(feedback):
                    if task_feedback.final_decision:
                        passed_tasks.add(j)
                        cumulative_passed.add(j)

                # 存储单轮结果和累积结果
                tasks_passed_by_round[e_round] = {
                    "count": len(passed_tasks),
                    "indices": passed_tasks,
                    "cumulative_count": len(cumulative_passed),
                    "cumulative_indices": cumulative_passed.copy(),
                }

        completion_stats[loop_round] = {
            "total_tasks": len(state.msgs[loop_round]["evolving feedback"][0].content),
            "rounds": tasks_passed_by_round,
            "max_round": max_evolving_round,
        }

    # 显示结果
    if completion_stats:
        # 在顶部添加聚合视图
        st.subheader("🔄 所有循环的聚合完成情况")

        # 创建用于比较的摘要数据
        summary_data = []
        total_tasks_across_loops = 0
        total_passed_r1 = 0
        total_passed_r3 = 0
        total_passed_r5 = 0
        total_passed_r10 = 0
        total_passed_final = 0

        for loop_round, stats in completion_stats.items():
            total_tasks = stats["total_tasks"]
            total_tasks_across_loops += total_tasks

            # 查找特定轮次的数据
            r1_passed = stats["rounds"].get(1, {}).get("cumulative_count", 0)
            total_passed_r1 += r1_passed

            # 对于第3轮，如果不存在确切的3则使用最接近的轮次
            if 3 in stats["rounds"]:
                r3_passed = stats["rounds"][3]["cumulative_count"]
            elif stats["max_round"] >= 3:
                max_r_below_3 = max([r for r in stats["rounds"].keys() if r <= 3])
                r3_passed = stats["rounds"][max_r_below_3]["cumulative_count"]
            else:
                r3_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_r3 += r3_passed

            # 对于第5轮，如果不存在确切的5则使用最接近的轮次
            if 5 in stats["rounds"]:
                r5_passed = stats["rounds"][5]["cumulative_count"]
            elif stats["max_round"] >= 5:
                max_r_below_5 = max([r for r in stats["rounds"].keys() if r <= 5])
                r5_passed = stats["rounds"][max_r_below_5]["cumulative_count"]
            else:
                r5_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_r5 += r5_passed

            # 对于第10轮
            if 10 in stats["rounds"]:
                r10_passed = stats["rounds"][10]["cumulative_count"]
            else:
                r10_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_r10 += r10_passed

            # 最终轮次完成情况
            final_passed = stats["rounds"][stats["max_round"]]["cumulative_count"] if stats["rounds"] else 0
            total_passed_final += final_passed

            # 添加到摘要表
            summary_data.append(
                {
                    "循环": f"循环 {loop_round}",
                    "总任务数": total_tasks,
                    "通过 (第1轮)": (
                        f"{r1_passed}/{total_tasks} ({r1_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                    "通过 (第3轮)": (
                        f"{r3_passed}/{total_tasks} ({r3_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                    "通过 (第5轮)": (
                        f"{r5_passed}/{total_tasks} ({r5_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                    "通过 (最终)": (
                        f"{final_passed}/{total_tasks} ({final_passed/total_tasks:.0%})" if total_tasks > 0 else "N/A"
                    ),
                }
            )

        if total_tasks_across_loops > 0:
            summary_data.append(
                {
                    "循环": "**总计**",
                    "总任务数": total_tasks_across_loops,
                    "通过 (第1轮)": f"**{total_passed_r1}/{total_tasks_across_loops} ({total_passed_r1/total_tasks_across_loops:.0%})**",
                    "通过 (第3轮)": f"**{total_passed_r3}/{total_tasks_across_loops} ({total_passed_r3/total_tasks_across_loops:.0%})**",
                    "通过 (第5轮)": f"**{total_passed_r5}/{total_tasks_across_loops} ({total_passed_r5/total_tasks_across_loops:.0%})**",
                    "通过 (最终)": f"**{total_passed_final}/{total_tasks_across_loops} ({total_passed_final/total_tasks_across_loops:.0%})**",
                }
            )

        st.table(pd.DataFrame(summary_data))

        # 摘要统计
        st.markdown("### 📊 整体完成进度:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="第1轮后",
                value=f"{total_passed_r1/total_tasks_across_loops:.0%}",
                help=f"{total_passed_r1}/{total_tasks_across_loops} 任务",
            )
        with col2:
            st.metric(
                label="第3轮后",
                value=f"{total_passed_r3/total_tasks_across_loops:.0%}",
                delta=f"{(total_passed_r3-total_passed_r1)/total_tasks_across_loops:.0%}",
                help=f"{total_passed_r3}/{total_tasks_across_loops} 任务",
            )
        with col3:
            st.metric(
                label="第5轮后",
                value=f"{total_passed_r5/total_tasks_across_loops:.0%}",
                delta=f"{(total_passed_r5-total_passed_r3)/total_tasks_across_loops:.0%}",
                help=f"{total_passed_r5}/{total_tasks_across_loops} 任务",
            )
        with col4:
            st.metric(
                label="最终完成",
                value=f"{total_passed_final/total_tasks_across_loops:.0%}",
                delta=f"{(total_passed_final-total_passed_r5)/total_tasks_across_loops:.0%}",
                help=f"{total_passed_final}/{total_tasks_across_loops} 任务",
            )

        # 按循环显示详细结果
        st.markdown("---")
        st.subheader("按循环的详细结果")

        for loop_round, stats in completion_stats.items():
            with st.expander(f"循环 {loop_round} 详情"):
                total_tasks = stats["total_tasks"]

                # 创建结果表
                data = []
                for e_round in range(1, min(11, stats["max_round"] + 1)):
                    if e_round in stats["rounds"]:
                        round_data = stats["rounds"][e_round]
                        data.append(
                            {
                                "演化轮次": e_round,
                                "通过任务": f"{round_data['count']}/{total_tasks} ({round_data['count']/total_tasks:.0%})",
                                "累积通过": f"{round_data['cumulative_count']}/{total_tasks} ({round_data['cumulative_count']/total_tasks:.0%})",
                            }
                        )
                    else:
                        data.append({"演化轮次": e_round, "通过任务": "N/A", "累积通过": "N/A"})

                df = pd.DataFrame(data)
                st.table(df)

                st.markdown("### 摘要:")
                if 1 in stats["rounds"]:
                    st.markdown(
                        f"- 第1轮后: **{stats['rounds'][1]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][1]['cumulative_count']/total_tasks:.0%})"
                    )

                if 3 in stats["rounds"]:
                    st.markdown(
                        f"- 第3轮后: **{stats['rounds'][3]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][3]['cumulative_count']/total_tasks:.0%})"
                    )
                elif stats["max_round"] >= 3:
                    max_round_below_3 = max([r for r in stats["rounds"].keys() if r <= 3])
                    st.markdown(
                        f"- 第3轮后: **{stats['rounds'][max_round_below_3]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][max_round_below_3]['cumulative_count']/total_tasks:.0%})"
                    )

                if 5 in stats["rounds"]:
                    st.markdown(
                        f"- 第5轮后: **{stats['rounds'][5]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][5]['cumulative_count']/total_tasks:.0%})"
                    )
                elif stats["max_round"] >= 5:
                    max_round_below_5 = max([r for r in stats["rounds"].keys() if r <= 5])
                    st.markdown(
                        f"- 第5轮后: **{stats['rounds'][max_round_below_5]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][max_round_below_5]['cumulative_count']/total_tasks:.0%})"
                    )

                if 10 in stats["rounds"]:
                    st.markdown(
                        f"- 第10轮后: **{stats['rounds'][10]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][10]['cumulative_count']/total_tasks:.0%})"
                    )
                elif stats["max_round"] >= 1:
                    st.markdown(
                        f"- 最终轮次 ({stats['max_round']}): **{stats['rounds'][stats['max_round']]['cumulative_count']}/{total_tasks}** 任务通过 ({stats['rounds'][stats['max_round']]['cumulative_count']/total_tasks:.0%})"
                    )
    else:
        st.info("没有任务完成数据可用。")


if state.scenario is not None:
    summary_window()
    if st.toggle("显示任务完成情况分析"):
        analyze_task_completion()

    # R&D循环窗口
    if isinstance(state.scenario, SIMILAR_SCENARIOS):
        st.header("R&D循环♾️", divider="rainbow", anchor="_rdloops")
        if len(state.msgs) > 1:
            r_options = list(state.msgs.keys())
            if 0 in r_options:
                r_options.remove(0)
            # 确保r_options不为空且index有效
            if r_options and state.lround - 1 >= 0 and state.lround - 1 < len(r_options):
                round = st.radio("**循环**", horizontal=True, options=r_options, index=state.lround - 1)
            elif r_options:
                # 如果index无效但有选项，使用默认index 0
                round = st.radio("**循环**", horizontal=True, options=r_options, index=0)
            else:
                # 如果没有选项，使用默认值1
                round = 1
        else:
            round = 1

        rf_c, d_c = st.columns([2, 2])
    elif isinstance(state.scenario, GeneralModelScenario):

        rf_c = st.container()
        d_c = st.container()
        round = 0
    else:
        st.error("未知场景!")
        st.stop()

    with rf_c:
        research_window()
        feedback_window()

    with d_c.container(border=True):
        evolving_window()


st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("#### 免责声明")
st.markdown(
    "*此内容由AI生成，可能不完全准确或不是最新的；对于关键事项，请与专业人士核实。*",
    unsafe_allow_html=True,
)