from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field

from rdagent.core.conf import ExtendedBaseSettings


class LLMSettings(ExtendedBaseSettings):
    # ===== 核心模型配置 =====
    backend: str = "rdagent.oai.backend.LiteLLMAPIBackend"
    """LLM 后端类型。默认使用 LiteLLM 后端，支持多种 LLM 提供商"""

    chat_model: str = "qwen3-max-preview"
    """聊天模型名称。推荐: qwen3-max-preview(通义千问), gpt-4o(OpenAI), deepseek/deepseek-chat(DeepSeek)"""

    embedding_model: str = "text-embedding-v4"
    """嵌入模型名称。推荐: text-embedding-v4(通义千问), text-embedding-3-small(OpenAI)"""

    # ===== 推理和响应配置 =====
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    """推理努力程度。适用于支持的模型(o1系列)。可选: low(快速), medium(平衡), high(深度思考)"""

    enable_response_schema: bool = True
    """启用响应格式校验。推荐: True。注意:不是所有模型都支持此功能"""
    # Whether to enable response_schema in chat models. may not work for models that do not support it.

    # ===== 响应处理配置 =====
    reasoning_think_rm: bool = True
    """是否移除思考标签。推荐: True。某些模型(o1系列)会返回 <think>...</think> 标签，设为 True 可移除这些内容"""
    """
    Some LLMs include <think>...</think> tags in their responses, which can interfere with the main output.
    Set reasoning_think_rm to True to remove any <think>...</think> content from responses.
    """

    # TODO: most of the settings are only used on deprec.DeprecBackend.
    # So they should move the settings to that folder.

    # ===== 日志和云配置 =====
    log_llm_chat_content: bool = True
    """是否记录 LLM 对话内容。推荐: True(调试时), False(生产环境保护隐私)"""

    # 以下 Azure 配置已废弃，请使用 chat_use_azure 和 embedding_use_azure
    use_azure: bool = Field(default=False, deprecated=True)
    """已废弃。请使用 chat_use_azure 和 embedding_use_azure"""

    chat_use_azure: bool = False
    """聊天模型是否使用 Azure OpenAI。推荐: False(除非使用 Azure)"""

    embedding_use_azure: bool = False
    """嵌入模型是否使用 Azure OpenAI。推荐: False(除非使用 Azure)"""

    # ===== Azure 认证配置 =====
    chat_use_azure_token_provider: bool = False
    """聊天模型是否使用 Azure Token Provider 认证。推荐: False(除非使用 Azure 托管身份)"""

    embedding_use_azure_token_provider: bool = False
    """嵌入模型是否使用 Azure Token Provider 认证。推荐: False(除非使用 Azure 托管身份)"""

    managed_identity_client_id: str | None = None
    """托管身份客户端 ID。仅在使用 Azure 托管身份时需要"""

    # ===== 重试和超时配置 =====
    max_retry: int = 10
    """最大重试次数。推荐: 3-10。网络不稳定时可适当增加"""

    retry_wait_seconds: int = 1
    """重试等待时间(秒)。推荐: 1-3秒。避免过于频繁的请求"""
    # ===== 缓存配置 =====
    dump_chat_cache: bool = False
    """是否导出聊天缓存。推荐: False(除非调试需要)"""

    use_chat_cache: bool = False
    """是否使用聊天缓存。推荐: True(节省成本), False(需要实时响应)"""

    dump_embedding_cache: bool = False
    """是否导出嵌入缓存。推荐: False(除非调试需要)"""

    use_embedding_cache: bool = False
    """是否使用嵌入缓存。推荐: True(嵌入相对稳定，缓存效果好)"""

    prompt_cache_path: str = str(Path.cwd() / "prompt_cache.db")
    """缓存数据库路径。推荐: 使用默认值或自定义路径"""

    max_past_message_include: int = 10
    """最大历史消息数量。推荐: 5-20。太少会失去上下文，太多会增加成本"""

    timeout_fail_limit: int = 10
    """超时失败限制。推荐: 5-10。连续超时这么多次后停止重试"""

    violation_fail_limit: int = 1
    """违规失败限制。推荐: 1-3。内容违规时的重试次数"""

    # ===== 缓存高级配置 =====
    use_auto_chat_cache_seed_gen: bool = False
    """是否自动生成缓存种子。推荐: False。启用后相同问题在同一轮才会命中缓存"""
    """
    `_create_chat_completion_inner_function` provides a feature to pass in a seed to affect the cache hash key
    We want to enable a auto seed generator to get different default seed for `_create_chat_completion_inner_function`
    if seed is not given.
    So the cache will only not miss you ask the same question on same round.
    """
    init_chat_cache_seed: int = 42
    """初始缓存种子。推荐: 使用默认值 42。影响缓存哈希键的生成"""

    # ===== 聊天模型参数配置 =====
    openai_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    """通用 API 密钥。推荐: 通过环境变量 OPENAI_API_KEY 设置"""

    chat_openai_api_key: str | None = None
    """聊天模型专用 API 密钥。可选。如果不设置则使用 openai_api_key"""

    chat_openai_base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    """聊天模型 API 基地址。推荐: 根据提供商设置(如 https://api.openai.com/v1)"""

    chat_azure_api_base: str = ""
    """Azure OpenAI 聊天模型端点。仅在使用 Azure 时需要"""

    chat_azure_api_version: str = ""
    """Azure OpenAI API 版本。推荐: 2024-02-15-preview 或更新版本"""

    chat_max_tokens: int | None = None
    """最大输出 token 数。推荐: None(使用模型默认值)或 2048-4096"""

    chat_temperature: float = 0.5
    """温度参数(0-1)。推荐: 0.1-0.3(理性任务), 0.7-0.9(创意任务)"""

    chat_stream: bool = True
    """是否使用流式输出。推荐: True(更好的用户体验)"""

    chat_seed: int | None = None
    """随机种子。可选。设置后可获得更稳定的输出结果"""

    chat_frequency_penalty: float = 0.0
    """频率惩罚(-2 到 2)。推荐: 0.0-0.2。减少重复词汇"""

    chat_presence_penalty: float = 0.0
    """存在惩罚(-2 到 2)。推荐: 0.0-0.2。鼓励话题多样性"""

    chat_token_limit: int = (
        100000  # 100000 is the maximum limit of gpt4, which might increase in the future version of gpt
    )
    """聊天 token 限制。推荐: 根据模型设置(GPT-4: 8k-32k, Claude: 100k+)"""

    default_system_prompt: str = "You are an AI assistant who helps to answer user's questions."
    """默认系统提示词。可根据具体任务自定义"""

    system_prompt_role: str = "system"
    """系统提示词角色。推荐: 'system'。o1 模型不支持 system 角色，可设为 'user'"""

    # ===== 嵌入模型配置 =====
    embedding_openai_api_key: str = ""
    """嵌入模型 API 密钥。推荐: 通过环境变量设置"""

    embedding_openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    """嵌入模型 API 基地址。推荐: 根据提供商设置"""

    embedding_azure_api_base: str = ""
    """Azure OpenAI 嵌入模型端点。仅在使用 Azure 时需要"""

    embedding_azure_api_version: str = ""
    """Azure OpenAI 嵌入模型 API 版本。推荐: 2024-02-15-preview"""

    embedding_max_str_num: int = 50
    """嵌入最大字符串数量。推荐: 50-100。批量处理时的数量限制"""

    embedding_max_length: int = 8192
    """嵌入模型最大 token 长度。推荐: 8192(默认), 32000(Qwen)。用作未知模型的后备值"""

    # ===== 本地 Llama2 配置(已过时) =====
    use_llama2: bool = False
    """是否使用本地 Llama2。不推荐: 已过时，建议使用云端 API"""

    llama2_ckpt_dir: str = "Llama-2-7b-chat"
    """Llama2 模型文件目录。仅在使用本地 Llama2 时需要"""

    llama2_tokenizer_path: str = "Llama-2-7b-chat/tokenizer.model"
    """Llama2 分词器路径。仅在使用本地 Llama2 时需要"""

    llams2_max_batch_size: int = 8
    """Llama2 最大批处理大小。推荐: 4-16，根据内存调整"""

    # ===== 服务器端点配置(企业级) =====
    use_gcr_endpoint: bool = False
    """是否使用 GCR 端点。仅限企业级部署"""

    gcr_endpoint_type: str = "llama2_70b"
    """GCR 端点类型。可选: llama2_70b, llama3_70b, phi2, phi3_4k, phi3_128k"""

    # 各端点配置(仅限企业级部署)
    llama2_70b_endpoint: str = ""
    """Llama2 70B 模型端点 URL"""
    llama2_70b_endpoint_key: str = ""
    """Llama2 70B 模型端点 API 密钥"""
    llama2_70b_endpoint_deployment: str = ""
    """Llama2 70B 模型部署名称"""

    llama3_70b_endpoint: str = ""
    """Llama3 70B 模型端点 URL"""
    llama3_70b_endpoint_key: str = ""
    """Llama3 70B 模型端点 API 密钥"""
    llama3_70b_endpoint_deployment: str = ""
    """Llama3 70B 模型部署名称"""

    phi2_endpoint: str = ""
    """Phi2 模型端点 URL"""
    phi2_endpoint_key: str = ""
    """Phi2 模型端点 API 密钥"""
    phi2_endpoint_deployment: str = ""
    """Phi2 模型部署名称"""

    phi3_4k_endpoint: str = ""
    """Phi3 4K 模型端点 URL"""
    phi3_4k_endpoint_key: str = ""
    """Phi3 4K 模型端点 API 密钥"""
    phi3_4k_endpoint_deployment: str = ""
    """Phi3 4K 模型部署名称"""

    phi3_128k_endpoint: str = ""
    """Phi3 128K 模型端点 URL"""
    phi3_128k_endpoint_key: str = ""
    """Phi3 128K 模型端点 API 密钥"""
    phi3_128k_endpoint_deployment: str = ""
    """Phi3 128K 模型部署名称"""

    # GCR 端点参数配置
    gcr_endpoint_temperature: float = 0.7
    """GCR 端点温度参数。推荐: 0.1-0.9"""

    gcr_endpoint_top_p: float = 0.9
    """GCR 端点 top_p 参数。推荐: 0.8-0.95"""

    gcr_endpoint_do_sample: bool = False
    """GCR 端点是否采样。推荐: True(创意任务), False(理性任务)"""

    gcr_endpoint_max_token: int = 100
    """GCR 端点最大 token 数。推荐: 1024-4096"""

    # ===== DeepSeek Azure 特殊配置 =====
    chat_use_azure_deepseek: bool = False
    """是否使用 Azure 上的 DeepSeek。仅在特殊情况下使用"""

    chat_azure_deepseek_endpoint: str = ""
    """Azure DeepSeek 端点 URL"""

    chat_azure_deepseek_key: str = ""
    """Azure DeepSeek API 密钥"""

    # ===== 模型映射配置 =====
    chat_model_map: dict[str, dict[str, str]] = {}
    """模型映射配置。高级用法，可为不同标签指定不同模型和参数"""
    # 示例: {"tag1": {"model": "gpt-4", "temperature": "0.7", "max_tokens": "2048"}}"""


LLM_SETTINGS = LLMSettings()
