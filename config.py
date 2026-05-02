"""
Fact-ARC 配置管理模块

包含配置数据类和交互式配置收集器。
启动时通过命令行交互收集所有必要参数。
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """应用程序全局配置"""
    # 主模型配置
    main_base_url: str
    main_api_key: str
    main_model_name: str

    # 核查模型配置
    verifier_base_url: str
    verifier_api_key: str
    verifier_model_name: str

    # Bocha API 配置
    bocha_api_key: str

    # 循环控制
    max_loops: int = 3
    confidence_threshold: int = 90


def mask_key(key: str) -> str:
    """脱敏显示密钥，仅显示前4位和后4位"""
    if not key:
        return "(未设置)"
    if len(key) <= 8:
        return key[:2] + "****" + key[-2:]
    return key[:4] + "****" + key[-4:]


async def fetch_models(base_url: str, api_key: str = "") -> List[str]:
    """
    调用 /v1/models 端点获取可用模型列表。

    Args:
        base_url: API 端点 base URL
        api_key: API 密钥（可选）

    Returns:
        模型 ID 列表，失败时返回空列表
    """
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            models = []
            if isinstance(data, dict) and "data" in data:
                for item in data["data"]:
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        models.append(item["id"])
                    elif isinstance(item, str):
                        models.append(item)
            logger.info(f"从 {base_url} 获取到 {len(models)} 个模型")
            return models
    except httpx.HTTPStatusError as e:
        logger.error(f"获取模型列表失败，HTTP {e.response.status_code}: {e}")
    except httpx.RequestError as e:
        logger.error(f"获取模型列表网络错误: {e}")
    except Exception as e:
        logger.error(f"获取模型列表异常: {e}")

    return []


def _choose_model(base_url: str, api_key: str, prompt: str) -> str:
    """
    从模型列表中让用户选择一个模型。

    Args:
        base_url: API 端点 base URL
        api_key: API 密钥
        prompt: 提示信息

    Returns:
        用户选择的模型名称
    """
    print(f"\n正在从 {base_url} 获取模型列表...")
    models = asyncio.run(fetch_models(base_url, api_key))

    if models:
        print("已获取模型列表")
        print(f"{prompt}")
        for i, m in enumerate(models):
            print(f"  [{i + 1}] {m}")
        while True:
            choice = input("请输入序号选择模型: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx]
                print(f"  无效选择，请输入 1-{len(models)} 之间的数字")
            except ValueError:
                print("  请输入有效的数字")
    else:
        print("  无法获取模型列表，请手动输入模型名称")
        model_name = input("请输入模型名称: ").strip()
        if model_name:
            return model_name
        print("  未输入模型名称，使用默认值 'gpt-3.5-turbo'")
        return "gpt-3.5-turbo"


def collect_config_interactive() -> AppConfig:
    """
    交互式收集配置信息。

    收集顺序：
    1. 主模型 base_url
    2. 主模型 api_key
    3. 主模型选择
    4. Bocha API key
    5. 核查模型 base_url
    6. 核查模型是否需要密钥
    7. 核查模型选择

    Returns:
        AppConfig 实例
    """
    print("=" * 60)
    print("  Fact-ARC 配置向导")
    print("=" * 60)

    # 1. 主模型 base_url
    main_base_url = input(
        "\n请输入你的主模型的baseURL（仅支持openai格式）\n"
        ">>> "
    ).strip()
    if not main_base_url:
        main_base_url = "https://api.openai.com/v1"
        print(f"  使用默认: {main_base_url}")

    # 2. 主模型 api_key
    main_api_key = input(
        "\n请输入你的api密钥\n"
        ">>> "
    ).strip()

    # 3. 主模型选择
    print("\n请选择模型")
    main_model_name = _choose_model(
        main_base_url, main_api_key,
        "请选择你的主模型的id"
    )

    # 4. Bocha API key
    bocha_api_key = input(
        "\n请输入联网搜索平台密钥（仅支持bocha）\n"
        ">>> "
    ).strip()

    # 5. 核查模型 base_url
    verifier_base_url = input(
        "\n请输入你的核查模型的baseURL（仅支持openai格式）\n"
        ">>> "
    ).strip()
    if not verifier_base_url:
        verifier_base_url = "https://api.openai.com/v1"
        print(f"  使用默认: {verifier_base_url}")

    # 6. 是否需要密钥
    need_key = input("\n是否需要密钥？(y/n)\n>>> ").strip().lower()
    verifier_api_key = ""
    if need_key == "y":
        verifier_api_key = input("请输入核查模型api密钥: ").strip()

    # 7. 核查模型选择
    verifier_model_name = _choose_model(
        verifier_base_url, verifier_api_key,
        "请选择你的核查模型的id"
    )

    config = AppConfig(
        main_base_url=main_base_url,
        main_api_key=main_api_key,
        main_model_name=main_model_name,
        verifier_base_url=verifier_base_url,
        verifier_api_key=verifier_api_key,
        verifier_model_name=verifier_model_name,
        bocha_api_key=bocha_api_key,
        max_loops=3,
        confidence_threshold=90,
    )

    # 打印配置摘要
    print("\n" + "=" * 60)
    print("  配置摘要")
    print("=" * 60)
    print(f"  主模型:     {main_model_name}")
    print(f"  主模型 URL: {main_base_url}")
    print(f"  主模型密钥: {mask_key(main_api_key)}")
    print(f"  核查模型:   {verifier_model_name}")
    print(f"  核查 URL:   {verifier_base_url}")
    print(f"  核查密钥:   {mask_key(verifier_api_key)}")
    print(f"  Bocha 密钥: {mask_key(bocha_api_key)}")
    print(f"  最大循环:   {config.max_loops}")
    print(f"  置信阈值:   {config.confidence_threshold}")
    print("=" * 60)

    logger.info("配置收集完成")
    return config