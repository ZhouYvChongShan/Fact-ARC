"""
Fact-ARC FastAPI 应用入口

提供 REST API 接口和 CLI 启动脚本。
通过 /v1/chat/completions 端点接收请求，
执行 Fact-ARC 纠错循环并返回结果。
"""

import json
import logging
import os
import sys

from contextlib import asynccontextmanager
from typing import Optional

import jieba
import jieba.posseg as pseg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn

from config import AppConfig, collect_config_interactive, mask_key
from models import FactARCRequest, FactARCResponse
from retriever import BochaRetriever
from verifier import FactVerifier
from loop import FactARCEngine

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 全局实例
# ============================================================
engine: FactARCEngine = None

CONFIG_FILE = os.path.expanduser("~/.factarc_config.json")


# ============================================================
# 关键词提取（基于 jieba 分词 + 词性过滤）
# ============================================================

QUESTION_PATTERNS = {
    '哪些', '哪个', '什么', '怎么', '如何', '为什么', '为何',
    '谁', '哪里', '哪儿', '什么时候', '多少', '几',
    '请问', '想问', '问一下', '麻烦', '帮忙', '帮我', '能否',
    '告诉我', '讲讲', '介绍', '解释', '说一下', '聊一聊',
    '知不知道', '了解', '我想', '我需要', '我要',
    '呢', '吗', '吧', '啊', '呀', '哦', '哈', '嘛',
    '有', '的', '是', '了', '在', '和', '与', '或',
}

KEEP_POS = {
    'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'a', 'eng', 'm', 't',
}


def extract_keywords(query: str, max_len: int = 80) -> str:
    """使用 jieba 分词 + 词性标注提取核心关键词。"""
    if not query or len(query.strip()) <= 3:
        return query.strip()

    words = pseg.cut(query)

    keywords = []
    for word, flag in words:
        word = word.strip()
        if not word:
            continue
        if len(word) <= 1 and not word.isalnum():
            continue
        if word in QUESTION_PATTERNS:
            continue
        if flag in KEEP_POS:
            keywords.append(word)
        elif word.isalnum() and len(word) > 1:
            keywords.append(word)

    result = ' '.join(keywords)

    if not result.strip():
        logger.info(f"jieba 提取关键词为空，使用原查询: '{query[:60]}'")
        result = query

    if len(result) > max_len:
        result = result[:max_len].rsplit(' ', 1)[0]

    logger.info(f"关键词提取: '{query[:60]}...' -> '{result}'")
    return result


# ============================================================
# 配置持久化
# ============================================================
def save_config(config: AppConfig, filepath: str = CONFIG_FILE):
    """保存配置到本地文件。"""
    import base64

    config_dict = {
        "main_base_url": config.main_base_url,
        "main_api_key": base64.b64encode(config.main_api_key.encode()).decode() if config.main_api_key else "",
        "main_model_name": config.main_model_name,
        "verifier_base_url": config.verifier_base_url,
        "verifier_api_key": base64.b64encode(config.verifier_api_key.encode()).decode() if config.verifier_api_key else "",
        "verifier_model_name": config.verifier_model_name,
        "bocha_api_key": base64.b64encode(config.bocha_api_key.encode()).decode() if config.bocha_api_key else "",
        "max_loops": config.max_loops,
        "confidence_threshold": config.confidence_threshold,
    }

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        os.chmod(filepath, 0o600)
        logger.info(f"配置已保存到: {filepath}")
        print(f"  💾 配置已保存到: {filepath}")
        return True
    except Exception as e:
        logger.warning(f"保存配置失败: {e}")
        print(f"  ⚠️  配置保存失败: {e}")
        return False


def load_config(filepath: str = CONFIG_FILE) -> Optional[AppConfig]:
    """从本地文件加载配置。"""
    import base64

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        def decode_key(key: str) -> str:
            if not key:
                return ""
            try:
                return base64.b64decode(key.encode()).decode()
            except Exception:
                return key

        config = AppConfig(
            main_base_url=config_dict.get("main_base_url", "https://api.openai.com/v1"),
            main_api_key=decode_key(config_dict.get("main_api_key", "")),
            main_model_name=config_dict.get("main_model_name", "gpt-3.5-turbo"),
            verifier_base_url=config_dict.get("verifier_base_url", "https://api.openai.com/v1"),
            verifier_api_key=decode_key(config_dict.get("verifier_api_key", "")),
            verifier_model_name=config_dict.get("verifier_model_name", "gpt-3.5-turbo"),
            bocha_api_key=decode_key(config_dict.get("bocha_api_key", "")),
            max_loops=config_dict.get("max_loops", 3),
            confidence_threshold=config_dict.get("confidence_threshold", 90),
        )

        logger.info(f"配置已从 {filepath} 加载")
        return config
    except Exception as e:
        logger.warning(f"加载配置失败: {e}")
        print(f"  ⚠️  配置文件损坏，将重新配置: {e}")
        return None


def create_engine() -> tuple:
    """通过交互式配置收集参数并创建 Fact-ARC 引擎实例。"""
    config = None

    existing_config = load_config()

    if existing_config:
        print("\n" + "=" * 60)
        print("  📁 检测到已保存的配置")
        print("=" * 60)
        print(f"  主模型:     {existing_config.main_model_name}")
        print(f"  主模型 URL: {existing_config.main_base_url}")
        print(f"  主模型密钥: {mask_key(existing_config.main_api_key)}")
        print(f"  核查模型:   {existing_config.verifier_model_name}")
        print(f"  核查 URL:   {existing_config.verifier_base_url}")
        print(f"  核查密钥:   {mask_key(existing_config.verifier_api_key)}")
        print(f"  Bocha 密钥: {mask_key(existing_config.bocha_api_key)}")
        print(f"  最大循环:   {existing_config.max_loops}")
        print(f"  置信阈值:   {existing_config.confidence_threshold}")
        print("=" * 60)

        while True:
            choice = input(
                "\n是否使用已保存的配置？\n"
                "  [y] 是，使用已有配置\n"
                "  [n] 否，重新配置\n"
                "  [d] 删除配置文件并退出\n"
                ">>> "
            ).strip().lower()

            if choice == 'y':
                config = existing_config
                print("  ✅ 使用已保存的配置")
                break
            elif choice == 'n':
                print("  🔄 进入重新配置流程...")
                break
            elif choice == 'd':
                try:
                    os.remove(CONFIG_FILE)
                    print(f"  🗑️  配置文件已删除: {CONFIG_FILE}")
                    print("  退出程序。")
                    sys.exit(0)
                except Exception as e:
                    print(f"  ❌ 删除失败: {e}")
                    sys.exit(1)
            else:
                print("  无效选择，请输入 y / n / d")

    if config is None:
        config = collect_config_interactive()

    print()
    save_choice = input("是否保存配置以便下次使用？(y/n)\n>>> ").strip().lower()
    if save_choice == 'y':
        save_config(config)

    retriever = BochaRetriever(
        api_key=config.bocha_api_key,
        keyword_extractor=extract_keywords,
    )
    verifier = FactVerifier(
        base_url=config.verifier_base_url,
        api_key=config.verifier_api_key,
        model_name=config.verifier_model_name,
    )
    engine_instance = FactARCEngine(
        main_config=config,
        retriever=retriever,
        verifier=verifier,
    )

    return engine_instance, config


# ============================================================
# FastAPI 应用
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fact-ARC 应用启动")
    yield
    logger.info("Fact-ARC 应用关闭")


app = FastAPI(
    title="Fact-ARC",
    description="Fact-based Auto-Regressive Correction - 基于事实比对的自回归纠错系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat/completions", response_model=FactARCResponse)
async def chat_completions(request: FactARCRequest):
    """
    Fact-ARC 核心接口。
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="服务未初始化，请检查启动日志")

    logger.info(
        f"收到请求: query='{request.query[:80]}{'...' if len(request.query) > 80 else ''}', "
        f"max_loops={request.max_loops}"
    )

    try:
        response = await engine.process(
            query=request.query,
            max_loops=request.max_loops,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"处理请求时发生未预期异常: {e}")
        raise HTTPException(status_code=500, detail=f"内部处理错误: {type(e).__name__}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.exception(f"未捕获的异常: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"内部服务器错误: {type(exc).__name__}"})


# ============================================================
# CLI 启动脚本
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀 Fact-ARC 启动中...")
    print("=" * 60)

    try:
        engine, config = create_engine()
    except KeyboardInterrupt:
        print("\n\n配置已取消，退出。")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"启动配置失败: {e}")
        print(f"\n❌ 配置失败: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  ✅ Fact-ARC 已启动")
    print(f"  📡 端口: http://localhost:1010")
    print(f"  📖 API 文档: http://localhost:1010/docs")
    print(f"  🔧 健康检查: http://localhost:1010/health")
    print("=" * 60)
    print(f"  主模型: {config.main_model_name}")
    print(f"  主模型 URL: {config.main_base_url}")
    print(f"  主模型密钥: {mask_key(config.main_api_key)}")
    print(f"  核查模型: {config.verifier_model_name}")
    print(f"  核查 URL: {config.verifier_base_url}")
    print(f"  核查密钥: {mask_key(config.verifier_api_key)}")
    print(f"  Bocha 密钥: {mask_key(config.bocha_api_key)}")
    print(f"  最大循环: {config.max_loops}")
    print(f"  置信阈值: {config.confidence_threshold}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=1010, log_level="info")