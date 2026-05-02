"""
Fact-ARC 核心纠错循环逻辑

实现"生成-检索-比对-纠错"的闭环验证架构。
通过循环调用主模型、检索器和核查模型，逐步修正回答中的事实错误。
"""

import asyncio
import json
import logging

from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from openai import AsyncOpenAI

from config import AppConfig
from models import (
    FactARCResponse,
    VerificationRecord,
)
from retriever import BochaRetriever
from verifier import FactVerifier

logger = logging.getLogger(__name__)


class FactARCEngine:
    """
    Fact-ARC 核心引擎。

    实现"生成-检索-比对-纠错"的主循环：
    1. 主模型生成初始回答
    2. 检索外部事实（自动提取关键词优化检索）
    3. 核查模型比对一致性
    4. 不一致时触发纠错，重新生成回答
    5. 达到置信度阈值或最大循环次数时退出

    检索关键词由 retriever 内部的关键词提取器自动处理，
    无需在引擎层面重复提取。
    """

    def __init__(
        self,
        main_config: AppConfig,
        retriever: BochaRetriever,
        verifier: FactVerifier,
    ):
        """
        初始化引擎。

        Args:
            main_config: 应用程序配置
            retriever: Bocha 检索器实例（内含关键词提取器）
            verifier: 核查模型实例
        """
        self.config = main_config
        self.retriever = retriever
        self.verifier = verifier

        # 初始化主模型的 OpenAI 客户端
        self.main_client = AsyncOpenAI(
            base_url=main_config.main_base_url,
            api_key=main_config.main_api_key,
        )
        self.main_model = main_config.main_model_name

        logger.info(
            f"FactARCEngine 初始化完成: main_model={self.main_model}, "
            f"max_loops={main_config.max_loops}, "
            f"threshold={main_config.confidence_threshold}"
        )

    async def _generate_initial_answer(self, query: str) -> str:
        """
        调用主模型生成初始回答（无任何检索上下文）。

        Args:
            query: 用户查询

        Returns:
            主模型的初始回答

        Raises:
            HTTPException: 主模型调用失败（含重试后）
        """
        logger.info(f"主模型初始生成: query='{query[:80]}{'...' if len(query) > 80 else ''}'")

        messages = [
            {
                "role": "system",
                "content": "你是一个知识渊博的助手，请根据你的知识直接回答用户的问题。",
            },
            {"role": "user", "content": query},
        ]

        answer = await self._call_main_model_with_retry(messages, "初始生成")
        logger.info(f"主模型初始回答长度: {len(answer)} 字符")
        return answer

    async def _correct_with_context(
        self,
        user_query: str,
        previous_answer: str,
        retrieved_facts: List[Dict[str, Any]],
    ) -> str:
        """
        基于检索事实对之前的回答进行纠正。

        构造包含原始问题、之前回答和检索事实的上下文，
        要求主模型根据事实重新生成正确的回答。

        Args:
            user_query: 用户的原始问题
            previous_answer: 模型之前的回答
            retrieved_facts: 检索到的事实列表

        Returns:
            纠错后的回答
        """
        facts_text = json.dumps(retrieved_facts, ensure_ascii=False, indent=2)

        system_message = (
            "你是一个基于事实回答的助手。用户之前的问题和你的回答以及检索到的事实如下，"
            "请根据事实重新生成正确的回答。"
        )

        user_message = (
            f"【原始问题】: {user_query}\n\n"
            f"【你之前的回答】: {previous_answer}\n\n"
            f"【检索到的事实】: {facts_text}\n\n"
            f"请基于上述检索到的事实内容，修正你之前回答中的错误，"
            f"给出一个准确的回答。如果事实中没有直接相关的信息，"
            f"请根据已有事实进行合理推断并标明。"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        corrected = await self._call_main_model_with_retry(messages, "纠错生成")
        logger.info(f"纠错后回答长度: {len(corrected)} 字符")
        return corrected

    async def _call_main_model_with_retry(
        self, messages: List[Dict[str, str]], context: str
    ) -> str:
        """
        调用主模型，带重试机制（最多2次重试，指数退避）。

        Args:
            messages: 消息列表
            context: 调用上下文描述（用于日志）

        Returns:
            模型的回答文本

        Raises:
            HTTPException: 所有重试均失败时抛出
        """
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.main_client.chat.completions.create(
                    model=self.main_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("主模型返回空内容")
                return content

            except Exception as e:
                last_error = e
                attempt_num = attempt + 1
                logger.warning(
                    f"主模型调用失败 ({context}, 第{attempt_num}次): {e}"
                )

                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"主模型调用最终失败 ({context}), "
                        f"已重试 {max_retries} 次"
                    )

        raise HTTPException(
            status_code=502,
            detail=f"主模型调用失败 ({context}): {str(last_error)}",
        )

    async def process(
        self, query: str, max_loops: Optional[int] = None
    ) -> FactARCResponse:
        """
        执行 Fact-ARC 核心循环。

        流程：
        1. 主模型生成初始回答
        2. 进入循环：
           a. 检索外部事实（retriever 内部自动提取关键词）
           b. 核查模型比对一致性
           c. 记录当前轮次信息
           d. 判断是否达到退出条件
           e. 不满足退出条件时触发纠错
        3. 返回最终结果

        Args:
            query: 用户查询
            max_loops: 最大循环次数（覆盖配置中的默认值）

        Returns:
            FactARCResponse 包含最终回答、循环次数和验证轨迹
        """
        max_iterations = (
            max_loops if max_loops is not None else self.config.max_loops
        )
        loop_count = 0
        current_answer: Optional[str] = None
        trail: List[VerificationRecord] = []

        logger.info(
            f"Fact-ARC 处理开始: query='{query[:80]}{'...' if len(query) > 80 else ''}', "
            f"max_loops={max_iterations}"
        )

        # 步骤1: 主模型生成初始回答
        current_answer = await self._generate_initial_answer(query)

        # 步骤2: 进入核查-纠错循环
        while loop_count < max_iterations:
            round_num = loop_count + 1
            logger.info(f"--- 第 {round_num} 轮开始 ---")

            # 2a. 检索外部事实
            try:
                facts = await self.retriever.search(query)
            except Exception as e:
                logger.error(f"检索失败，使用空事实列表: {e}")
                facts = []

            # 2b. 核查模型比对
            verify_response = await self.verifier.verify(current_answer, facts)

            # 2c. 记录当前轮次
            record = VerificationRecord(
                round=round_num,
                answer=current_answer,
                retrieved_facts=facts,
                verify_response=verify_response,
            )
            trail.append(record)

            logger.info(
                f"第 {round_num} 轮结果: consistent={verify_response.consistent}, "
                f"confidence={verify_response.confidence}, "
                f"facts_count={len(facts)}"
            )

            # 2d. 判断退出条件
            if verify_response.consistent and verify_response.confidence >= self.config.confidence_threshold:
                logger.info(
                    f"达到置信度阈值，退出循环 (confidence={verify_response.confidence})"
                )
                break

            if loop_count >= max_iterations - 1:
                logger.info(f"已达到最大循环次数 ({max_iterations})，退出循环")
                break

            # 2e. 触发纠错
            logger.info(f"触发第 {round_num} 轮纠错")
            current_answer = await self._correct_with_context(
                user_query=query,
                previous_answer=current_answer,
                retrieved_facts=facts,
            )
            loop_count += 1

        # 步骤3: 构建并返回响应
        response = FactARCResponse(
            final_answer=current_answer,
            correction_loop_count=loop_count,
            verification_trail=trail,
        )

        logger.info(
            f"Fact-ARC 处理完成: total_rounds={len(trail)}, "
            f"correction_loops={loop_count}, "
            f"final_answer_len={len(current_answer)}"
        )

        return response