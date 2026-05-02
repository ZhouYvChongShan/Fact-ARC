"""
Fact-ARC Pydantic 请求/响应模型定义

定义所有 API 交互所需的数据模型，包括核查请求/响应、
纠错请求/响应和核心 Fact-ARC 接口的输入输出。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    """核查请求：将模型回答与检索事实进行比对"""
    answer: str = Field(..., description="模型生成的回答文本")
    retrieved_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的事实列表，每项包含 title/snippet/url 等"
    )


class VerifyResponse(BaseModel):
    """核查响应：比对结果"""
    consistent: bool = Field(..., description="回答与事实是否一致")
    confidence: int = Field(
        ..., ge=0, le=100, description="置信度，0-100 之间的整数"
    )
    reason: str = Field(..., description="判断理由")


class CorrectionRequest(BaseModel):
    """纠错请求：要求模型基于事实重新生成回答"""
    user_query: str = Field(..., description="用户的原始问题")
    previous_answer: str = Field(..., description="模型之前的回答")
    retrieved_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的事实列表"
    )


class FactARCRequest(BaseModel):
    """Fact-ARC 核心请求"""
    query: str = Field(..., description="用户查询问题")
    max_loops: Optional[int] = Field(
        default=None,
        description="最大纠错循环次数，不填则使用服务端配置的默认值"
    )


class VerificationRecord(BaseModel):
    """单轮验证/纠错的完整记录"""
    round: int = Field(..., description="循环轮次（从1开始）")
    answer: str = Field(..., description="当前轮次的回答")
    retrieved_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="检索到的事实"
    )
    verify_response: VerifyResponse = Field(
        ..., description="核查模型的比对结果"
    )


class FactARCResponse(BaseModel):
    """Fact-ARC 核心响应"""
    final_answer: str = Field(..., description="最终回答（经过纠错循环后的）")
    correction_loop_count: int = Field(
        ..., description="实际执行的纠错循环次数"
    )
    verification_trail: List[VerificationRecord] = Field(
        default_factory=list,
        description="每轮的比对记录"
    )