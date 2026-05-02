"""
Fact-ARC 核查模型比对模块

使用核查模型对主模型回答与检索事实进行一致性比对。
包含完整的内置提示词和降级处理逻辑。
"""

import json
import logging
import re

from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from models import VerifyResponse

logger = logging.getLogger(__name__)

# ============================================================
# 内置系统提示词 - 事实一致性核查专家
# ============================================================
VERIFIER_SYSTEM_PROMPT = """你是一个事实一致性核查专家。你的唯一任务是比较【模型回答】与【参考资料】，判断是否存在事实性矛盾。

判断标准：
- 一致（置信度 ≥ 90）：模型回答的所有事实断言在参考资料中都有明确支持
- 模糊一致（70 ≤ 置信度 < 90）：模型回答的断言在参考资料中部分支持、部分缺失，或依赖推理
- 不确定/无法判定（50 ≤ 置信度 < 70）：参考资料不足以验证模型回答的核心断言
- 不一致（置信度 < 50）：模型回答中有明确与参考资料相悖的事实

特殊规则：
1. 只要模型回答中存在任何与参考资料明确矛盾的事实，必须输出置信度 ≤ 50
2. 区分"明确陈述"与"推测/建议"，推测类内容无需判错
3. 参考资料没提到不等于模型说错，除非模型声称"资料明确支持"
4. 对于无法从参考资料中验证的陈述，降低置信度但不要判定为错误
5. 如果参考资料为空列表[]，说明检索失败，此时如果模型回答是"无法找到相关信息"之类的表述，判定为一致（置信度 ≥ 90）；如果模型编造了具体事实，判定为不一致（置信度 ≤ 40）

输出格式：
你必须只输出一个合法的JSON对象，不要有任何额外的文字、解释、Markdown标记。
JSON格式如下：
{"consistent": true或false, "confidence": 0到100的整数, "reason": "判断理由"}

请直接输出JSON："""


class FactVerifier:
    """
    事实一致性核查器。

    使用独立的核查模型（兼容 OpenAI 格式）对主模型回答
    与检索到的事实进行比对，返回一致性判断结果。
    """

    def __init__(self, base_url: str, api_key: str, model_name: str):
        """
        初始化核查器。

        Args:
            base_url: 核查模型的 API base URL
            api_key: 核查模型的 API 密钥（可为空字符串）
            model_name: 核查模型的名称/ID
        """
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "not-needed",
        )
        self._has_api_key = bool(api_key)
        logger.info(
            f"FactVerifier 初始化完成: model={model_name}, "
            f"base_url={base_url}, has_key={self._has_api_key}"
        )

    async def verify(
        self, answer: str, retrieved_facts: List[Dict[str, Any]]
    ) -> VerifyResponse:
        """
        对模型回答与检索事实进行一致性比对。

        Args:
            answer: 主模型生成的回答文本
            retrieved_facts: 检索到的事实列表

        Returns:
            VerifyResponse 包含一致性判断、置信度和理由
        """
        # 构建用户消息
        facts_json = json.dumps(retrieved_facts, ensure_ascii=False, indent=2)
        user_message = (
            f"【模型回答】：{answer}\n\n"
            f"【参考资料】：{facts_json}"
        )

        logger.info(
            f"核查比对开始: answer_len={len(answer)}, facts_count={len(retrieved_facts)}"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            content = response.choices[0].message.content
            if content is None:
                logger.warning("核查模型返回空内容")
                return self._fallback_empty_response(retrieved_facts)

            # 去除首尾空白
            content = content.strip()

            # 日志打印完整响应（前200字符 + 尾50字符）
            if len(content) > 250:
                logger.info(f"核查模型原始响应(前200): {content[:200]}...(尾50): {content[-50:]}")
            else:
                logger.info(f"核查模型原始响应: {content}")

            result = self._parse_verifier_response(content, retrieved_facts)
            logger.info(
                f"核查结果: consistent={result.consistent}, "
                f"confidence={result.confidence}"
            )
            return result

        except Exception as e:
            logger.error(f"核查模型调用失败: {e}")
            # 降级：信任主模型
            return VerifyResponse(
                consistent=True,
                confidence=60,
                reason=f"核查模型不可用，跳过验证 ({type(e).__name__})",
            )

    def _fallback_empty_response(self, retrieved_facts: List[Dict[str, Any]]) -> VerifyResponse:
        """核查模型返回空内容时的兜底处理"""
        if not retrieved_facts:
            return VerifyResponse(
                consistent=True,
                confidence=70,
                reason="核查模型返回空内容且无参考资料，默认信任",
            )
        return VerifyResponse(
            consistent=True,
            confidence=60,
            reason="核查模型返回空内容，跳过验证",
        )

    def _parse_verifier_response(
        self, content: str, retrieved_facts: List[Dict[str, Any]] = None
    ) -> VerifyResponse:
        """
        解析核查模型的 JSON 响应，多层降级。

        解析策略：
        1. 直接 JSON 解析
        2. 从 Markdown 代码块中提取 JSON
        3. 从文本中提取平衡的花括号 JSON
        4. 逐行搜索 JSON 键值对
        5. 启发式文本分析

        Args:
            content: 核查模型的原始响应文本
            retrieved_facts: 检索到的事实列表（用于兜底判断）

        Returns:
            VerifyResponse 实例
        """
        if not content:
            return VerifyResponse(
                consistent=True,
                confidence=60,
                reason="核查模型返回空内容",
            )

        # ---- 策略1: 直接 JSON 解析 ----
        try:
            data = json.loads(content)
            return self._validate_and_build(data)
        except json.JSONDecodeError:
            pass

        # ---- 策略2: Markdown 代码块中的 JSON ----
        # 匹配 ```json ... ``` 或 ``` ... ```
        code_block_patterns = [
            r'```json\s*\n?([\s\S]*?)\n?```',
            r'```\s*\n?(\{[\s\S]*?\})\s*\n?```',
        ]
        for pattern in code_block_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    return self._validate_and_build(data)
                except json.JSONDecodeError:
                    continue

        # ---- 策略3: 提取平衡的花括号 JSON ----
        json_obj = self._extract_balanced_json(content)
        if json_obj:
            try:
                return self._validate_and_build(json_obj)
            except Exception:
                pass

        # ---- 策略4: 正则提取关键字段 ----
        result = self._extract_fields_by_regex(content)
        if result:
            return result

        # ---- 策略5: 启发式降级 ----
        logger.warning(f"所有 JSON 解析策略失败，使用启发式降级。原始内容前200字符: {content[:200]}")
        return self._heuristic_fallback(content, retrieved_facts)

    def _extract_balanced_json(self, text: str) -> Dict[str, Any] | None:
        """
        从文本中提取配对的 {} 包裹的 JSON 对象。

        Args:
            text: 可能包含 JSON 的文本

        Returns:
            解析后的字典，失败返回 None
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = text[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # 尝试修复常见问题：单引号替换、尾部逗号
                        try:
                            fixed = json_str.replace("'", '"')
                            fixed = re.sub(r',\s*}', '}', fixed)
                            fixed = re.sub(r',\s*]', ']', fixed)
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            return None
        return None

    def _extract_fields_by_regex(self, content: str) -> Optional[VerifyResponse]:
        """
        用正则从非标准文本中提取 consistent、confidence、reason 字段。

        Args:
            content: 原始响应文本

        Returns:
            VerifyResponse 或 None
        """
        content_lower = content.lower()

        # 提取 consistent
        consistent = None
        # 模式: "consistent": true/false 或 consistent: true/false
        cons_patterns = [
            r'"consistent"\s*:\s*(true|false)',
            r'consistent\s*[:：]\s*(true|false)',
            r'"consistent"\s*:\s*(True|False)',
        ]
        for pattern in cons_patterns:
            match = re.search(pattern, content)
            if match:
                consistent = match.group(1).lower() in ("true", "yes", "1")
                break

        # 提取 confidence
        confidence = None
        conf_patterns = [
            r'"confidence"\s*:\s*(\d+)',
            r'confidence\s*[:：]\s*(\d+)',
            r'置信度[：:]\s*(\d+)',
        ]
        for pattern in conf_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    confidence = int(match.group(1))
                except ValueError:
                    pass
                break

        # 提取 reason
        reason = ""
        reason_patterns = [
            r'"reason"\s*:\s*"([^"]+)"',
            r'reason\s*[:：]\s*["\u201c]?([^"\u201d\n]+)',
            r'理由[：:]\s*(.+?)(?:\n|$)',
        ]
        for pattern in reason_patterns:
            match = re.search(pattern, content)
            if match:
                reason = match.group(1).strip().rstrip('"').rstrip('\u201d').rstrip(',')
                break

        # 如果成功提取到至少两个关键字段，构建响应
        if consistent is not None and confidence is not None:
            return self._validate_and_build({
                "consistent": consistent,
                "confidence": confidence,
                "reason": reason or "正则提取（无理由字段）",
            })

        # 只提取到部分字段
        if consistent is not None:
            return VerifyResponse(
                consistent=consistent,
                confidence=70 if consistent else 40,
                reason=reason or "正则提取（部分字段）",
            )

        if confidence is not None:
            return VerifyResponse(
                consistent=confidence >= 90,
                confidence=confidence,
                reason=reason or "正则提取（部分字段）",
            )

        return None

    def _validate_and_build(self, data: Dict[str, Any]) -> VerifyResponse:
        """
        验证解析出的数据并构建 VerifyResponse。

        Args:
            data: 解析出的字典

        Returns:
            VerifyResponse 实例
        """
        consistent = data.get("consistent", True)
        confidence = data.get("confidence", 60)
        reason = data.get("reason", data.get("explanation", data.get("message", "")))

        # 确保类型正确
        if isinstance(consistent, str):
            consistent = consistent.lower() in ("true", "yes", "1")
        consistent = bool(consistent)

        try:
            confidence = int(confidence)
        except (ValueError, TypeError):
            confidence = 60
        confidence = max(0, min(100, confidence))

        reason = str(reason) if reason else "核查模型未提供理由"

        # 根据规则微调：如果 consistent=false 但 confidence 太高，修正
        if not consistent and confidence > 80:
            logger.warning(f"consistent=false 但 confidence={confidence} > 80，修正为 50")
            confidence = 50

        # 如果 consistent=true 但 confidence 太低，修正
        if consistent and confidence < 70:
            logger.warning(f"consistent=true 但 confidence={confidence} < 70，修正为 70")
            confidence = 70

        return VerifyResponse(
            consistent=consistent,
            confidence=confidence,
            reason=reason,
        )

    def _heuristic_fallback(
        self, content: str, retrieved_facts: List[Dict[str, Any]] = None
    ) -> VerifyResponse:
        """
        启发式降级解析：从非标准文本中推断一致性。

        Args:
            content: 无法解析的原始文本
            retrieved_facts: 检索到的事实列表

        Returns:
            降级的 VerifyResponse
        """
        content_lower = content.lower()

        # 如果参考资料为空且模型说"没有信息"
        if retrieved_facts is not None and len(retrieved_facts) == 0:
            no_info_keywords = [
                "无法", "没有找到", "无相关", "无信息", "暂无", "未检索到",
                "unknown", "no result", "no information", "not found",
            ]
            if any(kw in content_lower for kw in no_info_keywords):
                return VerifyResponse(
                    consistent=True,
                    confidence=95,
                    reason="参考资料为空，模型如实说明无信息，判定一致",
                )

        # 检测矛盾关键词
        contradiction_keywords = [
            "矛盾", "不一致", "错误", "不正确", "相悖", "不符合", "假", "虚构", "编造",
            "contradiction", "inconsistent", "incorrect", "false", "wrong", "inaccurate",
        ]
        strong_contradiction = [
            "明确矛盾", "严重不符", "完全错误",
        ]
        has_strong = any(kw in content_lower for kw in strong_contradiction)
        has_contradiction = has_strong or any(kw in content_lower for kw in contradiction_keywords)

        # 检测一致关键词
        consistent_keywords = [
            "一致", "正确", "符合", "支持", "无矛盾", "准确", "真实", "可靠",
            "consistent", "correct", "supported", "accurate", "reliable",
        ]
        has_consistent = any(kw in content_lower for kw in consistent_keywords)

        # 检测不确定关键词
        uncertain_keywords = [
            "不确定", "无法判断", "无法验证", "难以判断", "部分",
            "uncertain", "unclear", "partial", "incomplete",
        ]
        has_uncertain = any(kw in content_lower for kw in uncertain_keywords)

        # 提取数字作为置信度
        confidence = 60
        conf_match = re.search(
            r'(?:置信度|confidence|分数|score)[：:\s]*(\d+)',
            content_lower
        )
        if conf_match:
            try:
                confidence = int(conf_match.group(1))
            except ValueError:
                pass
        else:
            # 尝试找任意 0-100 的数字
            nums = re.findall(r'\b(\d{1,3})\b', content)
            valid_nums = [int(n) for n in nums if 0 <= int(n) <= 100]
            if len(valid_nums) == 1:
                confidence = valid_nums[0]
            elif has_strong:
                confidence = 20
            elif has_contradiction and not has_consistent:
                confidence = 35
            elif has_uncertain:
                confidence = 55
            elif has_consistent and not has_contradiction:
                confidence = 85
            else:
                confidence = 60

        confidence = max(0, min(100, confidence))
        consistent = confidence >= 90

        return VerifyResponse(
            consistent=consistent,
            confidence=confidence,
            reason=f"启发式解析（核查模型返回非标准格式，已自动推断）: {content[:150]}",
        )