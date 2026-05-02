"""
Fact-ARC Bocha API 检索模块

使用 Bocha API 进行外部知识检索，返回格式化的搜索结果供核查模块使用。
支持 jieba 关键词提取，自动优化检索查询。
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class BochaRetriever:
    """Bocha API 检索器。"""

    BASE_URL = "https://api.bochaai.com/v1/web-search"

    def __init__(self, api_key: str, keyword_extractor: Optional[Callable[[str], str]] = None):
        self.api_key = api_key
        self.keyword_extractor = keyword_extractor
        logger.info(
            f"BochaRetriever 初始化完成"
            f"{' (含关键词提取器)' if keyword_extractor else ''}"
        )

    async def search(self, query: str, count: int = 8) -> List[Dict[str, Any]]:
        if self.keyword_extractor:
            search_query = self.keyword_extractor(query)
        else:
            search_query = query

        results = await self._do_search(search_query, count)

        if not results and search_query != query:
            logger.info(f"关键词搜索无结果，用原始查询重试: '{query[:60]}'")
            results = await self._do_search(query, count)

        return results

    async def _do_search(self, query: str, count: int = 8) -> List[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": count,
        }

        logger.info(f"Bocha 检索: '{query[:80]}{'...' if len(query) > 80 else ''}'")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.BASE_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                raise RuntimeError("Bocha API 认证失败：API 密钥无效或已过期") from e
            elif status == 429:
                raise RuntimeError("Bocha API 速率限制：请求过于频繁，请稍后重试") from e
            elif status == 403:
                raise RuntimeError("Bocha API 权限不足：请检查API Key权限") from e
            else:
                raise RuntimeError(f"Bocha API 请求失败: HTTP {status}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Bocha API 网络请求失败: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Bocha API 调用异常: {e}") from e

        results = self._parse_response(data)
        logger.info(f"Bocha 检索完成，获得 {len(results)} 条结果")
        return results

    def _parse_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []

        code = data.get("code")
        if code is not None and code != 200:
            return []

        raw_items = []
        inner_data = data.get("data")

        if isinstance(inner_data, dict):
            web_pages = inner_data.get("webPages")
            if isinstance(web_pages, dict):
                value = web_pages.get("value")
                if isinstance(value, list):
                    raw_items = value

        if not raw_items and isinstance(inner_data, dict):
            for key in ("results", "items", "webPages"):
                val = inner_data.get(key)
                if isinstance(val, list):
                    raw_items = val
                    break

        if not raw_items:
            for key in ("results", "items"):
                val = data.get(key)
                if isinstance(val, list):
                    raw_items = val
                    break
            web_pages = data.get("webPages")
            if isinstance(web_pages, dict):
                value = web_pages.get("value")
                if isinstance(value, list):
                    raw_items = value

        if not raw_items:
            return []

        formatted = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            formatted.append({
                "title": item.get("name", item.get("title", "")),
                "snippet": item.get("summary", item.get("snippet", item.get("description", ""))),
                "url": item.get("url", item.get("displayUrl", item.get("link", ""))),
                "date": item.get("datePublished", item.get("dateLastCrawled", item.get("date", ""))),
                "site_name": item.get("siteName", item.get("site_name", "")),
                "site_icon": item.get("siteIcon", item.get("site_icon", "")),
            })

        formatted = [f for f in formatted if f["title"].strip() or f["snippet"].strip()]
        return formatted