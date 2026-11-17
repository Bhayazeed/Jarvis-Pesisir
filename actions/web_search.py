# web_search.py
# Gemini grounded-search replaced with DuckDuckGo + Ollama LLM summarization.
import json
import sys
from pathlib import Path


def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR = _get_base_dir()


def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title":   r.get("title",  ""),
                "snippet": r.get("body",   ""),
                "url":     r.get("href",   ""),
            })
    return results


def _scrape_url(url: str, max_chars: int = 4000) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.extract()
        text = soup.get_text(separator=" ", strip=True)
        # Collapse multiple spaces
        import re
        text = re.sub(r'\s+', ' ', text)
        return text[:max_chars]
    except Exception:
        return ""



def _format_ddg(query: str, results: list[dict]) -> str:
    if not results:
        return f"No results found for: {query}"
    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        if r.get("title"):   lines.append(f"{i}. {r['title']}")
        if r.get("snippet"): lines.append(f"   {r['snippet']}")
        if r.get("url"):     lines.append(f"   {r['url']}")
        lines.append("")
    return "\n".join(lines).strip()


def _llm_summarize(query: str, raw_results: str) -> str:
    try:
        from core.llm_client import call_llm_text
        system = (
            "You are JARVIS. Summarize web search results clearly and concisely. "
            "Answer the user's query directly. Be factual. Address user as 'sir'. "
            "If the query or context implies Malay/Indonesian, summarize in fluent Malay/Indonesian."
        )
        prompt = (
            f"User question: {query}\n\n"
            f"Web search results:\n{raw_results[:4000]}\n\n"
            "Answer the question based on these results:"
        )
        return call_llm_text(prompt, system=system)
    except Exception:
        return raw_results


def _compare(items: list[str], aspect: str) -> str:
    all_results: dict[str, list] = {}
    for item in items:
        try:
            all_results[item] = _ddg_search(f"{item} {aspect}", max_results=3)
        except Exception:
            all_results[item] = []

    lines = [f"Comparison — {aspect.upper()}", "─" * 40]
    for item in items:
        lines.append(f"\n▸ {item}")
        for r in all_results.get(item, [])[:2]:
            if r.get("snippet"):
                lines.append(f"  • {r['snippet']}")
    raw = "\n".join(lines)
    return _llm_summarize(f"Compare {', '.join(items)} regarding {aspect}", raw)


def web_search(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    params = parameters or {}
    query  = params.get("query", "").strip()
    mode   = params.get("mode",  "search").lower().strip()
    items  = params.get("items", [])
    aspect = params.get("aspect", "general").strip() or "general"

    if not query and not items:
        return "Please provide a search query, sir."

    if items and mode != "compare":
        mode = "compare"

    if player:
        player.write_log(f"[Search] {query or ', '.join(items)}")

    print(f"[WebSearch] 🔍 Query: {query!r}  Mode: {mode}")

    try:
        if mode == "compare" and items:
            print(f"[WebSearch] 📊 Comparing: {items}")
            return _compare(items, aspect)

        results = _ddg_search(query)
        if results and results[0].get("url"):
            print(f"[WebSearch] 🌐 Scraping top result: {results[0]['url']}")
            page_text = _scrape_url(results[0]["url"])
            if page_text:
                results[0]["snippet"] = f"[FULL PAGE CONTENT]\n{page_text}"

        raw     = _format_ddg(query, results)
        print(f"[WebSearch] ✅ DDG: {len(results)} result(s).")
        # Let Ollama summarise the results for a cleaner spoken response
        return _llm_summarize(query, raw)

    except Exception as e:
        print(f"[WebSearch] ❌ Failed: {e}")
        return f"Search failed, sir: {e}"
