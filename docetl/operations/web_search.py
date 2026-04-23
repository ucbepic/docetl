import base64
import copy
import re
from typing import Literal, Optional
from urllib.parse import quote_plus, unquote

from jinja2 import Environment, StrictUndefined

from docetl.operations.base import BaseOperation
from docetl.operations.utils.playwright import stealth_browser_sync

SUPPORTED_ENGINES = Literal["brave", "bing", "google", "duckduckgo"]


def _search_brave(query: str, max_results: int) -> list[dict]:
    with stealth_browser_sync() as (browser, page):
        hits = []
        offset = 0
        while len(hits) < max_results:
            url = f"https://search.brave.com/search?q={quote_plus(query)}&source=web&offset={offset}"
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)
            results = page.query_selector_all('div[data-type="web"]')
            if not results:
                break
            for r in results:
                title_el = r.query_selector(".title")
                link_el = r.query_selector("a[href]")
                snippet_el = r.query_selector(".generic-snippet .content")
                title = title_el.text_content().strip() if title_el else ""
                href = link_el.get_attribute("href") if link_el else ""
                snippet = snippet_el.text_content().strip() if snippet_el else ""
                if title and href:
                    hits.append({"title": title, "description": snippet, "url": href})
                if len(hits) >= max_results:
                    break
            offset += len(results)
        return hits


def _decode_bing_url(href: str) -> str:
    m = re.search(r"u=a1([A-Za-z0-9_-]+)", href)
    if m:
        try:
            return base64.urlsafe_b64decode(m.group(1) + "==").decode(
                "utf-8", errors="replace"
            )
        except Exception:
            pass
    return href


def _search_bing(query: str, max_results: int) -> list[dict]:
    with stealth_browser_sync() as (browser, page):
        hits = []
        start = 1
        while len(hits) < max_results:
            url = f"https://www.bing.com/search?q={quote_plus(query)}&count=10&first={start}"
            page.goto(url, wait_until="domcontentloaded")
            results = page.query_selector_all("li.b_algo")
            if not results:
                break
            for r in results:
                title_el = r.query_selector("h2 a")
                snippet_el = r.query_selector(".b_caption p")
                title = title_el.text_content().strip() if title_el else ""
                raw_url = title_el.get_attribute("href") if title_el else ""
                url_clean = _decode_bing_url(raw_url) if raw_url else ""
                snippet = snippet_el.text_content().strip() if snippet_el else ""
                if title and url_clean:
                    hits.append(
                        {"title": title, "description": snippet, "url": url_clean}
                    )
                if len(hits) >= max_results:
                    break
            start += len(results)
        return hits


def _search_google(query: str, max_results: int) -> list[dict]:
    with stealth_browser_sync() as (browser, page):
        hits = []
        start = 0
        while len(hits) < max_results:
            url = f"https://www.google.com/search?q={quote_plus(query)}&num=10&start={start}"
            page.goto(url, wait_until="domcontentloaded")
            h3s = page.query_selector_all("h3")
            if not h3s:
                break
            for h3 in h3s:
                title = h3.text_content().strip()
                href = h3.evaluate(
                    "el => { let p = el; while(p && p.tagName !== 'A') p = p.parentElement;"
                    " return p ? p.href : ''; }"
                )
                snippet = h3.evaluate(
                    """el => {
                        let block = el;
                        while (block && !block.dataset.hveid) block = block.parentElement;
                        if (!block) return '';
                        let div = block.querySelector('div[style]');
                        return div ? div.textContent.trim() : '';
                    }"""
                )
                if (
                    title
                    and href
                    and href.startswith("http")
                    and "google.com" not in href
                ):
                    hits.append({"title": title, "description": snippet, "url": href})
                if len(hits) >= max_results:
                    break
            if not h3s:
                break
            start += 10
        return hits


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    with stealth_browser_sync() as (browser, page):
        page.goto(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            wait_until="domcontentloaded",
        )
        results = page.query_selector_all(".result__body")
        hits = []
        for r in results:
            title_el = r.query_selector(".result__a")
            snippet_el = r.query_selector(".result__snippet")
            title = title_el.text_content().strip() if title_el else ""
            href = title_el.get_attribute("href") if title_el else ""
            snippet = snippet_el.text_content().strip() if snippet_el else ""
            m = re.search(r"uddg=([^&]+)", href or "")
            if m:
                href = unquote(m.group(1))
            if title and href:
                hits.append({"title": title, "description": snippet, "url": href})
            if len(hits) >= max_results:
                break
        return hits


class WebSearchOperation(BaseOperation):
    """
    Performs a web search for each document using a Jinja2 query template
    and stores a list of result dicts (title, description, url) in an output field.

    Uses Playwright (headless browser) — no API keys required.

    Example config:
        type: web_search
        search_engine: brave           # brave | bing | google | duckduckgo
        query_template: "{{ input.topic }} recent news"
        output_field: search_results   # field to store list of results
        max_results: 10                # max number of results per document (optional)
    """

    class schema(BaseOperation.schema):
        type: str = "web_search"
        search_engine: SUPPORTED_ENGINES = "brave"
        query_template: str
        output_field: str = "search_results"
        max_results: Optional[int] = 10

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        engine = self.config.get("search_engine", "brave")
        query_template = self.config["query_template"]
        output_field = self.config["output_field"]
        max_results = self.config.get("max_results", 10)

        env = Environment(undefined=StrictUndefined)
        tmpl = env.from_string(query_template)

        output = []
        for doc in input_data:
            new_doc = copy.deepcopy(doc)
            query = tmpl.render({"input": doc}).strip()
            try:
                if not query:
                    results = []
                elif engine == "brave":
                    results = _search_brave(query, max_results)
                elif engine == "bing":
                    results = _search_bing(query, max_results)
                elif engine == "google":
                    results = _search_google(query, max_results)
                elif engine == "duckduckgo":
                    results = _search_duckduckgo(query, max_results)
                else:
                    results = []
            except Exception as e:
                results = [{"title": "ERROR", "description": str(e), "url": ""}]
            new_doc[output_field] = results
            output.append(new_doc)

        return output, 0.0
