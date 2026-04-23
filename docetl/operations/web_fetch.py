import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

import requests

from docetl.operations.base import BaseOperation


def _extract_body(html: str) -> str:
    """Extract content of <body> tag; fall back to full HTML if not found."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    return str(body) if body else html


def _to_markdown(html: str) -> str:
    """Convert HTML to markdown."""
    import markdownify

    return markdownify.markdownify(html, heading_style="ATX")


def _fetch_with_playwright(url: str, timeout: int) -> str:
    """Fetch a single URL using Playwright (sync wrapper around async)."""
    from playwright.async_api import async_playwright

    async def _fetch():
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            # Patch out navigator.webdriver before any page script runs
            await context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            page = await context.new_page()
            await page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            except Exception:
                pass
            content = await page.content()
            await browser.close()
            return content

    return asyncio.run(_fetch())


class WebFetchOperation(BaseOperation):
    """
    Fetches URLs from a field in each document and stores the fetched content
    in an output field. Supports both single URL strings and lists of URLs.
    Fetching is done in parallel using a thread pool.

    Example config:
        type: web_fetch
        url_field: url          # field containing URL or list of URLs
        output_field: content   # field to store fetched content
        timeout: 10             # per-request timeout in seconds (optional)
        max_workers: 10         # max parallel fetch threads (optional)
        use_playwright: false   # use Playwright for JS-rendered pages (optional)
    """

    class schema(BaseOperation.schema):
        type: str = "web_fetch"
        url_field: str
        output_field: str
        timeout: Optional[int] = 10
        max_workers: Optional[int] = 10
        use_playwright: Optional[bool] = False
        body_only: Optional[bool] = False
        convert_to_markdown: Optional[bool] = False

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        url_field = self.config["url_field"]
        output_field = self.config["output_field"]
        timeout = self.config.get("timeout", 10)
        max_workers = self.config.get("max_workers", 10)
        use_playwright = self.config.get("use_playwright", False)
        body_only = self.config.get("body_only", False)
        convert_to_markdown = self.config.get("convert_to_markdown", False)

        # Collect all (doc_index, url_index, url) triples
        tasks = []
        for doc_idx, doc in enumerate(input_data):
            value = doc.get(url_field)
            if value is None:
                continue
            if isinstance(value, list):
                for url_idx, url in enumerate(value):
                    if url:
                        tasks.append((doc_idx, url_idx, url))
            else:
                tasks.append((doc_idx, None, value))

        # results[doc_idx] = str (single) or dict[url_idx -> str] (list)
        results: dict = {}

        def fetch(doc_idx, url_idx, url):
            try:
                if use_playwright:
                    content = _fetch_with_playwright(url, timeout)
                else:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    content = response.text
                if body_only:
                    content = _extract_body(content)
                if convert_to_markdown:
                    content = _to_markdown(content)
                return doc_idx, url_idx, content
            except Exception as e:
                return doc_idx, url_idx, f"ERROR: {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch, doc_idx, url_idx, url): (doc_idx, url_idx)
                for doc_idx, url_idx, url in tasks
            }
            for future in as_completed(futures):
                doc_idx, url_idx, content = future.result()
                if url_idx is None:
                    results[doc_idx] = content
                else:
                    if doc_idx not in results:
                        results[doc_idx] = {}
                    results[doc_idx][url_idx] = content

        output = []
        for doc_idx, doc in enumerate(input_data):
            new_doc = copy.deepcopy(doc)
            if doc_idx in results:
                raw = results[doc_idx]
                if isinstance(raw, dict):
                    # Reconstruct list in original order
                    original_urls = doc.get(url_field, [])
                    new_doc[output_field] = [
                        raw.get(i, "") for i in range(len(original_urls))
                    ]
                else:
                    new_doc[output_field] = raw
            else:
                # No URL found — set empty
                value = doc.get(url_field)
                new_doc[output_field] = [] if isinstance(value, list) else ""
            output.append(new_doc)

        return output, 0.0
