import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from docetl.operations.base import BaseOperation
from docetl.operations.utils.playwright import stealth_browser_async


def _make_urls_absolute(html: str, base_url: str) -> str:
    """Rewrite all relative URLs in html to absolute using base_url."""
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    attrs = [
        ("a", "href"),
        ("img", "src"),
        ("script", "src"),
        ("link", "href"),
        ("form", "action"),
        ("iframe", "src"),
        ("source", "src"),
        ("source", "srcset"),
        ("video", "src"),
        ("audio", "src"),
    ]
    for tag, attr in attrs:
        for el in soup.find_all(tag, **{attr: True}):
            val = el[attr]
            if val and not val.startswith(("data:", "javascript:", "#", "mailto:")):
                el[attr] = urljoin(base_url, val)
    return str(soup)


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


def _fetch_with_playwright(url: str, timeout: int) -> tuple[str, bool]:
    """
    Fetch a single URL using Playwright.

    Returns (content, is_html). If the navigation triggers a file download,
    falls back to requests and returns the raw content with is_html=False.
    """
    from playwright.async_api import async_playwright

    async def _fetch():
        download_url = None

        async with async_playwright() as p:
            browser, context, page = await stealth_browser_async(
                p, accept_downloads=True
            )

            # Capture download events before navigating
            download_future: asyncio.Future = asyncio.get_event_loop().create_future()

            async def on_download(download):
                if not download_future.done():
                    download_future.set_result(download.url)

            page.on("download", on_download)

            try:
                await page.goto(
                    url, timeout=timeout * 1000, wait_until="domcontentloaded"
                )
            except Exception:
                # Navigation may "fail" when a download is triggered
                pass

            # If a download was triggered, bail out and use requests instead
            if download_future.done():
                download_url = download_future.result()

            if download_url is None:
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=timeout * 1000
                    )
                except Exception:
                    pass
                content = await page.content()
            else:
                content = None

            await browser.close()
            return content, download_url

    content, download_url = asyncio.run(_fetch())

    if download_url is not None:
        # Fall back to requests for the actual binary/non-HTML resource
        response = requests.get(download_url or url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        is_html = "html" in content_type.lower()
        if is_html:
            return response.text, True
        return response.content.decode("latin-1"), False

    # Sniff the playwright-rendered HTML
    is_html = bool(content and content.lstrip().lower().startswith(("<!", "<html")))
    return content, is_html


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
        body_only: false        # only keep <body> content (optional, HTML only)
        convert_to_markdown: false  # convert HTML to markdown (optional, HTML only)
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
                    content, is_html = _fetch_with_playwright(url, timeout)
                else:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    is_html = "html" in content_type.lower()
                    content = (
                        response.text if is_html else response.content.decode("latin-1")
                    )

                if is_html:
                    if body_only:
                        content = _extract_body(content)
                    content = _make_urls_absolute(content, url)
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
                    original_urls = doc.get(url_field, [])
                    new_doc[output_field] = [
                        raw.get(i, "") for i in range(len(original_urls))
                    ]
                else:
                    new_doc[output_field] = raw
            else:
                value = doc.get(url_field)
                new_doc[output_field] = [] if isinstance(value, list) else ""
            output.append(new_doc)

        return output, 0.0
