import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

from docetl.operations.base import BaseOperation
from docetl.operations.utils.playwright import stealth_browser_async
from docetl.operations.utils.validation import lookup_field

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


# Content-types that markitdown can convert to markdown
_MARKITDOWN_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

_CONTENT_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}


def _to_markdown_from_bytes(data: bytes, content_type: str, url: str) -> str:
    """Convert binary document (PDF, DOCX, XLSX, etc.) to markdown via markitdown."""
    import os
    import tempfile
    from urllib.parse import urlparse

    from markitdown import MarkItDown

    # Determine file extension: prefer URL path, fall back to content-type map
    url_path = urlparse(url).path
    _, url_ext = os.path.splitext(url_path)
    ext = url_ext or _CONTENT_TYPE_EXTENSIONS.get(content_type, "")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        md = MarkItDown()
        return md.convert(tmp).text_content
    finally:
        os.unlink(tmp)


def _fetch_with_playwright(url: str, timeout: int) -> tuple[str, bool, str]:
    """
    Fetch a single URL using Playwright.

    Returns (content, is_html, content_type). If the navigation triggers a file download,
    falls back to requests and returns the raw content with is_html=False.
    For binary content, content is the raw bytes decoded as latin-1.
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
        raw_content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        is_html = "html" in raw_content_type.lower()
        if is_html:
            return response.text, True, raw_content_type
        return response.content.decode("latin-1"), False, raw_content_type

    # Sniff the playwright-rendered HTML
    is_html = bool(content and content.lstrip().lower().startswith(("<!", "<html")))
    return content, is_html, "text/html" if is_html else ""


class WebFetchOperation(BaseOperation):
    """
    Fetches URLs and stores the fetched content in an output field.
    The URL can come from a field in each document (url_field) or be a
    static URL applied to every document (url). Supports both single URL
    strings and lists of URLs. Fetching is done in parallel using a thread pool.

    Example config (dynamic URL from field):
        type: web_fetch
        url_field: url          # field containing URL or list of URLs
        output_field: content   # field to store fetched content
        timeout: 10             # per-request timeout in seconds (optional)
        max_workers: 10         # max parallel fetch threads (optional)
        use_playwright: false   # use Playwright for JS-rendered pages (optional)
        body_only: false        # only keep <body> content (optional, HTML only)
        convert_to_markdown: false  # convert HTML/PDF/DOCX/XLSX/PPTX to markdown (optional)

    Example config (static URL):
        type: web_fetch
        url: https://example.com/data.json   # static URL fetched for every document
        output_field: content
    """

    class schema(BaseOperation.schema):
        type: str = "web_fetch"
        url_field: Optional[str] = None
        url: Optional[str] = None
        output_field: str
        timeout: Optional[int] = 10
        max_workers: Optional[int] = 10
        use_playwright: Optional[bool] = False
        body_only: Optional[bool] = False
        convert_to_markdown: Optional[bool] = False

    def execute(self, input_data: list[dict]) -> tuple[list[dict], float]:
        url_field = self.config.get("url_field")
        static_url = self.config.get("url")
        output_field = self.config["output_field"]
        timeout = self.config.get("timeout", 10)
        max_workers = self.config.get("max_workers", 10)
        use_playwright = self.config.get("use_playwright", False)
        body_only = self.config.get("body_only", False)
        convert_to_markdown = self.config.get("convert_to_markdown", False)

        # Collect all (doc_index, url_index, url) triples
        tasks = []
        for doc_idx, doc in enumerate(input_data):
            if static_url is not None:
                value = static_url
            elif url_field is not None:
                value = lookup_field(doc, url_field)
            else:
                value = None
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
                    # HEAD-check first: if content-type is a binary/convertible type,
                    # playwright won't get the bytes (renders empty HTML), so use requests directly.
                    try:
                        head = requests.head(url, timeout=timeout, allow_redirects=True)
                        head_ct = head.headers.get("Content-Type", "").split(";")[0].strip()
                    except Exception:
                        head_ct = ""
                    if head_ct in _MARKITDOWN_TYPES:
                        response = requests.get(url, timeout=timeout)
                        response.raise_for_status()
                        raw_content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                        is_html = False
                        is_convertible = raw_content_type in _MARKITDOWN_TYPES
                        raw_bytes = response.content
                        content = response.content.decode("latin-1")
                    else:
                        content, is_html, raw_content_type = _fetch_with_playwright(url, timeout)
                        is_convertible = raw_content_type in _MARKITDOWN_TYPES
                        raw_bytes = content.encode("latin-1") if (is_convertible and content is not None) else None
                else:
                    response = requests.get(url, timeout=timeout)
                    response.raise_for_status()
                    raw_content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                    is_html = "html" in raw_content_type
                    is_convertible = raw_content_type in _MARKITDOWN_TYPES
                    if is_html:
                        content = response.text
                        raw_bytes = None
                    else:
                        raw_bytes = response.content
                        content = response.content.decode("latin-1")

                if is_html:
                    if body_only:
                        content = _extract_body(content)
                    content = _make_urls_absolute(content, url)
                    if convert_to_markdown:
                        content = _to_markdown(content)
                elif is_convertible and convert_to_markdown and raw_bytes is not None:
                    content = _to_markdown_from_bytes(raw_bytes, raw_content_type, url)

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
                    if static_url is not None:
                        original_urls = static_url if isinstance(static_url, list) else [static_url]
                    else:
                        original_urls = lookup_field(doc, url_field) if url_field is not None else []
                        if not isinstance(original_urls, list):
                            original_urls = [original_urls]
                    new_doc[output_field] = [
                        raw.get(i, "") for i in range(len(original_urls))
                    ]
                else:
                    new_doc[output_field] = raw
            else:
                if static_url is not None:
                    value = static_url
                elif url_field is not None:
                    value = lookup_field(doc, url_field)
                else:
                    value = None
                new_doc[output_field] = [] if isinstance(value, list) else ""
            output.append(new_doc)

        return output, 0.0
