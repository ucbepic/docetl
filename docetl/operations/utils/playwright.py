"""Shared Playwright browser utilities with bot-detection evasion."""

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_WEBDRIVER_INIT_SCRIPT = (
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
)

_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]


def stealth_browser_sync(headless: bool = True):
    """
    Sync context manager yielding (browser, page) with bot-detection evasion.

    Usage:
        with stealth_browser_sync() as (browser, page):
            page.goto(...)
    """
    from contextlib import contextmanager
    from playwright.sync_api import sync_playwright

    @contextmanager
    def _ctx():
        pw = sync_playwright().start()
        try:
            browser = pw.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
            try:
                ctx = browser.new_context(user_agent=_USER_AGENT)
                ctx.add_init_script(_WEBDRIVER_INIT_SCRIPT)
                page = ctx.new_page()
                yield browser, page
            finally:
                browser.close()
        finally:
            pw.stop()

    return _ctx()


async def stealth_browser_async(playwright, headless: bool = True, **context_kwargs):
    """
    Async helper: launch a stealth browser and return (browser, context, page).
    The caller is responsible for awaiting browser.close().

    Extra keyword arguments are forwarded to new_context() (e.g. accept_downloads=True).

    Usage (inside `async with async_playwright() as p`):
        browser, context, page = await stealth_browser_async(p, accept_downloads=True)
        try:
            ...
        finally:
            await browser.close()
    """
    browser = await playwright.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
    context = await browser.new_context(user_agent=_USER_AGENT, **context_kwargs)
    await context.add_init_script(_WEBDRIVER_INIT_SCRIPT)
    page = await context.new_page()
    return browser, context, page
