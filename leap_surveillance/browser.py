"""Chromium navigation via the browser-use library.

This module owns the actual page-fetching path: URL safety checks, the LLM-driven
browser agent (one Chromium instance at a time via the shared semaphore), and the
BrowserEvidence return shape.

It deliberately does NOT own:
  - decide_browser (judge-side prompt that proposes a URL)
  - refine_with_browser (LLM call that integrates the extracted text)
  - the orchestration that chains those together
Those stay in research.py / run_surveillance.py because they're LLM calls, not
Chromium calls.
"""

import asyncio
import ipaddress
import os
import threading
from urllib.parse import urlparse

from .common import DEFAULT_BROWSER_MODEL, TEST_MODEL, _env_float, _env_int, provider_for_model
from .models import BrowserEvidence


BROWSER_TIMEOUT = _env_float("LEAP_BROWSER_TIMEOUT", 180.0)
MAX_BROWSER_STEPS = _env_int("LEAP_BROWSER_MAX_STEPS", 15)
BROWSER_EVIDENCE_LIMIT = _env_int("LEAP_BROWSER_EVIDENCE_LIMIT", 4000)

# Limit browser-use Agent invocations to one at a time across worker threads.
# Each Agent spawns a Chromium process; running >1 concurrently risks resource contention.
_BROWSER_SEMAPHORE = threading.Semaphore(1)


def is_safe_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False, f"Invalid scheme: {parsed.scheme}"

        host = parsed.hostname or ""

        # Avoid letting the agent wander into search engines / CAPTCHA loops.
        search_domains = (
            "google.com",
            "googleusercontent.com",
            "bing.com",
            "duckduckgo.com",
            "yahoo.com",
            "baidu.com",
            "yandex.com",
            "search.brave.com",
        )
        if any(host == d or host.endswith(f".{d}") for d in search_domains):
            return False, "Search engine domain blocked"

        if host in ("localhost", "127.0.0.1", "::1"):
            return False, "Localhost blocked"

        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                return False, f"Private/reserved IP blocked: {host}"
        except ValueError:
            pass

        if host == "metadata.google.internal":
            return False, "Metadata endpoint blocked"

        return True, ""
    except Exception as e:
        return False, f"URL parse error: {e}"


def _get_browser_llm(model: str):
    bare_model = model.split("/", 1)[-1] if "/" in model else model
    if provider_for_model(model) == "anthropic":
        from browser_use.llm.anthropic.chat import ChatAnthropic as BrowserChatAnthropic
        return BrowserChatAnthropic(model=bare_model, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    from browser_use.llm.openai.chat import ChatOpenAI as BrowserChatOpenAI
    return BrowserChatOpenAI(model=bare_model, api_key=os.environ.get("OPENAI_API_KEY"))


def browser_extract(
    url: str, objective: str, test_mode: bool = False, model_override: str | None = None
) -> BrowserEvidence:
    """Drive Chromium via browser-use to extract `objective` from `url`."""
    # PDF viewers make browser-use extraction unreliable.
    if url.lower().endswith(".pdf"):
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error="PDF URL not supported by browser_extract (use web_search evidence instead)",
        )

    safe, reason = is_safe_url(url)
    if not safe:
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error=f"Unsafe URL: {reason}",
        )

    async def _extract():
        from browser_use import Agent, Browser

        model = model_override or (TEST_MODEL if test_mode else DEFAULT_BROWSER_MODEL)
        llm = _get_browser_llm(model)
        browser = Browser(headless=True)
        try:
            agent = Agent(
                task=f"Go to {url} and {objective}. Return only the extracted data.",
                llm=llm,
                browser=browser,
            )
            return await asyncio.wait_for(
                agent.run(max_steps=MAX_BROWSER_STEPS), timeout=BROWSER_TIMEOUT
            )
        finally:
            await browser.stop()

    try:
        with _BROWSER_SEMAPHORE:
            result = asyncio.run(_extract())
        # Prefer final result; full histories include transient errors.
        extracted = getattr(result, "final_result", lambda: None)() or ""

        # browser-use sometimes returns its own error text instead of raising.
        failure_markers = [
            "Invalid schema for response_format",
            "Stopping due to 3 consecutive failures",
            "LLM API call failed",
            '"error":',
            "was not successful",
            "Unfinished",
            "CAPTCHA",
            "recaptcha",
            "ERR_CERT",
        ]
        if not extracted or any(m in extracted for m in failure_markers):
            return BrowserEvidence(
                url=url,
                objective=objective,
                extracted_text="",
                success=False,
                error="browser-use run failed (see logs)",
            )

        return BrowserEvidence(url=url, objective=objective, extracted_text=extracted, success=True)
    except ImportError:
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error="browser-use not installed",
        )
    except Exception as e:
        return BrowserEvidence(
            url=url, objective=objective, extracted_text="", success=False, error=str(e)
        )
