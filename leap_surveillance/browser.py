"""Chromium navigation via browser-use: URL safety checks and page extraction."""

import asyncio
import ipaddress
import os
import re
import threading
from urllib.parse import urlparse

import requests

from .common import DEFAULT_BROWSER_MODEL, TEST_MODEL, _env_float, _env_int, provider_for_model, strip_provider_prefix
from .models import BrowserEvidence


BROWSER_TIMEOUT = _env_float("LEAP_BROWSER_TIMEOUT", 180.0)
MAX_BROWSER_STEPS = _env_int("LEAP_BROWSER_MAX_STEPS", 15)
BROWSER_EVIDENCE_LIMIT = _env_int("LEAP_BROWSER_EVIDENCE_LIMIT", 4000)

# Each browser-use Agent spawns a Chromium process, so cap concurrency across worker threads.
_BROWSER_SEMAPHORE = threading.Semaphore(1)


def _unusable_extraction_reason(text: str) -> str | None:
    """Detect bot gates and page chrome that are not usable metric evidence."""
    lowered = (text or "").lower()
    if not lowered.strip():
        return "empty extraction"

    failure_markers = [
        ("captcha", "captcha / bot-check page"),
        ("recaptcha", "captcha / bot-check page"),
        ("performing security verification", "security verification page"),
        ("please make sure you are authorized to access this page", "bot-check warning page"),
        ("just a moment...", "security interstitial page"),
        ("cloudflare ray id", "security interstitial page"),
        ("access denied", "access denied page"),
        ("403 forbidden", "forbidden page"),
        ("404 not found", "not found page"),
        ("invalid schema for response_format", "browser-use agent failure"),
        ("stopping due to 3 consecutive failures", "browser-use agent failure"),
        ("llm api call failed", "browser-use agent failure"),
        ("was not successful", "browser-use agent failure"),
        ("unfinished", "browser-use agent failure"),
        ("err_cert", "browser certificate failure"),
    ]
    for marker, reason in failure_markers:
        if marker in lowered:
            return reason

    # Some data portals render only form controls through text readers.
    if re.search(r"\bselect\s+select\s+select\b", lowered):
        return "page chrome only; no extracted data"

    return None


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
    bare_model = strip_provider_prefix(model)
    if provider_for_model(model) == "anthropic":
        from browser_use.llm.anthropic.chat import ChatAnthropic as BrowserChatAnthropic
        return BrowserChatAnthropic(model=bare_model, api_key=os.environ.get("ANTHROPIC_API_KEY"))
    from browser_use.llm.openai.chat import ChatOpenAI as BrowserChatOpenAI
    return BrowserChatOpenAI(model=bare_model, api_key=os.environ.get("OPENAI_API_KEY"))


def wayback_snapshot(url: str, target_date: str) -> BrowserEvidence:
    """Fetch the Wayback Machine snapshot of `url` closest to `target_date` (YYYY-MM-DD), read via Jina."""
    ts = target_date.replace("-", "")[:8]
    try:
        avail = requests.get(
            "http://archive.org/wayback/available", params={"url": url, "timestamp": ts}, timeout=20
        ).json()
        snap_url = ((avail.get("archived_snapshots") or {}).get("closest") or {}).get("url")
        if not snap_url:
            return BrowserEvidence(url=url, objective=f"wayback {target_date}", extracted_text="",
                                   success=False, error="No Wayback snapshot near target date")
        for fetch_url in (f"https://r.jina.ai/{snap_url}", snap_url):
            r = requests.get(fetch_url, timeout=30,
                             headers={"Accept": "text/markdown", "X-Return-Format": "markdown"})
            # Require a digit — empty JS shells pass the length check but have no actual numeric data.
            if r.status_code == 200 and len(r.text.strip()) > 200 and re.search(r"\d+", r.text):
                return BrowserEvidence(url=snap_url, objective=f"wayback {target_date}",
                                       extracted_text=r.text, success=True)
        error = "JS-rendered snapshot: no numeric data found" if r.status_code == 200 else f"Snapshot fetch returned {r.status_code}"
        return BrowserEvidence(url=snap_url, objective=f"wayback {target_date}", extracted_text="",
                               success=False, error=error)
    except Exception as e:
        return BrowserEvidence(url=url, objective=f"wayback {target_date}", extracted_text="",
                               success=False, error=f"Wayback error: {e}")


def browser_extract(
    url: str, objective: str, test_mode: bool = False, model_override: str | None = None,
    as_of_date: str | None = None, skip_jina: bool = False,
) -> BrowserEvidence:
    """Drive Chromium via browser-use to extract `objective` from `url`."""
    # Download files can't be navigated or read by the browser agent.
    if any(url.lower().endswith(ext) for ext in (".pdf", ".zip", ".xlsx", ".xls", ".csv")):
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error="Download URL not supported by browser_extract (use web_search evidence instead)",
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

    # For a past resolution date, the as-of-date snapshot beats the live page (which shows today's value).
    if as_of_date:
        snap = wayback_snapshot(url, as_of_date)
        if snap.success:
            snap.objective = objective
            return snap

    # Try Jina Reader first (avoids Chromium bot-detection) unless the caller knows it's a JS dashboard.
    jina_unusable_error = ""
    if not skip_jina:
        try:
            r = requests.get(
                f"https://r.jina.ai/{url}",
                timeout=30,
                headers={"Accept": "text/markdown", "X-Return-Format": "markdown"},
            )
            if r.status_code == 200 and len(r.text.strip()) > 200:
                unusable_reason = _unusable_extraction_reason(r.text)
                if unusable_reason:
                    jina_unusable_error = f"Jina reader returned unusable content: {unusable_reason}"
                else:
                    return BrowserEvidence(url=url, objective=objective, extracted_text=r.text, success=True)
        except Exception:
            pass

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

        unusable_reason = _unusable_extraction_reason(extracted)
        if unusable_reason or '"error":' in extracted:
            return BrowserEvidence(
                url=url,
                objective=objective,
                extracted_text="",
                success=False,
                error=f"browser-use returned unusable content: {unusable_reason or 'error payload'}",
            )

        return BrowserEvidence(url=url, objective=objective, extracted_text=extracted, success=True)
    except ImportError:
        error = "browser-use not installed"
        if jina_unusable_error:
            error = f"{jina_unusable_error}; {error}"
        return BrowserEvidence(
            url=url,
            objective=objective,
            extracted_text="",
            success=False,
            error=error,
        )
    except Exception as e:
        error = str(e) or "browser-use returned no final result before timeout/step limit"
        if jina_unusable_error:
            error = f"{jina_unusable_error}; browser-use failed: {error}"
        return BrowserEvidence(
            url=url, objective=objective, extracted_text="", success=False, error=error
        )
