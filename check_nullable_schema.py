#!/usr/bin/env python
"""Check the fully-nullable value schema (no -999 sentinel).

We replaced the `-999` "no value" sentinel with real JSON `null` on the value fields:
StrictOfficialValue.value, StrictCurrentValue.value, StrictForecastValue.forecast_value,
and (already) StrictResolutionValue.value / best_guess_resolution. This script confirms:

  Part A (offline, no API keys):
    - each value field renders as a nullable union (["number","null"])
    - unit bounds land on the NUMBER branch of forecast_value, not the anyOf wrapper
      (so a null is allowed but non-null values are still bounded)
    - a payload with null values round-trips through strict_to_regular_response

  Part B (--live, needs OPENAI_API_KEY and/or ANTHROPIC_API_KEY):
    - sends the REAL StrictSurveillanceResponse strict schema to GPT and Claude and
      asks for a forecast_value of null, confirming (a) strict mode ACCEPTS a
      nullable+bounded number field and (b) the model can actually emit null.
      This is the thing we could not verify offline.

Usage:
    python check_nullable_schema.py            # offline checks only
    python check_nullable_schema.py --live     # also make one cheap call per provider

Safe to delete after you've confirmed the behavior.
"""
import argparse
import json
import os
import sys

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leap_surveillance import models as m
from leap_surveillance.common import TEST_MODEL, TEST_CLAUDE_MODEL, strip_provider_prefix

# A small bounded quantile question: one dimension, one past date, q50 only.
DIMS = ["Overall"]
QUANTS = [50]
DATES = ["2020-01-01"]
UNIT_MIN, UNIT_MAX = 0.0, 100.0


def _numeric_branch(prop: dict) -> dict:
    for sub in prop.get("anyOf", []):
        if sub.get("type") == "number":
            return sub
    return prop


# ---------------------------------------------------------------- Part A (offline)

def part_a() -> bool:
    print("=== Part A: offline schema + round-trip ===")
    ok = True

    def check(label, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    schema = m.make_strict_schema(
        m.StrictSurveillanceResponse,
        allowed_dimensions=DIMS, allowed_quantiles=QUANTS, allowed_forecast_dates=DATES,
        unit_min=UNIT_MIN, unit_max=UNIT_MAX,
    )
    defs = schema["$defs"]
    fv = defs["StrictForecastValue"]["properties"]["forecast_value"]
    ov = defs["StrictOfficialValue"]["properties"]["value"]
    cv = defs["StrictCurrentValue"]["properties"]["value"]
    rv = defs["StrictResolutionValue"]["properties"]["value"]

    check("forecast_value is nullable union", "null" in json.dumps(fv))
    check("official value is nullable union", "null" in json.dumps(ov))
    check("current value is nullable union", "null" in json.dumps(cv))
    check("resolution value is nullable union", "null" in json.dumps(rv))

    num = _numeric_branch(fv)
    check("bounds on forecast_value NUMBER branch (min=0,max=100)",
          num.get("minimum") == 0 and num.get("maximum") == 100)
    check("bounds NOT on the anyOf wrapper", "minimum" not in fv and "maximum" not in fv)

    # All fields remain required (nullable, not optional-absent).
    req = defs["StrictForecastValue"]["required"]
    check("forecast_value still required (nullable, not absent)", "forecast_value" in req)

    print(f"\n  strict forecast_value schema: {json.dumps(fv)}")

    # Round-trip: null values (no -999) parse to None.
    exp = [m.ExpectedForecast(DATES[0], "Overall", 50, "forecast")]
    payload = {
        "last_official_values": [{"dimension": "Overall", "value": None, "date": "", "source": ""}],
        "current_estimates": [{"dimension": "Overall", "value": 50.0, "confidence": 60}],
        "resolution_values": [],
        "forecasts": [{"forecast_date": DATES[0], "dimension": "Overall", "quantile": 50,
                       "forecast_value": None, "color_code": "white"}],
        "rationale": "test", "sources": [{"url": "u", "title": "t", "snippet": "s"}],
    }
    resp, _ = m.strict_to_regular_response(json.dumps(payload), exp, "quantile")
    check("null forecast_value -> None", resp.forecasts[0].forecast_value is None)
    check("null official value -> None", resp.last_official_values[0].value is None)
    check("numeric current value preserved", resp.current_estimates[0].value == 50.0)

    # Legacy -999 still tolerated by the backstop.
    p2 = json.loads(json.dumps(payload))
    p2["forecasts"][0]["forecast_value"] = -999
    resp2, _ = m.strict_to_regular_response(json.dumps(p2), exp, "quantile")
    check("legacy -999 still maps to None (backstop)", resp2.forecasts[0].forecast_value is None)

    print(f"\n  Part A: {'ALL PASS' if ok else 'SOME FAILED'}\n")
    return ok


# ---------------------------------------------------------------- Part B (live)

LIVE_PROMPT = (
    "Produce a surveillance forecast for a test question whose unit is bounded to [0, 100].\n"
    "There is exactly one dimension 'Overall' and one target date '2020-01-01' with quantile 50.\n"
    "IMPORTANT for this test: you have NO data, so set the q50 forecast_value to null (JSON null), "
    "and set the single last_official_values entry's value to null. "
    "Set current_estimates to one 'Overall' entry with value 50 and confidence 50. "
    "Leave resolution_values empty. Provide a one-sentence rationale and exactly one source "
    "with url/title/snippet. Use color_code 'white'."
)


def _live_schema() -> dict:
    return m.make_strict_schema(
        m.StrictSurveillanceResponse,
        allowed_dimensions=DIMS, allowed_quantiles=QUANTS, allowed_forecast_dates=DATES,
        unit_min=UNIT_MIN, unit_max=UNIT_MAX,
    )


def _report_parsed(provider: str, text: str) -> None:
    exp = [m.ExpectedForecast(DATES[0], "Overall", 50, "forecast")]
    resp, _ = m.strict_to_regular_response(text, exp, "quantile")
    fv = resp.forecasts[0].forecast_value if resp.forecasts else "NO FORECASTS"
    ov = resp.last_official_values[0].value if resp.last_official_values else "NO LOV"
    print(f"  [{provider}] accepted schema + parsed OK. forecast_value={fv!r}  official_value={ov!r}")
    print(f"  [{provider}] -> {'PASS (emitted null)' if fv is None else 'schema OK but model did not emit null'}")


def live_openai() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("  [openai] skipped (no OPENAI_API_KEY)")
        return
    try:
        import openai
        client = openai.OpenAI()
        resp = client.responses.create(
            model=strip_provider_prefix(TEST_MODEL),
            input=[{"role": "user", "content": LIVE_PROMPT}],
            text={"format": {"type": "json_schema", "name": "StrictSurveillanceResponse",
                             "schema": _live_schema(), "strict": True}},
        )
        text = getattr(resp, "output_text", None)
        if not text:  # fallback extraction
            text = "".join(c.text for item in resp.output for c in getattr(item, "content", [])
                           if getattr(c, "type", "") == "output_text")
        _report_parsed(f"openai:{TEST_MODEL}", text)
    except Exception as e:  # noqa: BLE001 - we want to SEE a schema-rejection here
        print(f"  [openai] ERROR (does strict mode reject nullable+bounded?): {type(e).__name__}: {e}")


def live_anthropic() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  [anthropic] skipped (no ANTHROPIC_API_KEY)")
        return
    try:
        import anthropic
        client = anthropic.Anthropic()
        tool = {"name": "produce_forecast", "description": "Output the structured surveillance forecast.",
                "input_schema": _live_schema()}
        resp = client.messages.create(
            model=strip_provider_prefix(TEST_CLAUDE_MODEL),
            max_tokens=2000,
            tools=[tool],
            tool_choice={"type": "tool", "name": "produce_forecast"},
            messages=[{"role": "user", "content": LIVE_PROMPT}],
        )
        tool_use = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
        if tool_use is None:
            print("  [anthropic] no tool_use block returned")
            return
        _report_parsed(f"anthropic:{TEST_CLAUDE_MODEL}", json.dumps(tool_use.input))
    except Exception as e:  # noqa: BLE001
        print(f"  [anthropic] ERROR (does strict mode reject nullable+bounded?): {type(e).__name__}: {e}")


def part_b() -> None:
    print("=== Part B: live provider checks (needs API keys) ===")
    live_openai()
    live_anthropic()
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also hit OpenAI/Anthropic APIs")
    args = ap.parse_args()
    a_ok = part_a()
    if args.live:
        part_b()
    else:
        print("(run with --live to test that both providers accept the nullable schema)")
    return 0 if a_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
