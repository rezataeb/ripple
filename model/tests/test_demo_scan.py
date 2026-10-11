"""The cached market scan must stay honestly labelled.

The agent is allowed to state web numbers because they are cited and checkable.
Demo-mode figures are not real, so if the `demo` source or the note ever gets
dropped the card would present invented competitor prices as real evidence.
That is the one thing this file exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "web" / "public" / "model" / "market_scan.demo.json"
SCHEMA = ROOT / "contract" / "market_scan.schema.json"

pytestmark = pytest.mark.skipif(
    not SCAN.exists(), reason="run model/scripts/make_demo_scan.py first"
)


@pytest.fixture(scope="module")
def scans() -> dict:
    return json.loads(SCAN.read_text())


def test_every_cached_scan_matches_the_schema(scans) -> None:
    schema = json.loads(SCHEMA.read_text())
    for key, doc in scans.items():
        jsonschema.validate(doc, schema)
        assert doc["query"], key


def test_every_cached_scan_is_marked_as_demo_data(scans) -> None:
    for key, doc in scans.items():
        assert doc["source"] == "demo", f"{key}: must not claim to be a live scan"
        assert "llustrative" in doc["note"], f"{key}: the note must say it is illustrative"
        assert "fictional" in doc["note"].lower(), f"{key}: the note must say so plainly"


def test_demo_scans_do_not_use_real_looking_urls(scans) -> None:
    """A real-looking link invites someone to treat invented data as checkable."""
    for key, doc in scans.items():
        for c in doc["competitors"]:
            assert c["url"].startswith("https://example.com/"), f"{key}: {c['url']}"


def test_the_comparison_sentence_matches_its_own_numbers(scans) -> None:
    """The UI prints `sentence` verbatim, so it must agree with the counts."""
    for key, doc in scans.items():
        comp = doc["comparison"]
        prices = [c["price"] for c in doc["competitors"] if c["price"] is not None]
        assert comp["total"] == len(prices), key
        assert comp["above_count"] == sum(1 for p in prices if comp["your_price"] > p), key
        assert comp["below_count"] == sum(1 for p in prices if comp["your_price"] < p), key
        assert f"{comp['your_price']:,.2f}" in comp["sentence"], key
        if comp["above_count"] and comp["below_count"]:
            assert f"{comp['above_count']} of {comp['total']}" in comp["sentence"], key


def test_the_austin_scan_reproduces_the_spec_example(scans) -> None:
    """Spec Section 9 quotes this line; the demo should actually produce it."""
    comp = scans["coffee subscription austin"]["comparison"]
    assert comp["sentence"] == "Your price $25.92 is above 2 of 3 competitors in Austin."


def test_every_competitor_carries_a_source(scans) -> None:
    for key, doc in scans.items():
        urls = {s["url"] for s in doc["sources"]}
        for c in doc["competitors"]:
            assert c["url"] in urls, f"{key}: {c['competitor']} has no entry in sources"
