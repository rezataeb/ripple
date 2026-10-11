"""Build the cached market scan for demo mode.

Spec Section 9: "Results are cached by query; demo mode uses a cached scan," so the
live demo works with no Nimble key.

These entries are **illustrative sample data**, not a real scan. Brightside Coffee
Subscriptions is a fictional company, so its competitors are fictional too, with
example.com URLs. Shipping invented competitor names, real-looking prices and
real-looking links would be fabricated web evidence, and the whole point of the
market-scan card is that web numbers are cited and checkable. Every record is
marked `source: "demo"` and carries a note saying so, and the UI must show it.

When NIMBLE_API_KEY is set, the agent route fetches live results in this same shape
and caches them by query, replacing these.

Run: uv run python model/scripts/make_demo_scan.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public" / "model"
CONTRACT = ROOT / "contract"

SEEN = "2026-10-11T09:00:00Z"
DEMO_NOTE = (
    "Illustrative sample data, not a live scan. Brightside Coffee is a fictional "
    "company, so these competitors are fictional too. Add NIMBLE_API_KEY to fetch "
    "real competitor prices and offers with real sources."
)


def entry(name, price, offer, slug, unit="per bag, monthly subscription", conf=0.9):
    return {
        "competitor": name,
        "price": price,
        "unit": unit,
        "offer": offer,
        "url": f"https://example.com/{slug}",
        "seen_at": SEEN,
        "confidence": conf,
    }


def comparison(your_price: float, competitors: list[dict], city: str) -> dict:
    prices = sorted(c["price"] for c in competitors if c["price"] is not None)
    above = sum(1 for p in prices if your_price > p)
    below = sum(1 for p in prices if your_price < p)
    median = (
        None
        if not prices
        else (
            prices[len(prices) // 2]
            if len(prices) % 2
            else (prices[len(prices) // 2 - 1] + prices[len(prices) // 2]) / 2
        )
    )
    where = f" in {city}" if city else ""
    if above and not below:
        tail = f"above all {len(prices)} competitors{where}"
    elif above:
        tail = f"above {above} of {len(prices)} competitors{where}"
    elif below:
        tail = f"below all {len(prices)} competitors{where}"
    else:
        tail = f"level with {len(prices)} competitors{where}"
    return {
        "your_price": your_price,
        "above_count": above,
        "below_count": below,
        "total": len(prices),
        "median_competitor_price": median,
        "sentence": f"Your price ${your_price:,.2f} is {tail}.",
    }


def scan(query: str, product: str, city: str, your_price: float, competitors: list[dict]) -> dict:
    return {
        "query": query,
        "product": product,
        "city": city,
        "source": "demo",
        "fetched_at": SEEN,
        "competitors": competitors,
        "comparison": comparison(your_price, competitors, city),
        "note": DEMO_NOTE,
        "sources": [{"title": c["competitor"], "url": c["url"]} for c in competitors],
    }


# One entry per city the demo can ask about, keyed by a normalized query so the
# agent route can look one up without a network call.
SCANS = {
    "coffee subscription austin": scan(
        "coffee subscription price Austin",
        "coffee subscription",
        "Austin",
        25.92,
        [
            entry("Rolling Hills Roasters", 22.50, "Free shipping over $30", "rolling-hills"),
            entry("Cold Brew Collective", 24.00, None, "cold-brew-collective"),
            entry("Lone Star Coffee Club", 27.95, "First month 50% off", "lone-star-coffee"),
        ],
    ),
    "coffee subscription boston": scan(
        "coffee subscription price Boston",
        "coffee subscription",
        "Boston",
        19.20,
        [
            entry("Harbor Bean Co.", 21.00, "Student discount 15%", "harbor-bean"),
            entry("Commonwealth Coffee", 23.50, None, "commonwealth-coffee"),
            entry("North End Roasting", 18.75, "2 bags for $34", "north-end-roasting"),
        ],
    ),
    "coffee subscription miami": scan(
        "coffee subscription price Miami",
        "coffee subscription",
        "Miami",
        24.00,
        [
            entry("Bayside Brew Club", 18.90, "Launch offer: first bag free", "bayside-brew"),
            entry("Calle Ocho Coffee", 20.25, None, "calle-ocho-coffee"),
        ],
    ),
    "coffee subscription": scan(
        "coffee subscription price",
        "coffee subscription",
        "",
        24.00,
        [
            entry("Rolling Hills Roasters", 22.50, "Free shipping over $30", "rolling-hills"),
            entry("Cold Brew Collective", 24.00, None, "cold-brew-collective"),
            entry("Harbor Bean Co.", 21.00, "Student discount 15%", "harbor-bean"),
            entry("Lone Star Coffee Club", 27.95, "First month 50% off", "lone-star-coffee"),
        ],
    ),
}


def main() -> None:
    schema = json.loads((CONTRACT / "market_scan.schema.json").read_text())
    for key, doc in SCANS.items():
        jsonschema.validate(doc, schema)
        assert doc["source"] == "demo" and "Illustrative" in doc["note"], key

    PUBLIC.mkdir(parents=True, exist_ok=True)
    out = PUBLIC / "market_scan.demo.json"
    out.write_text(json.dumps(SCANS, indent=1) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB), {len(SCANS)} cached queries")
    for key, doc in SCANS.items():
        print(f"  {key:<34} {len(doc['competitors'])} competitors | "
              f"{doc['comparison']['sentence']}")


if __name__ == "__main__":
    main()
