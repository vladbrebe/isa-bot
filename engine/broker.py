"""Trading 212 Public API (v0) client for an Invest or Stocks ISA account.

Auth is HTTP Basic: API key as the username, API secret as the password. Demo (paper money)
and live are separate environments with separate keys.

Rate limits are per account and per endpoint, so calls to each endpoint are spaced out and a
429 waits for the x-ratelimit-reset time. Market orders are not idempotent in this API: an
order POST is retried only after a 429 (rejected before it reached the book), never after a
timeout or a server error, because the order may already exist.
"""
from __future__ import annotations

import time

import requests

BASE = {"demo": "https://demo.trading212.com", "live": "https://live.trading212.com"}

# Minimum seconds between calls to each endpoint, from the published limits plus a margin.
SPACING = {
    "summary": 5.2,        # 1 / 5s
    "positions": 1.1,      # 1 / 1s
    "pending": 5.2,        # 1 / 5s
    "order": 1.3,          # 50 / min (market orders and cancels)
    "instruments": 50.5,   # 1 / 50s
    "exchanges": 30.5,     # 1 / 30s
    "transactions": 3.1,   # 20 / min
}


class T212Error(RuntimeError):
    def __init__(self, status: int, body: str, path: str):
        super().__init__(f"Trading 212 API returned {status} on {path}: {body[:300]}")
        self.status, self.body, self.path = status, body, path


class Trading212:
    def __init__(self, key: str, secret: str, env: str = "demo", timeout: float = 20.0):
        if env not in BASE:
            raise ValueError(f"T212_ENV must be 'demo' or 'live', not {env!r}")
        if not key or not secret:
            raise ValueError("T212_API_KEY and T212_API_SECRET must both be set")
        self.env, self.base, self.timeout = env, BASE[env], timeout
        self.s = requests.Session()
        self.s.auth = (key, secret)
        self.s.headers.update({"Accept": "application/json", "User-Agent": "isa-quant-bot"})
        self._last: dict[str, float] = {}

    def _space(self, bucket: str) -> None:
        gap = SPACING.get(bucket, 1.1)
        wait = self._last.get(bucket, -1e9) + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[bucket] = time.monotonic()

    def _call(self, method: str, path: str, bucket: str, *, params=None, json=None, safe_to_retry=True):
        for attempt in range(8):
            self._space(bucket)
            try:
                r = self.s.request(method, self.base + path, params=params, json=json, timeout=self.timeout)
            except requests.RequestException:
                if safe_to_retry and attempt < 3:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise
            if r.status_code == 429:
                reset = r.headers.get("x-ratelimit-reset")
                try:
                    wait = float(reset) - time.time() if reset else 5.0
                except ValueError:
                    wait = 5.0
                time.sleep(min(max(wait, 1.0), 65.0))
                continue
            if (r.status_code >= 500 or r.status_code == 408) and safe_to_retry and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            if not r.ok:
                raise T212Error(r.status_code, r.text, path)
            return r.json() if r.content else None
        raise T212Error(429, "still rate limited after 8 attempts", path)

    # ---- reads -------------------------------------------------------------------------------
    def account_summary(self) -> dict:
        return self._call("GET", "/api/v0/equity/account/summary", "summary")

    def positions(self) -> list[dict]:
        return self._call("GET", "/api/v0/equity/positions", "positions") or []

    def pending_orders(self) -> list[dict]:
        return self._call("GET", "/api/v0/equity/orders", "pending") or []

    def instruments(self) -> list[dict]:
        return self._call("GET", "/api/v0/equity/metadata/instruments", "instruments") or []

    def exchanges(self) -> list[dict]:
        return self._call("GET", "/api/v0/equity/metadata/exchanges", "exchanges") or []

    def transactions(self, since_iso: str | None, max_pages: int = 40) -> list[dict]:
        """Deposits, withdrawals, fees and interest since a time, following nextPagePath."""
        params = {"limit": 50, **({"time": since_iso} if since_iso else {})}
        page = self._call("GET", "/api/v0/equity/history/transactions", "transactions", params=params)
        items = list(page.get("items", [])) if page else []
        for _ in range(max_pages):
            nxt = page.get("nextPagePath") if page else None
            if not nxt:
                break
            page = self._call("GET", nxt, "transactions")
            items += page.get("items", []) if page else []
        return items

    # ---- writes ------------------------------------------------------------------------------
    def market_order(self, ticker: str, quantity: float) -> dict:
        """Positive quantity buys, negative sells. Never blindly retried (see module doc)."""
        body = {"ticker": ticker, "quantity": quantity, "extendedHours": False}
        return self._call("POST", "/api/v0/equity/orders/market", "order", json=body, safe_to_retry=False)

    def cancel_order(self, order_id: int) -> None:
        self._call("DELETE", f"/api/v0/equity/orders/{int(order_id)}", "order")
