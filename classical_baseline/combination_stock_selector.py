"""
Classical "combination" baseline: rank stocks using EPS, P/E, and dividend yield together.

Higher EPS and dividend yield improve the score; lower P/E improves the score
(same direction as ``select_by_pe``). Returns the top ``num`` tickers for
``implement_classical``.

``stocks_data`` comes from ``backtester.clean.build_stock_data``; this module does
not call yfinance.
"""


def _stock_score(stock: str, stocks_data: dict[str, dict[str, float]]) -> float:
    """Combine fundamentals: eps × dividend_yield / pe when pe > 0, else 0."""
    data = stocks_data.get(stock, {})
    eps = float(data.get("eps") or 0.0)
    pe_ratio = float(data.get("pe_ratio") or 0.0)
    dividend_yield = float(data.get("dividend_yield") or 0.0)

    if pe_ratio <= 0.0:
        return 0.0
    return eps * dividend_yield / pe_ratio


def combination_stock_selector(
    stocks: list[str],
    stocks_data: dict[str, dict[str, float]],
    num: int,
) -> tuple[str, ...]:
    if not stocks:
        return ()
    candidates = [(symbol, _stock_score(symbol, stocks_data)) for symbol in stocks]
    candidates.sort(key=lambda item: item[1], reverse=True)

    n = max(1, min(int(num), len(candidates)))
    return tuple(symbol for symbol, _ in candidates[:n])
