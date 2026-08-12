"""StockGPT Enhanced -- NSE market intelligence engine.

A clean rebuild of the original StockGPT pipeline. Design goals that shaped
every module in this package:

1. One authoritative value per concept. The original project accumulated
   multiple generations of the "same" column (e.g. two independent risk
   penalties, three rewrites of the technical score) because features were
   patched in over time without retiring what came before. Here, every
   score/column is computed exactly once, by exactly one function.

2. Missing data stays missing. Silently coercing an unfetched ratio to 0
   makes "we don't know" indistinguishable from "the true value is zero" --
   and a 0 debt-to-equity reads as *safe* to every downstream risk check.
   Every module here preserves NaN for genuinely unknown values and carries
   an explicit `*_data_available` flag instead of guessing.

3. Nothing is hardcoded to today's data. Any bound that varies by ticker
   (price, market cap, volume) is derived from the live dataset, never a
   fixed constant. Ratios and percentages (RSI, ROE, score bands) are the
   one category that *is* fine to fix, because they're already scale-free.

See README.md for the full design rationale and a comparison against the
original StockGPT project.
"""

__version__ = "2.0.0"
