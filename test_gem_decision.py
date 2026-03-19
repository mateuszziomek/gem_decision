"""Tests for gem_decision.py — GEM Decision Tool."""
from __future__ import annotations

import math
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

import gem_decision as gd


# ─── helpers ──────────────────────────────────────────────────────────

def _daily_series(start: str, values: list[float], name: str = "TEST") -> pd.Series:
    """Create a daily pd.Series from a start date and list of values."""
    idx = pd.bdate_range(start=start, periods=len(values))
    s = pd.Series(values, index=idx, name=name)
    return s


# ═══════════════════════════════════════════════════════════════════════
# TestComputeRollingDates
# ═══════════════════════════════════════════════════════════════════════

class TestComputeRollingDates(unittest.TestCase):

    def test_mid_month(self):
        """Feb 13, 2026 → end=Dec 31 2025, start=Dec 31 2024."""
        start, end = gd.compute_rolling_dates(datetime(2026, 2, 13))
        self.assertEqual(end, pd.Timestamp("2025-12-31"))
        self.assertEqual(start, pd.Timestamp("2024-12-31"))

    def test_jan_reference(self):
        """Jan 15, 2026 → end=Nov 30 2025, start=Nov 30 2024."""
        start, end = gd.compute_rolling_dates(datetime(2026, 1, 15))
        self.assertEqual(end, pd.Timestamp("2025-11-30"))
        self.assertEqual(start, pd.Timestamp("2024-11-30"))

    def test_mar_reference(self):
        """Mar 5, 2026 → end=Jan 31 2026, start=Jan 31 2025."""
        start, end = gd.compute_rolling_dates(datetime(2026, 3, 5))
        self.assertEqual(end, pd.Timestamp("2026-01-31"))
        self.assertEqual(start, pd.Timestamp("2025-01-31"))

    def test_leap_year(self):
        """Apr 10, 2024 → end=Feb 29 2024 (leap year)."""
        start, end = gd.compute_rolling_dates(datetime(2024, 4, 10))
        self.assertEqual(end, pd.Timestamp("2024-02-29"))
        self.assertEqual(start, pd.Timestamp("2023-02-28"))

    def test_end_is_end_of_month(self):
        """End date is always last day of a month."""
        for month in range(1, 13):
            _, end = gd.compute_rolling_dates(datetime(2026, month, 15))
            # Next day should be 1st of next month
            next_day = end + pd.Timedelta(days=1)
            self.assertEqual(next_day.day, 1, f"Failed for reference month={month}")

    def test_12m_period(self):
        """Start and end are always ~12 months apart."""
        start, end = gd.compute_rolling_dates(datetime(2026, 6, 1))
        delta_months = (end.year - start.year) * 12 + (end.month - start.month)
        self.assertEqual(delta_months, 12)


# ═══════════════════════════════════════════════════════════════════════
# TestLastPriceOnOrBefore
# ═══════════════════════════════════════════════════════════════════════

class TestLastPriceOnOrBefore(unittest.TestCase):

    def test_exact_match(self):
        s = _daily_series("2025-01-06", [100, 101, 102])  # Mon-Wed
        price, actual = gd.last_price_on_or_before(s, pd.Timestamp("2025-01-07"))
        self.assertEqual(price, 101.0)
        self.assertEqual(actual, pd.Timestamp("2025-01-07"))

    def test_weekend_gap(self):
        """Saturday target should return Friday's price."""
        s = _daily_series("2025-01-06", [100, 101, 102, 103, 104])  # Mon-Fri
        price, actual = gd.last_price_on_or_before(s, pd.Timestamp("2025-01-11"))  # Saturday
        self.assertEqual(price, 104.0)
        self.assertEqual(actual, pd.Timestamp("2025-01-10"))  # Friday

    def test_holiday_gap(self):
        """Target in gap returns last available before gap."""
        idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-08"])
        s = pd.Series([10, 11, 12], index=idx, name="TEST")
        price, actual = gd.last_price_on_or_before(s, pd.Timestamp("2025-01-06"))
        self.assertEqual(price, 11.0)
        self.assertEqual(actual, pd.Timestamp("2025-01-03"))

    def test_empty_series(self):
        s = pd.Series([], dtype=float, name="EMPTY")
        s.index = pd.DatetimeIndex([])
        price, actual = gd.last_price_on_or_before(s, pd.Timestamp("2025-01-01"))
        self.assertTrue(math.isnan(price))
        self.assertIsNone(actual)

    def test_target_before_data(self):
        s = _daily_series("2025-06-01", [100, 101])
        price, actual = gd.last_price_on_or_before(s, pd.Timestamp("2025-01-01"))
        self.assertTrue(math.isnan(price))
        self.assertIsNone(actual)

    def test_nan_skipping(self):
        """NaN values in series are skipped."""
        idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
        s = pd.Series([100, float("nan"), 102], index=idx, name="TEST")
        price, actual = gd.last_price_on_or_before(s, pd.Timestamp("2025-01-03"))
        self.assertEqual(price, 100.0)
        self.assertEqual(actual, pd.Timestamp("2025-01-02"))


# ═══════════════════════════════════════════════════════════════════════
# TestComputeReturnPln
# ═══════════════════════════════════════════════════════════════════════

class TestComputeReturnPln(unittest.TestCase):

    def test_pln_only(self):
        """PLN asset, no FX conversion. 100 → 110 = +10%."""
        idx = pd.to_datetime(["2024-12-31", "2025-06-30", "2025-12-31"])
        prices = pd.Series([100.0, 105.0, 110.0], index=idx, name="PLN_ASSET")
        result = gd.compute_return_pln(prices, None,
                                       pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"))
        self.assertAlmostEqual(result["return_pct"], 10.0)

    def test_usd_with_fx(self):
        """USD asset with FX: price * fx_rate at start vs end."""
        idx = pd.to_datetime(["2024-12-31", "2025-12-31"])
        prices = pd.Series([50.0, 55.0], index=idx, name="USD_ASSET")
        fx = pd.Series([4.0, 4.2], index=idx, name="USDPLN=X")
        # PLN: 50*4=200 → 55*4.2=231, return = 15.5%
        result = gd.compute_return_pln(prices, fx,
                                       pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"))
        self.assertAlmostEqual(result["return_pct"], 15.5)

    def test_zero_start_price(self):
        idx = pd.to_datetime(["2024-12-31", "2025-12-31"])
        prices = pd.Series([0.0, 100.0], index=idx, name="ZERO")
        result = gd.compute_return_pln(prices, None,
                                       pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"))
        self.assertTrue(math.isnan(result["return_pct"]))

    def test_nan_prices_exits(self):
        """NaN start price → _check_data_gap exits (no data on or before start)."""
        idx = pd.to_datetime(["2024-12-31", "2025-12-31"])
        prices = pd.Series([float("nan"), 100.0], index=idx, name="NAN_START")
        with self.assertRaises(SystemExit):
            gd.compute_return_pln(prices, None,
                                  pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"))

    def test_negative_return(self):
        idx = pd.to_datetime(["2024-12-31", "2025-12-31"])
        prices = pd.Series([100.0, 80.0], index=idx, name="DOWN")
        result = gd.compute_return_pln(prices, None,
                                       pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31"))
        self.assertAlmostEqual(result["return_pct"], -20.0)


# ═══════════════════════════════════════════════════════════════════════
# TestMakeDecision
# ═══════════════════════════════════════════════════════════════════════

class TestMakeDecision(unittest.TestCase):

    def test_positive_momentum_buys(self):
        ranking = [
            {"name": "S&P 500", "return_pct": 15.0},
            {"name": "Gold", "return_pct": 5.0},
        ]
        d = gd.make_decision(ranking)
        self.assertEqual(d["action"], "BUY")
        self.assertEqual(d["asset"], "S&P 500")

    def test_negative_momentum_risk_off(self):
        ranking = [
            {"name": "S&P 500", "return_pct": -5.0},
            {"name": "Gold", "return_pct": -10.0},
        ]
        d = gd.make_decision(ranking)
        self.assertEqual(d["action"], "RISK_OFF")

    def test_zero_return_is_risk_off(self):
        ranking = [{"name": "S&P 500", "return_pct": 0.0}]
        d = gd.make_decision(ranking)
        self.assertEqual(d["action"], "RISK_OFF")

    def test_empty_ranking(self):
        d = gd.make_decision([])
        self.assertEqual(d["action"], "NO_DATA")

    def test_all_nan(self):
        ranking = [
            {"name": "A", "return_pct": float("nan")},
            {"name": "B", "return_pct": float("nan")},
        ]
        d = gd.make_decision(ranking)
        self.assertEqual(d["action"], "NO_DATA")

    def test_mixed_nan_and_valid(self):
        ranking = [
            {"name": "Valid", "return_pct": 5.0},
            {"name": "Bad", "return_pct": float("nan")},
        ]
        d = gd.make_decision(ranking)
        self.assertEqual(d["action"], "BUY")
        self.assertEqual(d["asset"], "Valid")


# ═══════════════════════════════════════════════════════════════════════
# TestValidateFxRates
# ═══════════════════════════════════════════════════════════════════════

class TestValidateFxRates(unittest.TestCase):

    def _fx_series(self, values: list[float]) -> pd.Series:
        idx = pd.bdate_range("2025-01-01", periods=len(values))
        return pd.Series(values, index=idx, name="USDPLN=X")

    def test_valid_passes(self):
        fx_data = {"USD": self._fx_series([3.8, 4.0, 4.2])}
        gd.validate_fx_rates(fx_data)  # should not raise

    def test_too_low_exits(self):
        fx_data = {"USD": self._fx_series([1.5, 4.0])}
        with self.assertRaises(SystemExit):
            gd.validate_fx_rates(fx_data)

    def test_too_high_exits(self):
        fx_data = {"USD": self._fx_series([4.0, 8.0])}
        with self.assertRaises(SystemExit):
            gd.validate_fx_rates(fx_data)

    def test_empty_exits(self):
        s = pd.Series([], dtype=float, name="USDPLN=X")
        s.index = pd.DatetimeIndex([])
        fx_data = {"USD": s}
        with self.assertRaises(SystemExit):
            gd.validate_fx_rates(fx_data)


# ═══════════════════════════════════════════════════════════════════════
# TestValidateTickerCurrencies
# ═══════════════════════════════════════════════════════════════════════

class TestValidateTickerCurrencies(unittest.TestCase):

    def _make_mock_tickers(self, currency_map: dict[str, str | None]):
        """Build a mock yf.Tickers result with given ticker→currency mapping."""
        mock_batch = MagicMock()
        tickers_dict = {}
        for ticker, currency in currency_map.items():
            t = MagicMock()
            t.info = {"currency": currency} if currency is not None else {}
            tickers_dict[ticker] = t
        mock_batch.tickers = tickers_dict
        return mock_batch

    @patch("gem_decision.yf.Tickers")
    def test_all_usd_passes(self, mock_tickers_cls):
        mock_tickers_cls.return_value = self._make_mock_tickers({
            "SPXS.L": "USD", "CNDX.L": "USD", "EIMI.L": "USD",
            "IGLN.L": "USD", "IB01.L": "USD",
        })
        assets = [
            {"name": "A", "ticker": "SPXS.L", "source": "yahoo", "currency": "USD"},
            {"name": "B", "ticker": "CNDX.L", "source": "yahoo", "currency": "USD"},
        ]
        risk_off = {"name": "R", "ticker": "IB01.L", "source": "yahoo", "currency": "USD"}
        gd.validate_ticker_currencies(assets, risk_off)  # should not raise

    @patch("gem_decision.yf.Tickers")
    def test_gbx_mismatch_exits(self, mock_tickers_cls):
        mock_tickers_cls.return_value = self._make_mock_tickers({
            "IGLN.L": "GBX", "IB01.L": "USD",
        })
        assets = [{"name": "Gold", "ticker": "IGLN.L", "source": "yahoo", "currency": "USD"}]
        risk_off = {"name": "R", "ticker": "IB01.L", "source": "yahoo", "currency": "USD"}
        with self.assertRaises(SystemExit):
            gd.validate_ticker_currencies(assets, risk_off)

    @patch("gem_decision.yf.Tickers")
    def test_missing_currency_warns(self, mock_tickers_cls):
        """Missing currency in info → warning, not exit."""
        mock_tickers_cls.return_value = self._make_mock_tickers({
            "SPXS.L": None, "IB01.L": "USD",
        })
        assets = [{"name": "A", "ticker": "SPXS.L", "source": "yahoo", "currency": "USD"}]
        risk_off = {"name": "R", "ticker": "IB01.L", "source": "yahoo", "currency": "USD"}
        # Should warn but not exit
        gd.validate_ticker_currencies(assets, risk_off)


# ═══════════════════════════════════════════════════════════════════════
# TestDataCoverage
# ═══════════════════════════════════════════════════════════════════════

class TestDataCoverage(unittest.TestCase):

    def test_small_gap_warns(self):
        """Gap of 20 days for start (>15) should warn but not exit."""
        requested = pd.Timestamp("2025-01-31")
        actual = pd.Timestamp("2025-01-10")  # 21 days gap
        # Should not raise — just prints warning
        gd._check_data_gap("test start", requested, actual, gd.DATA_GAP_WARN_START)

    def test_large_gap_exits(self):
        """Gap > 60 days should sys.exit(1)."""
        requested = pd.Timestamp("2025-06-30")
        actual = pd.Timestamp("2025-04-01")  # 90 days gap
        with self.assertRaises(SystemExit):
            gd._check_data_gap("test start", requested, actual, gd.DATA_GAP_WARN_START)


if __name__ == "__main__":
    unittest.main()
