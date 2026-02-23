# GEM Decision Tool

Narzędzie do kwartalnej/miesięcznej decyzji inwestycyjnej w ramach strategii **Global Equity Momentum** (Gary Antonacci) z perspektywy polskiego inwestora na koncie **IKE/IKZE** w Bossa (BOŚ).

## Jak działa

1. Pobiera 12-miesięczne stopy zwrotu aktywów w **PLN**
2. Tworzy ranking momentum
3. Jeśli najlepsze aktywo > 0% → **KUP** je
4. Jeśli najlepsze aktywo ≤ 0% → **RISK-OFF** (obligacje US Treasury)
5. Cały kapitał w jednym aktywie — zero dywersyfikacji wewnątrz strategii

## Uniwersum aktywów

| Aktywo | Ticker Bossa | Giełda | Waluta |
|--------|-------------|--------|--------|
| S&P 500 | SPXS | Euronext Amsterdam | EUR |
| NASDAQ 100 | CNDX | Euronext Amsterdam | EUR |
| MSCI EM IMI | EMIM | Euronext Amsterdam | EUR |
| Gold | EGLN | LSE London | EUR |
| mWIG40 TR | ETFBM40TR | GPW Warszawa | PLN |
| US Treasury 0-1yr (risk-off) | IB01 | LSE London | USD |

## Instalacja

```bash
# Python 3.10+ wymagany (zalecany 3.13)
python3 -m venv venv
source venv/bin/activate
pip install pandas yfinance matplotlib
```

## Użycie

```bash
source venv/bin/activate

# Bieżący sygnał
python gem_decision.py

# Z wykresem porównawczym
python gem_decision.py --chart

# Historyczny check (np. co by było w czerwcu 2025)
python gem_decision.py --date 2025-06-30

# Bez zapisu do logu
python gem_decision.py --no-log
```

## Przykładowe wyjście

```
📅 Okres analizy: 2024-12-31 → 2025-12-31 (12M rolling)

📊 Ranking Momentum (w PLN):
  #1  Gold                  +44.23%
  #2  mWIG40 TR             +37.72%
  #3  MSCI EM IMI           +15.58%
  #4  NASDAQ 100             +4.98%
  #5  S&P 500                +2.83%

🎯 DECYZJA: KUP Gold — momentum +44.23%

🏦 Jak kupić na Bossa:
   Ticker:  IGLN
   Giełda:  LSE (London)
   Waluta:  USD
```

## Rutyna

1. **1. dnia miesiąca** — odpal `python gem_decision.py`
2. Jeśli aktywo się zmieniło → sprzedaj stare, kup nowe na Bossa
3. Jeśli bez zmian → nic nie rób
4. Sygnały zmieniają się średnio **1-2× rocznie**

## Pliki

| Plik | Opis |
|------|------|
| `gem_decision.py` | Główny skrypt |
| `decision_log.csv` | Historia decyzji (tworzony automatycznie) |

## Konfiguracja

W `gem_decision.py` na górze pliku:

- `LOOKBACK_MONTHS = 12` — okres obserwacji
- `SIGNAL_LAG_MONTHS = 1` — pomijanie ostatniego miesiąca (skip-month)
- `ASSETS` — lista aktywów do porównania
- `RISK_OFF` — aktywo risk-off
