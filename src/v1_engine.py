"""
Options Pricing Engine v1
─────────────────────────
Deliberately unoptimized — same philosophy as matching engine v1.
Works correctly. Benchmarks poorly. Improvement path is obvious.

Known v1 sins (to fix in v2):
  • scalar math module — no numpy vectorization
  • d1/d2 recomputed in every function call (no caching)
  • loop-based Monte Carlo — pure Python, no vectorization
  • norm_cdf approximation via math.erfc — scipy in v2
  • no implied vol surface, single point only
  • binomial tree uses Python lists not numpy arrays

Run:
    python src/v1_engine.py
"""

import math
import time
import random
import sys

# ── Type aliases (documentation only, Python doesn't enforce) ─────────────
Spot   = float   # underlying price
Strike = float   # option strike price
Rate   = float   # risk-free rate (annualised)
Vol    = float   # volatility (annualised)
Tenor  = float   # time to expiry (years)

# ── Normal distribution ───────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Cumulative standard normal distribution."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

# ── Black-Scholes d1, d2 ──────────────────────────────────────────────────

def _d1(S: Spot, K: Strike, r: Rate, sigma: Vol, T: Tenor) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / \
           (sigma * math.sqrt(T))

def _d2(S: Spot, K: Strike, r: Rate, sigma: Vol, T: Tenor) -> float:
    return _d1(S, K, r, sigma, T) - sigma * math.sqrt(T)

# ── Black-Scholes Pricing ─────────────────────────────────────────────────

def bs_call(S: Spot, K: Strike, r: Rate, sigma: Vol, T: Tenor) -> float:
    """European call option price via Black-Scholes."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_put(S: Spot, K: Strike, r: Rate, sigma: Vol, T: Tenor) -> float:
    """European put option price via Black-Scholes."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_price(S, K, r, sigma, T, option_type: str = 'call') -> float:
    """Unified pricer — call or put."""
    return bs_call(S, K, r, sigma, T) if option_type == 'call' \
           else bs_put(S, K, r, sigma, T)

# ── Greeks ────────────────────────────────────────────────────────────────

def delta(S, K, r, sigma, T, option_type='call') -> float:
    d1 = _d1(S, K, r, sigma, T)
    if option_type == 'call':
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0

def gamma(S, K, r, sigma, T) -> float:
    d1 = _d1(S, K, r, sigma, T)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def theta(S, K, r, sigma, T, option_type='call') -> float:
    d1 = _d1(S, K, r, sigma, T)
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = norm_pdf(d1)
    if option_type == 'call':
        return (-(S * pdf_d1 * sigma) / (2.0 * math.sqrt(T))
                - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365.0
    return (-(S * pdf_d1 * sigma) / (2.0 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365.0

def vega(S, K, r, sigma, T) -> float:
    d1 = _d1(S, K, r, sigma, T)
    return S * norm_pdf(d1) * math.sqrt(T) * 0.01

def rho(S, K, r, sigma, T, option_type='call') -> float:
    d2 = _d2(S, K, r, sigma, T)
    if option_type == 'call':
        return K * T * math.exp(-r * T) * norm_cdf(d2) * 0.01
    return -K * T * math.exp(-r * T) * norm_cdf(-d2) * 0.01

def all_greeks(S, K, r, sigma, T, option_type='call') -> dict:
    return {
        'delta' : delta(S, K, r, sigma, T, option_type),
        'gamma' : gamma(S, K, r, sigma, T),
        'theta' : theta(S, K, r, sigma, T, option_type),
        'vega'  : vega(S, K, r, sigma, T),
        'rho'   : rho(S, K, r, sigma, T, option_type),
    }

# ── Put-Call Parity Check ─────────────────────────────────────────────────
def put_call_parity_check(S, K, r, T, call_price, put_price, tol=0.01) -> bool:
    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    return abs(lhs - rhs) < tol

# ── Monte Carlo Pricer ────────────────────────────────────────────────────
def mc_price(S, K, r, sigma, T, option_type='call',
             n_paths: int = 10_000, seed: int = 42) -> float:
    random.seed(seed)
    discount = math.exp(-r * T)
    drift    = (r - 0.5 * sigma ** 2) * T
    diffuse  = sigma * math.sqrt(T)
    total_payoff = 0.0
    for _ in range(n_paths):
        z  = random.gauss(0.0, 1.0)
        ST = S * math.exp(drift + diffuse * z)
        if option_type == 'call':
            total_payoff += max(ST - K, 0.0)
        else:
            total_payoff += max(K - ST, 0.0)
    return discount * total_payoff / n_paths

def mc_price_with_stderr(S, K, r, sigma, T, option_type='call',
                          n_paths=10_000, seed=42) -> tuple:
    random.seed(seed)
    discount = math.exp(-r * T)
    drift    = (r - 0.5 * sigma ** 2) * T
    diffuse  = sigma * math.sqrt(T)
    payoffs = []
    for _ in range(n_paths):
        z  = random.gauss(0.0, 1.0)
        ST = S * math.exp(drift + diffuse * z)
        payoffs.append(max(ST - K, 0.0) if option_type == 'call'
                       else max(K - ST, 0.0))
    mean = sum(payoffs) / n_paths
    variance = sum((p - mean) ** 2 for p in payoffs) / (n_paths - 1)
    stderr = math.sqrt(variance / n_paths)
    return discount * mean, discount * stderr

# ── Binomial Tree Pricer ──────────────────────────────────────────────────
def binomial_price(S, K, r, sigma, T, option_type='call',
                   n_steps: int = 100, american: bool = False) -> float:
    dt      = T / n_steps
    u       = math.exp(sigma * math.sqrt(dt))
    d       = 1.0 / u
    p       = (math.exp(r * dt) - d) / (u - d)
    discount = math.exp(-r * dt)
    stock = [S * (u ** (n_steps - 2 * j)) for j in range(n_steps + 1)]
    if option_type == 'call':
        values = [max(stock[j] - K, 0.0) for j in range(n_steps + 1)]
    else:
        values = [max(K - stock[j], 0.0) for j in range(n_steps + 1)]
    for i in range(n_steps - 1, -1, -1):
        stock  = [stock[j] / u for j in range(i + 1)]
        values = [discount * (p * values[j] + (1.0 - p) * values[j + 1])
                  for j in range(i + 1)]
        if american:
            if option_type == 'call':
                values = [max(values[j], stock[j] - K) for j in range(i + 1)]
            else:
                values = [max(values[j], K - stock[j]) for j in range(i + 1)]
    return values[0]

# ── Implied Volatility ────────────────────────────────────────────────────
def implied_vol(market_price: float, S, K, r, T,
                option_type='call', tol: float = 1e-6,
                max_iter: int = 100) -> float:
    sigma = 0.5
    for _ in range(max_iter):
        price = bs_price(S, K, r, sigma, T, option_type)
        v     = vega(S, K, r, sigma, T) / 0.01
        diff  = price - market_price
        if abs(diff) < tol:
            return sigma
        if abs(v) < 1e-10:
            return float('nan')
        sigma -= diff / v
        sigma  = max(1e-4, min(sigma, 10.0))
    return float('nan')

# ── Benchmark ─────────────────────────────────────────────────────────────
def run_benchmark():
    S     = 24_000.0
    K     = 24_000.0
    r     = 0.065
    sigma = 0.15
    T     = 30 / 365

    N_BS  = 100_000
    N_MC  = 200
    N_BIN = 1_000

    t0 = time.perf_counter()
    for _ in range(N_BS):
        bs_call(S, K, r, sigma, T)
    t1 = time.perf_counter()
    bs_per_sec = N_BS / (t1 - t0)

    t0 = time.perf_counter()
    for _ in range(N_BS):
        all_greeks(S, K, r, sigma, T)
    t1 = time.perf_counter()
    greeks_per_sec = N_BS / (t1 - t0)

    t0 = time.perf_counter()
    for _ in range(N_MC):
        mc_price(S, K, r, sigma, T, n_paths=1_000)
    t1 = time.perf_counter()
    mc_per_sec = N_MC / (t1 - t0)

    t0 = time.perf_counter()
    for _ in range(N_BIN):
        binomial_price(S, K, r, sigma, T)
    t1 = time.perf_counter()
    bin_per_sec = N_BIN / (t1 - t0)

    call = bs_call(S, K, r, sigma, T)
    put  = bs_put(S, K, r, sigma, T)
    pcp  = put_call_parity_check(S, K, r, T, call, put)
    mc_c, mc_se = mc_price_with_stderr(S, K, r, sigma, T, n_paths=50_000)
    bin_c = binomial_price(S, K, r, sigma, T, n_steps=200)
    g     = all_greeks(S, K, r, sigma, T)
    iv    = implied_vol(call, S, K, r, T)

    print("══════════════════════════════════════════════")
    print("  Options Pricing Engine v1 — Python")
    print("══════════════════════════════════════════════")
    print(f"  Underlying (Nifty)  : ₹{S:,.0f}")
    print(f"  Strike (ATM)        : ₹{K:,.0f}")
    print(f"  Risk-free rate      : {r*100:.1f}%")
    print(f"  Implied vol         : {sigma*100:.1f}%")
    print(f"  Days to expiry      : 30")
    print("──────────────────────────────────────────────")
    print(f"  BS Call             : ₹{call:,.2f}")
    print(f"  BS Put              : ₹{put:,.2f}")
    print(f"  MC Call (50K paths) : ₹{mc_c:,.2f}  ± ₹{mc_se:.2f}")
    print(f"  Binomial Call       : ₹{bin_c:,.2f}")
    print(f"  Put-Call Parity     : {'PASS ✓' if pcp else 'FAIL ✗'}")
    print(f"  Implied Vol (round) : {iv*100:.4f}%")
    print("──────────────────────────────────────────────")
    print(f"  Delta (call)        :  {g['delta']:.4f}")
    print(f"  Gamma               :  {g['gamma']:.6f}")
    print(f"  Theta (per day)     : ₹{g['theta']:,.2f}")
    print(f"  Vega  (per 1% vol)  : ₹{g['vega']:,.2f}")
    print(f"  Rho   (per 1% rate) : ₹{g['rho']:,.2f}")
    print("──────────────────────────────────────────────")
    print(f"  BS throughput       : {bs_per_sec:>12,.0f} prices/sec")
    print(f"  Greeks throughput   : {greeks_per_sec:>12,.0f} greek-sets/sec")
    print(f"  MC throughput       : {mc_per_sec:>12,.0f} prices/sec (1K paths)")
    print(f"  Binomial throughput : {bin_per_sec:>12,.0f} prices/sec (100 steps)")
    print("══════════════════════════════════════════════")
    print()
    print("  v1 known issues (improvement path for v2):")
    print("  • scalar math — no numpy vectorization")
    print("  • d1/d2 recomputed per greek call")
    print("  • MC uses Python loop — 100x faster with numpy")
    print("  • binomial uses Python lists — numpy in v2")
    print("  • Newton-Raphson IV — Brent's method in v2")
    print("══════════════════════════════════════════════")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
        S, K, r, sigma, T = 24000, 24000, 0.065, 0.15, 30/365
        t0 = time.perf_counter()
        for _ in range(N):
            bs_call(S, K, r, sigma, T)
        t1 = time.perf_counter()
        rate = N / (t1 - t0)
        print(f"BS calls: {N}  Time: {t1-t0:.3f}s  Rate: {rate:,.0f} vals/sec")
    else:
        run_benchmark()
