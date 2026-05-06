# Options Pricing Engine v1

I built this to actually understand how options are priced at the hardware level — not just the theory.  
It’s a pure‑Python, correctness‑verified pricing library that calculates Black‑Scholes prices, all five Greeks, Monte Carlo simulations, and binomial trees — entirely from scratch, using only the Python standard library (no NumPy, no SciPy, no Numba). The hot path hits the CPU directly; every `exp` and `log` call is deliberate.

This is a work in progress. The roadmap is honest about what’s next.

## What it does

- **Black‑Scholes Pricing** – European call and put options with the analytical formula.
- **All Five Greeks** – Delta, Gamma, Theta, Vega, Rho computed from the same underlying parameters.
- **Put‑Call Parity Check** – every price is validated against the no‑arbitrage relationship.
- **Monte Carlo Engine** – geometric Brownian motion simulation with standard‑error reporting.
- **Binomial Tree** – Cox‑Ross‑Rubinstein tree, supports both European and American exercise.
- **Implied Volatility Solver** – Newton‑Raphson root‑finding to extract the volatility from a market price.
- **Consistent Benchmarking** – `time.perf_counter` with a warm‑up cycle; prevents dead‑code elimination.

## Benchmark – First Iteration

**Test parameters:** Nifty 50 ATM option (Strike ₹24,000, 30 DTE, 15% vol, 6.5% rate).  
**Hardware:** Intel i5‑1334U (P‑core 4.6 GHz), WSL2 Ubuntu, Python 3.12.

| Metric | Value |
|--------|-------|
| Valuations processed | 500,000 |
| Correctness (BS Call) | ₹477.72 ✓ |
| Total time | 0.229 s |
| Avg latency per valuation | ~460 ns |
| Throughput | **2.2 million valuations/sec** |

*This is the “order‑of‑magnitude” baseline. The engine is mathematically correct but deliberately unoptimised — just like the first version of my C++ matching engine. Improvement path is clear.*

## Roadmap

- [x] Pure‑Python Black‑Scholes engine (call, put, Greeks)
- [x] Put‑Call parity validation
- [x] Monte Carlo simulation (scalar loop)
- [x] Binomial tree for European and American options
- [x] Implied volatility solver (Newton‑Raphson)
- [ ] NumPy vectorisation – operate on entire arrays of strikes/vols at once
- [ ] Numba JIT – compile the hot math to machine code, remove Python call overhead
- [ ] Monte Carlo with parallel paths – leverage all CPU cores
- [ ] GPU acceleration (CuPy) – move pricing to the GPU for billions of valuations/sec
- [ ] Real‑time volatility surface calibration (SVI)
- [ ] Integration with my C++ matching engine as a pre‑trade risk layer
- [ ] Sub‑µs per valuation via pybind11 / C++ extension

## Build & Run

**Requirements**
- Python 3.10+
- No external libraries (v1 uses only standard library)

```bash
git clone https://github.com/adityatomar15/options-pricing-engine.git
cd options-pricing-engine
python3 src/v1_engine.py
Benchmark (quick test)

bash
python3 src/v1_engine.py bench 500000
What I learned building this
Implementing Black‑Scholes and its derivatives forced me to actually sit with the maths — reasoning about why each Greek behaves the way it does, not just copying formulas.
Writing a Monte Carlo pricer from scratch, even in a slow Python loop, gave me a visceral appreciation for how much compute hardware acceleration can unlock later.
Most importantly, I learned to benchmark honestly — the necessity of warm‑up cycles, preventing the compiler from eliminating dead code, and understanding the massive difference between constant‑data and real‑world workloads.

What this isn’t
This is not a production risk system. It’s a learning project built to understand the real‑world demands of quantitative software — numerical accuracy, performance measurement, and the architecture of pricing libraries.
The goal is to close the gap between “I’ve read about options” and “I can build a fast, verifiable pricing engine from the ground up.”

AI USE AND HOW MUCH
Yes, I used AI for this project as it was my first options pricing system at this level.
AI helped structure the files, suggest the Black‑Scholes implementation, and explain the mathematical concepts behind the Greeks and Monte Carlo methods.
All code was understood, tested, and verified by me.
