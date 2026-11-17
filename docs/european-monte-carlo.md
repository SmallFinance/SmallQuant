# European Monte Carlo timings

A European call priced by Monte Carlo on geometric Brownian motion. SmallQuant runs in float32 on the CPU and on the GPU. QuantLib's `MCEuropeanEngine` runs in float64 on one CPU thread. Each timed figure is the median of three repeats after one warmup. `vs QL` is the QuantLib time divided by that row's time.

Contract: spot 100, strike 100, one year, rate 5%, volatility 20%, dividend 0. Closed form **10.450584**.

Machine: Mac mini (Mac16,10), Apple M4, 10 CPU cores (4 performance, 6 efficiency), 10-core GPU, 16 GB unified memory, Metal 4. Script: `scripts/european_mc.py`.

## 100,000 paths

| Engine | Device | Threads | Steps | Price | Stderr | Seconds | Msteps/s | vs QL |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QuantLib | CPU | 1 | 1 | 10.4160 | 0.0464 | 0.0092 | 10.93 | 1.0 |
| SmallQuant | CPU | 1 | 1 | 10.4385 | 0.0465 | 0.0012 | 85.73 | 7.9 |
| SmallQuant | CPU | 10 | 1 | 10.4385 | 0.0465 | 0.0014 | 71.74 | 6.6 |
| SmallQuant | M4 GPU | — | 1 | 10.4246 | 0.0466 | 0.0061 | 16.51 | 1.5 |
| QuantLib | CPU | 1 | 64 | 10.4760 | 0.0468 | 0.5114 | 12.51 | 1.0 |
| SmallQuant | CPU | 1 | 64 | 10.4311 | 0.0464 | 0.0487 | 131.50 | 10.5 |
| SmallQuant | CPU | 10 | 64 | 10.4311 | 0.0464 | 0.0448 | 142.84 | 11.4 |
| SmallQuant | M4 GPU | — | 64 | 10.5227 | 0.0471 | 0.0134 | 476.09 | 38.1 |

One step is too small for the GPU. At 64 steps the GPU is about 3.3 times the 10-thread CPU and 38 times QuantLib.

## 1,000 paths, 1,000 steps

| Engine | Device | Threads | Price | Stderr | Seconds | Msteps/s | vs QL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QuantLib | CPU | 1 | 10.5389 | 0.4802 | 0.0771 | 12.97 | 1.0 |
| SmallQuant | CPU | 1 | 10.1265 | 0.4556 | 0.0076 | 131.62 | 10.1 |
| SmallQuant | CPU | 10 | 10.1265 | 0.4556 | 0.0068 | 146.76 | 11.3 |
| SmallQuant | M4 GPU | — | 10.1604 | 0.4354 | 0.0064 | 156.64 | 12.1 |

The GPU and the 10-thread CPU are tied. The standard error is about 0.45 because there are only 1,000 paths.

## 10,000 paths, 10,000 steps

| Engine | Device | Threads | Price | Stderr | Seconds | Msteps/s | vs QL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QuantLib | CPU | 1 | 10.3785 | 0.1464 | 7.9724 | 12.54 | 1.0 |
| SmallQuant | CPU | 1 | 10.7396 | 0.1512 | 0.8234 | 121.45 | 9.7 |
| SmallQuant | CPU | 10 | 10.7396 | 0.1512 | 0.7238 | 138.15 | 11.0 |
| SmallQuant | M4 GPU | — | 10.5494 | 0.1476 | 0.1486 | 672.83 | 53.6 |

The GPU is about 5 times the 10-thread CPU and 54 times QuantLib.

## 1,000,000 paths, 1,000 steps

This run was stopped. The float32 path tensor is 4 GB, and the geometric Brownian motion step keeps several copies. The process peaked at 18.8 GB on a 16 GB machine and moved about 10 GB into swap. No timing was recorded.
