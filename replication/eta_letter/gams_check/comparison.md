# GAMS (NEOS) vs PyDICE-2023, eta letter

| phi | status (model/solve) | SCC 2025 GAMS | PyDICE | diff % | SCC 2050 GAMS | PyDICE | diff % | peak T GAMS | PyDICE | objective GAMS | PyDICE | max SCC path diff % (2025-2100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.95 | 2.0/1.0 | 59.0665 | 59.0667 | -0.0003 | 124.5398 | 124.5410 | -0.0010 | 2.6011 | 2.6011 | 6659.477000 | 6659.476851 | 0.0062 |
| 1.30 | 2.0/1.0 | 54.3639 | 54.3644 | -0.0008 | 127.1308 | 127.1343 | -0.0027 | 2.4655 | 2.4655 | 6070.046000 | 6070.046052 | 0.0196 |
| 1.50 | 2.0/1.0 | 56.2605 | 56.2614 | -0.0016 | 142.2441 | 142.2513 | -0.0051 | 2.3115 | 2.3115 | 20764.690000 | 20764.692005 | 0.0366 |
| 1.75 | 2.0/1.0 | 68.6944 | 68.6971 | -0.0040 | 196.7464 | 196.7695 | -0.0117 | 2.0593 | 2.0593 | 184378.376100 | 184378.376003 | 0.0781 |

phi = 1.75: values read from NEOS job 25198403 (results page in Chrome); the listing file itself was not saved here.
All four: CONOPT4, model status 2 (locally optimal), solver status 1. At phi = 1.50 CONOPT noted large unscaled residuals from scale factors (largest 7.8e-4); the solution is still reported optimal and matches PyDICE.
Summary: 2025 SCC within 0.004%, 2050 SCC within 0.012%, SCC path 2025-2100 within 0.08%, peak warming and objective identical to the printed precision.
