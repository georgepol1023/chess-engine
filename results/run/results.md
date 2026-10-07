Overall move-matching accuracy: **46.8%** on 99,063 held-out positions.

| Player rating | Positions | Accuracy (true rating) | Accuracy (model told 1500) |
|---|---|---|---|
| 800-999 | 4,331 | 43.9% | 43.5% |
| 1000-1199 | 8,067 | 45.9% | 45.3% |
| 1200-1399 | 13,444 | 46.8% | 46.7% |
| 1400-1599 | 18,095 | 46.3% | 46.3% |
| 1600-1799 | 18,752 | 47.8% | 47.7% |
| 1800-1999 | 19,036 | 47.2% | 46.8% |
| 2000-2199 | 12,557 | 47.7% | 46.7% |
| 2200-2399 | 3,796 | 46.3% | 45.3% |
| 2400-2599 | 985 | 44.3% | 43.8% |

| Clock remaining | Positions | Accuracy |
|---|---|---|
| <10s | 2,159 | 46.9% |
| 10-30s | 4,981 | 46.8% |
| 30-60s | 6,395 | 48.1% |
| >60s | 85,528 | 46.7% |

Think time: Spearman correlation 0.60 between predicted and actual, median error 1.7s.

Outcome head: predicts the final result correctly 61.5% of the time.
