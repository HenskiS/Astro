# ACTIVE MANAGEMENT ANALYSIS - KEY FINDINGS

## Summary

Analyzed 14,636 trades to understand how winners and losers behave over time and identify optimal stop loss levels.

## Maximum Adverse Excursion (MAE) Analysis

### Current Stop Loss Performance

| Stop Level | Winners Cut | Winners Kept | Losers Cut | Net Effect |
|------------|-------------|--------------|------------|------------|
| **-4.0% (current)** | 72 (0.7%) | 9,675 (99.3%) | 1,134 | **+6,363%** ✓ |
| -3.5% | 127 (1.3%) | 9,620 (98.7%) | 1,573 | +4,550% |
| -3.0% | 199 (2.0%) | 9,548 (98.0%) | 2,125 | +2,568% |
| -2.5% | 345 (3.5%) | 9,402 (96.5%) | 2,856 | +141% |
| -2.0% | 666 (6.8%) | 9,081 (93.2%) | 3,593 | **-2,464%** ✗ |

**Key Finding:** The current -4% emergency stop is nearly optimal, preserving 99.3% of winners.

### Winner MAE Distribution

- **50.1%** of winners never go more than 0.5% underwater
- **49.9%** of winners go underwater >0.5% before winning (need breathing room!)
- **Average winner MAE:** -0.75%
- **Median winner MAE:** -0.50%
- **95th percentile:** -2.24% (5% of winners go this deep before winning)

### Loser MAE Distribution

- **Average loser MAE:** -3.12%
- **Median loser MAE:** -2.77%
- **95th percentile:** -6.44%

**Insight:** Losers go much deeper underwater than winners. The -4% stop effectively separates them.

## Time-Based Behavior Analysis

### Early Divergence

Winners and losers diverge **by day 3**:

| Day | Winners Avg | Losers Avg | Difference |
|-----|-------------|------------|------------|
| **Day 1** | +0.10% | -0.16% | +0.26% |
| **Day 3** | +0.28% | -0.42% | **+0.71%** |
| **Day 5** | +0.43% | -0.64% | +1.07% |
| **Day 7** | +0.56% | -0.82% | +1.38% |
| **Day 10** | +0.72% | -1.03% | +1.75% |

### Early Prediction Power

**By Day 5:**
- 74.0% of winners are profitable
- 78.7% of losers are underwater
- Clear separation, but **26% of winners still underwater!**

**By Day 7:**
- 79.3% of winners are profitable
- 81.4% of losers are underwater
- 20.7% of winners still need more time

### Conditional Win Rates by Day 5 P&L

| Day 5 P&L | Win Rate | Interpretation |
|-----------|----------|----------------|
| **Down >2%** | 11.0% | Almost always losers |
| **Down 1-2%** | 20.8% | Mostly losers |
| **Down 0.5-1%** | 31.9% | Likely losers |
| **Down 0-0.5%** | 45.3% | Coin flip |
| **Up 0-0.5%** | 70.9% | Likely winners |
| **Up 0.5-1%** | 89.7% | Very likely winners |
| **Up >1%** | 95.4% | Almost certainly winners |

### Exit Timing

- **Winners:** Exit in 8.7 days avg (6 days median)
- **Losers:** Take 29.9 days avg (30 days median) - hit the 15-day emergency stop

## Tested Stop Rules

| Rule | Winners Cut | Losers Cut | Verdict |
|------|-------------|------------|---------|
| Day 5: -1.5% | 127 | 822 | ✗ Cuts too many winners |
| Day 5: -2.0% | 44 | 357 | ✗ Still hurts winners |
| Day 7: -1.5% | 119 | 1,234 | ✗ Best ratio but still negative |
| Day 7: -2.0% | 50 | 648 | ✗ Not worth it |
| **Day 10: -2.5%** | **10** | **578** | **✓ Potentially beneficial** |

## Recommendations

### 1. KEEP Current -4% Emergency Stop ✓

**Reasoning:**
- Preserves 99.3% of winners (only loses 72 trades)
- Cuts 1,134 losing trades effectively
- Net effect: +6,363% over the backtest
- Well-calibrated to the natural separation between winners/losers

**Action:** No change needed.

---

### 2. CONSIDER: Day 10 / -2.5% Early Stop (NEW)

**Proposal:** Add a tighter stop at day 10 if position is down >2.5%

**Impact:**
- Would cut only 10 winners (0.1%)
- Would cut 578 losers early (before they hit full -4% at day 15)
- Reduces capital tied up in losing positions

**Trade-off:**
- Slightly more complex logic
- Saves ~5 days on 578 trades
- Minimal impact on winners

**Action:** Test this in optimization script to measure exact impact on final capital.

---

### 3. DO NOT Use Aggressive Early Stops ✗

**Avoid:**
- Day 5 stops (too early, cuts 127-822 winners depending on level)
- Anything tighter than -2% before day 10
- Any rule that cuts >1% of winners

**Reasoning:**
- 26% of winners are still underwater at day 5
- 20.7% of winners need more than 7 days to turn positive
- Early stops hurt more than they help

---

### 4. ALTERNATIVE: Dynamic Stop by Confidence

**Idea:** Use tighter stops for lower-confidence trades

Example:
- Confidence 0.65-0.70: Use -3.5% stop (more defensive)
- Confidence 0.70-0.80: Use -4.0% stop (current)
- Confidence 0.80+: Use -4.5% stop (give more room)

**Reasoning:**
- Higher confidence trades deserve more patience
- Lower confidence trades should exit faster if moving against us
- Could optimize risk/reward by confidence bucket

**Action:** Test in optimization script.

---

### 5. ALTERNATIVE: Reduce Emergency Stop Days (15 → 12)

**Current:** 15 days @ -4%
**Proposed:** 12 days @ -4%

**Impact:**
- Exit losers 3 days earlier
- Free up capital faster
- May cut a few more winners that turn around late

**Action:** Test in optimization script to see if net positive.

---

## Key Insights

### 1. Winners Need Breathing Room
- 50% of winners never go >0.5% underwater
- But the other 50% need room to work
- Some winners go as deep as -2% before winning
- **Lesson:** Don't over-manage with tight stops

### 2. Early P&L is Predictive
- By day 5, there's clear divergence
- Down >2% on day 5 = 11% win rate (almost always a loser)
- Up >1% on day 5 = 95% win rate (almost certainly a winner)
- **Lesson:** Could use early P&L to adjust position sizing or add to winners

### 3. Time Matters
- Winners exit fast (8.7 days avg)
- Losers linger (29.9 days avg, hitting emergency stop)
- **Lesson:** Time-based exits make sense for losers, not winners

### 4. Current Stop is Well-Calibrated
- -4% at day 15 hits the sweet spot
- Losers average -3.12% MAE (caught by -4%)
- Winners average -0.75% MAE (safe from -4%)
- **Lesson:** Don't fix what isn't broken

---

## Next Steps - Priority Order

1. **✓ KEEP current settings** - They're nearly optimal
2. **TEST:** Day 10 / -2.5% early stop
3. **TEST:** Dynamic stops by confidence level
4. **TEST:** Reduce emergency stop days (15 → 12)
5. **EXPLORE:** Position scaling based on early P&L (add to winners at day 5 if up >1%)

---

## Visualization Files Generated

1. `mae_mfe_analysis.png` - MAE/MFE scatter plots and distributions
2. `time_behavior_analysis.png` - Time-based trajectory analysis

---

## Bottom Line

**The current -4% emergency stop at 15 days is nearly optimal.** It preserves 99.3% of winners while cutting losers effectively.

The only viable improvement is a **Day 10 / -2.5% early stop** that would cut bad losers 5 days earlier without significantly impacting winners (only 10 trades affected).

More aggressive early stops hurt performance by cutting winners that need time to develop.
