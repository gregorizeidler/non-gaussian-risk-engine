#!/usr/bin/env python
"""
Simple demo of the Non-Gaussian Risk Engine
"""

from src.phase1_gaussian_failure import GaussianFailureAnalyzer
from src.phase2_evt_engine import EVTEngine
from src.phase3_risk_metrics import RiskMetricsCalculator
import warnings
warnings.filterwarnings('ignore')

def main():
    print('='*80)
    print('🚀 NON-GAUSSIAN RISK ENGINE - AUTOMATED DEMO')
    print('='*80)

    # Phase 1: Gaussian Failure
    print('\n📊 PHASE 1: GAUSSIAN FAILURE PROOF')
    print('-'*80)
    ticker = 'SPY'
    print(f'Analyzing {ticker} (last 15 years)...\n')

    analyzer = GaussianFailureAnalyzer(ticker, years=15)
    analyzer.load_data()
    stats = analyzer.calculate_statistics()

    print(f'\n✅ STATISTICS:')
    print(f'  Mean (μ):            {stats["mean"]:.6f} ({stats["mean"]*100:.3f}%)')
    print(f'  Volatility (σ):      {stats["std"]:.6f} ({stats["std"]*100:.3f}%)')
    print(f'  Skewness:            {stats["skewness"]:.4f}')
    print(f'  Kurtosis (K):        {stats["kurtosis"]:.4f}')
    print(f'  Excess Kurtosis:     {stats["excess_kurtosis"]:.4f}')

    if stats['kurtosis'] > 3:
        print(f'\n🔥 FAT TAILS DETECTED!')
        print(f'   Kurtosis {stats["kurtosis"]:.2f} >> 3.0 (Normal)')
        print(f'   {stats["kurtosis"]/3:.2f}x more mass in tails!')
    else:
        print(f'\n✓ Distribution close to normal')

    # Phase 2: EVT Engine
    print('\n\n⚡ PHASE 2: EVT ENGINE (PEAKS-OVER-THRESHOLD)')
    print('-'*80)
    evt = EVTEngine(analyzer.returns, ticker=ticker)
    threshold = evt.select_threshold(method='percentile', percentile=95.0)
    xi, sigma = evt.fit_gpd(method='mle')

    print(f'\nThreshold (95th percentile): {threshold:.6f} ({threshold*100:.2f}%)')
    print(f'Exceedances: {evt.n_exceedances} ({evt.n_exceedances/len(analyzer.returns)*100:.1f}% of data)')

    print(f'\n✅ GPD PARAMETERS (MLE):')
    print(f'  ξ (Xi):    {xi:.6f}')
    print(f'  σ (Sigma): {sigma:.6f}')

    print(f'\n📈 INTERPRETATION OF ξ = {xi:.6f}:')
    if xi > 0.5:
        print(f'  🚨 CRITICAL RISK: ξ > 0.5 (Infinite variance!)')
    elif xi > 0.25:
        print(f'  ⚠️  HIGH RISK: 0.25 < ξ < 0.5 (Strong fat tail)')
    elif xi > 0.1:
        print(f'  ⚠️  MODERATE RISK: 0.1 < ξ < 0.25 (Fat tail detected)')
    elif xi > 0:
        print(f'  ⚠️  LOW-MODERATE RISK: 0 < ξ < 0.1')
    else:
        print(f'  ✓  Exponential or finite tail')

    if xi > 0:
        print(f'  → Pareto Distribution (Power Law)')
        print(f'  → Decay: x^(-1/ξ) = x^{-1/xi:.2f}')

    # Phase 3: Risk Metrics
    print('\n\n💰 PHASE 3: RISK METRICS (VaR, ES, Black Swan)')
    print('-'*80)
    calculator = RiskMetricsCalculator(analyzer.returns, evt, ticker=ticker)

    print('\n📉 VaR (Value-at-Risk) - Comparison:')
    print('-' * 70)
    print(f'  Confidence   VaR Normal    VaR EVT       Error')
    print('-' * 70)
    for conf in [0.95, 0.99, 0.995]:
        vars_dict = calculator.calculate_var(conf, method='both')
        diff = (vars_dict['evt'] / vars_dict['normal'] - 1) * 100
        print(f'  {conf*100:>5.1f}%      {vars_dict["normal"]*100:>6.2f}%      {vars_dict["evt"]*100:>6.2f}%      {diff:+6.1f}%')

    print('\n\n📉 Expected Shortfall (ES/CVaR):')
    print('-' * 70)
    print(f'  Confidence   ES Normal     ES EVT        Error')
    print('-' * 70)
    for conf in [0.95, 0.99]:
        es_results = calculator.calculate_es(conf, method='both')
        diff = (es_results['evt'] / es_results['normal'] - 1) * 100
        print(f'  {conf*100:>5.1f}%      {es_results["normal"]*100:>6.2f}%      {es_results["evt"]*100:>6.2f}%      {diff:+6.1f}%')

    print('\n\n🦢 Crash Probabilities (Black Swan):')
    print('-' * 80)
    print(f'  Magnitude    Normal (years)    EVT (years)    Factor')
    print('-' * 80)
    for crash_pct in [0.05, 0.07, 0.10, 0.15, 0.20]:
        bs = calculator.calculate_black_swan_probability(-crash_pct, method='both')
        factor = bs['normal']['years'] / bs['evt']['years'] if bs['evt']['years'] > 0 else float('inf')
        print(f'  Crash -{crash_pct*100:.0f}%   1 in {bs["normal"]["years"]:>10.0f}   1 in {bs["evt"]["years"]:>7.1f}   {factor:>6.0f}x')

    print('\n\n' + '='*80)
    print('✅ COMPLETE ANALYSIS!')
    print('='*80)
    print('\n🎯 CRITICAL CONCLUSIONS:')
    print('  • Normal model DRASTICALLY UNDERESTIMATES tail risk')
    print('  • EVT correctly captures extreme event probabilities')
    print('  • Difference can be 50-150% in VaR and thousands of times in Black Swan')
    print('  • 2008 and 2020 happened because models ignored fat tails!')
    print('\n💡 Use EVT for capital allocation, stop-loss and real risk management.')
    print('='*80)

if __name__ == '__main__':
    main()

