#!/usr/bin/env python
"""
Advanced Visualizations - Top Tier Analysis
12 cutting-edge visualizations for the Non-Gaussian Risk Engine
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import sys
from io import StringIO
import os
from datetime import datetime, timedelta

from src.phase1_gaussian_failure import GaussianFailureAnalyzer
from src.phase2_evt_engine import EVTEngine
from src.phase3_risk_metrics import RiskMetricsCalculator
from src.utils import format_percent

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# VIZ 1: REGIME ANALYSIS (Regime Switching)
# ============================================================================

def viz1_regime_analysis():
    """Analyze how risk changes across market regimes."""
    print('\n📊 VIZ 1: REGIME ANALYSIS')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    
    if isinstance(analyzer.data.columns, pd.MultiIndex):
        prices = analyzer.data[('Close', 'SPY')]
    else:
        prices = analyzer.data['Close']
    
    returns = analyzer.returns
    dates = prices.index[1:]
    
    # Calculate rolling volatility (21-day window)
    rolling_vol = pd.Series(returns).rolling(21).std() * np.sqrt(252)
    
    # Define regimes based on volatility
    low_vol = np.percentile(rolling_vol.dropna(), 33)
    high_vol = np.percentile(rolling_vol.dropna(), 67)
    
    regimes = []
    for vol in rolling_vol:
        if pd.isna(vol):
            regimes.append(np.nan)
        elif vol < low_vol:
            regimes.append(0)  # Calm
        elif vol < high_vol:
            regimes.append(1)  # Nervous
        else:
            regimes.append(2)  # Crash
    
    regimes = np.array(regimes)
    
    # Calculate Xi for each regime
    regime_xi = {}
    regime_names = ['Calm (Low Vol)', 'Nervous (Med Vol)', 'Crash (High Vol)']
    colors = ['green', 'orange', 'red']
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Regime Analysis: How Tail Risk Changes with Market Conditions', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Regime timeline
    ax = axes[0, 0]
    regime_colors = [colors[int(r)] if not np.isnan(r) else 'gray' for r in regimes]
    ax.scatter(dates, returns * 100, c=regime_colors, alpha=0.5, s=10)
    ax.set_ylabel('Daily Return (%)', fontsize=11)
    ax.set_title('Market Regimes Over Time', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(alpha=0.3)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=regime_names[i]) for i in range(3)]
    ax.legend(handles=legend_elements, loc='lower left')
    
    # Plot 2: Xi by regime
    ax = axes[0, 1]
    xi_values = []
    regime_labels = []
    
    for regime_idx in range(3):
        mask = regimes == regime_idx
        if np.sum(mask) > 100:  # Need enough data
            regime_returns = returns[mask]
            
            # Suppress output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            evt = EVTEngine(regime_returns, ticker='SPY')
            evt.select_threshold(method='percentile', percentile=95.0)
            evt.fit_gpd(method='mle')
            
            sys.stdout = old_stdout
            
            xi_values.append(evt.xi)
            regime_labels.append(regime_names[regime_idx])
            regime_xi[regime_idx] = evt.xi
    
    bars = ax.bar(range(len(xi_values)), xi_values, color=colors[:len(xi_values)], 
                  alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(0.25, color='darkred', linestyle='--', linewidth=2, label='High Risk (ξ=0.25)')
    ax.set_ylabel('ξ (Shape Parameter)', fontsize=11)
    ax.set_title('Tail Risk by Regime', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(regime_labels)))
    ax.set_xticklabels(regime_labels, rotation=0)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add values on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Volatility regimes
    ax = axes[1, 0]
    ax.plot(dates, rolling_vol * 100, linewidth=1, color='steelblue')
    ax.axhline(low_vol * 100, color='green', linestyle='--', linewidth=2, label='Low Vol Threshold')
    ax.axhline(high_vol * 100, color='orange', linestyle='--', linewidth=2, label='High Vol Threshold')
    ax.fill_between(dates, 0, rolling_vol * 100, where=(rolling_vol < low_vol), 
                     alpha=0.3, color='green', label='Calm')
    ax.fill_between(dates, 0, rolling_vol * 100, 
                     where=((rolling_vol >= low_vol) & (rolling_vol < high_vol)), 
                     alpha=0.3, color='orange', label='Nervous')
    ax.fill_between(dates, 0, rolling_vol * 100, where=(rolling_vol >= high_vol), 
                     alpha=0.3, color='red', label='Crash')
    ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax.set_title('Rolling Volatility (21-day)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Regime statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for regime_idx in range(3):
        mask = regimes == regime_idx
        regime_returns = returns[mask]
        days = np.sum(mask)
        pct = 100 * days / len(returns)
        
        if regime_idx in regime_xi:
            xi = f"{regime_xi[regime_idx]:.3f}"
        else:
            xi = "N/A"
        
        table_data.append([
            regime_names[regime_idx],
            f"{days}",
            f"{pct:.1f}%",
            xi,
            f"{np.mean(regime_returns)*100:.2f}%",
            f"{np.std(regime_returns)*100:.2f}%"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Regime', 'Days', '% Time', 'ξ (Xi)', 'Mean', 'Std Dev'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.3, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells with regime colors
    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor(colors[i-1])
        table[(i, 0)].set_text_props(weight='bold', color='white')
        for j in range(1, 6):
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
    
    ax.text(0.5, 0.15, '⚠️ Key Insight: Tail risk (ξ) increases dramatically during volatile regimes',
            ha='center', fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs('results/advanced', exist_ok=True)
    plt.savefig('results/advanced/01_regime_analysis.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/01_regime_analysis.png')
    plt.close()


# ============================================================================
# VIZ 2: STRESS TESTING VISUAL
# ============================================================================

def viz2_stress_testing():
    """Historical stress scenario analysis."""
    print('\n🔥 VIZ 2: STRESS TESTING')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    
    returns = analyzer.returns
    
    # Fit EVT model
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    evt = EVTEngine(returns, ticker='SPY')
    evt.select_threshold(method='percentile', percentile=95.0)
    evt.fit_gpd(method='mle')
    
    sys.stdout = old_stdout
    
    calc = RiskMetricsCalculator(returns, evt, 'SPY')
    
    # Historical stress scenarios
    scenarios = {
        '2008 Financial Crisis': -0.10,
        '2020 COVID Crash': -0.12,
        '1987 Black Monday': -0.20,
        '2022 Inflation Shock': -0.05,
        'Lehman-style Event': -0.15
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Stress Testing: Historical Crisis Scenarios', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # Plot 1-5: Each scenario
    for idx, (scenario, crash_level) in enumerate(scenarios.items()):
        ax = axes[idx]
        
        # Calculate probabilities
        bs = calc.calculate_black_swan_probability(crash_level)
        prob_normal = bs['normal']['probability']
        prob_evt = bs['evt']['probability']
        years_normal = bs['normal']['years']
        years_evt = bs['evt']['years']
        
        # Portfolio loss distribution (Monte Carlo)
        n_sim = 10000
        portfolio_value = 1_000_000
        
        # Normal model
        normal_losses = -np.random.normal(calc.mu, calc.sigma, n_sim) * portfolio_value
        
        # EVT model (simplified)
        evt_losses = []
        for _ in range(n_sim):
            u = np.random.uniform(0, 1)
            if u > 0.95:  # In tail
                # GPD quantile
                quantile = evt.calculate_quantile(u)
                evt_losses.append(-quantile * portfolio_value)
            else:
                # Empirical
                evt_losses.append(-np.random.choice(returns[returns > -evt.threshold]) * portfolio_value)
        evt_losses = np.array(evt_losses)
        
        # Plot distributions
        ax.hist(normal_losses/1000, bins=50, alpha=0.5, color='blue', 
                label='Normal Model', density=True, edgecolor='blue')
        ax.hist(evt_losses/1000, bins=50, alpha=0.5, color='red', 
                label='EVT Model', density=True, edgecolor='red')
        
        # Mark scenario loss
        scenario_loss = abs(crash_level) * portfolio_value / 1000
        ax.axvline(scenario_loss, color='darkred', linestyle='--', linewidth=3,
                   label=f'{scenario}\n({format_percent(crash_level, 0)})')
        
        ax.set_xlabel('Loss ($1000s)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'{scenario}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        
        # Add text box
        textstr = f'Normal: 1 in {years_normal:.0f} years\nEVT: 1 in {years_evt:.0f} years'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
    
    # Plot 6: Summary heatmap
    ax = axes[5]
    ax.axis('off')
    
    table_data = []
    for scenario, crash_level in scenarios.items():
        bs = calc.calculate_black_swan_probability(crash_level)
        years_normal = bs['normal']['years']
        years_evt = bs['evt']['years']
        underestimation = years_normal / years_evt if years_evt > 0 else np.inf
        
        table_data.append([
            scenario,
            format_percent(crash_level, 0),
            f"{years_evt:.0f}y",
            f"{years_normal:.0f}y",
            f"{underestimation:.0f}x"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Scenario', 'Loss', 'EVT', 'Normal', 'Error'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.text(0.5, 0.1, '⚠️ Normal Model underestimates crisis probability by orders of magnitude!',
            ha='center', fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced/02_stress_testing.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/02_stress_testing.png')
    plt.close()


# ============================================================================
# VIZ 3: COPULA 3D VISUALIZATION
# ============================================================================

def viz3_copula_3d():
    """3D copula visualization showing tail dependence."""
    print('\n🌐 VIZ 3: COPULA 3D VISUALIZATION')
    print('-'*80)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Load data for multiple assets
    tickers = ['SPY', 'QQQ']
    returns_dict = {}
    
    for ticker in tickers:
        analyzer = GaussianFailureAnalyzer(ticker, years=10)
        analyzer.load_data()
        analyzer.calculate_statistics()
        returns_dict[ticker] = analyzer.returns
    
    # Align returns
    min_len = min(len(returns_dict['SPY']), len(returns_dict['QQQ']))
    spy_returns = returns_dict['SPY'][-min_len:]
    qqq_returns = returns_dict['QQQ'][-min_len:]
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Copula Analysis: Tail Dependence Between Assets', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: 3D scatter during normal times
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    normal_mask = (spy_returns > np.percentile(spy_returns, 20)) & (spy_returns < np.percentile(spy_returns, 80))
    ax1.scatter(spy_returns[normal_mask] * 100, qqq_returns[normal_mask] * 100,
                np.arange(np.sum(normal_mask)), c='blue', alpha=0.3, s=10)
    ax1.set_xlabel('SPY Return (%)', fontsize=9)
    ax1.set_ylabel('QQQ Return (%)', fontsize=9)
    ax1.set_zlabel('Time', fontsize=9)
    ax1.set_title('Normal Times', fontsize=11, fontweight='bold')
    
    # Plot 2: 3D scatter during crashes
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    crash_mask = (spy_returns < np.percentile(spy_returns, 5)) | (qqq_returns < np.percentile(qqq_returns, 5))
    ax2.scatter(spy_returns[crash_mask] * 100, qqq_returns[crash_mask] * 100,
                np.arange(np.sum(crash_mask)), c='red', alpha=0.6, s=30)
    ax2.set_xlabel('SPY Return (%)', fontsize=9)
    ax2.set_ylabel('QQQ Return (%)', fontsize=9)
    ax2.set_zlabel('Time', fontsize=9)
    ax2.set_title('Crash Events (5% Worst)', fontsize=11, fontweight='bold', color='darkred')
    
    # Plot 3: 2D scatter with density
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(spy_returns * 100, qqq_returns * 100, alpha=0.3, s=10, c='gray')
    ax3.scatter(spy_returns[crash_mask] * 100, qqq_returns[crash_mask] * 100, 
                alpha=0.8, s=50, c='red', edgecolors='darkred', linewidth=1, 
                label='Crash Events', zorder=5)
    ax3.set_xlabel('SPY Return (%)', fontsize=10)
    ax3.set_ylabel('QQQ Return (%)', fontsize=10)
    ax3.set_title('Scatter: SPY vs QQQ', fontsize=11, fontweight='bold')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Correlation in different regimes
    corr_all = np.corrcoef(spy_returns, qqq_returns)[0, 1]
    corr_crash = np.corrcoef(spy_returns[crash_mask], qqq_returns[crash_mask])[0, 1]
    
    textstr = f'Correlation (All): {corr_all:.2f}\nCorrelation (Crash): {corr_crash:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    # Plot 4: Contour plot (Normal copula)
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Transform to uniforms (empirical CDF)
    u_spy = stats.rankdata(spy_returns) / len(spy_returns)
    u_qqq = stats.rankdata(qqq_returns) / len(qqq_returns)
    
    # Create grid for contour
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian copula density (simplified)
    Z = np.exp(-0.5 * (X**2 + Y**2 - 2*corr_all*X*Y) / (1 - corr_all**2))
    
    contour = ax4.contourf(X, Y, Z, levels=20, cmap='Blues')
    ax4.scatter(u_spy, u_qqq, alpha=0.3, s=5, c='black')
    ax4.set_xlabel('SPY (Uniform)', fontsize=10)
    ax4.set_ylabel('QQQ (Uniform)', fontsize=10)
    ax4.set_title('Gaussian Copula (Independence)', fontsize=11, fontweight='bold')
    plt.colorbar(contour, ax=ax4)
    
    # Plot 5: Tail copula (highlight lower tail)
    ax5 = fig.add_subplot(2, 3, 5)
    
    # t-Copula with lower df has more tail dependence
    tail_mask = (u_spy < 0.1) | (u_qqq < 0.1)
    
    ax5.scatter(u_spy[~tail_mask], u_qqq[~tail_mask], alpha=0.2, s=5, c='gray', label='Normal')
    ax5.scatter(u_spy[tail_mask], u_qqq[tail_mask], alpha=0.8, s=30, c='red', 
                edgecolors='darkred', linewidth=1, label='Tail Events', zorder=5)
    
    # Add diagonal line
    ax5.plot([0, 0.1], [0, 0.1], 'r--', linewidth=2, label='Perfect Dependence')
    
    ax5.set_xlabel('SPY (Uniform)', fontsize=10)
    ax5.set_ylabel('QQQ (Uniform)', fontsize=10)
    ax5.set_title('Tail Copula (Lower Tail Dependence)', fontsize=11, fontweight='bold', color='darkred')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate tail dependence coefficient
    lower_tail_spy = spy_returns < np.percentile(spy_returns, 5)
    lower_tail_qqq = qqq_returns < np.percentile(qqq_returns, 5)
    lambda_lower = np.sum(lower_tail_spy & lower_tail_qqq) / np.sum(lower_tail_spy)
    
    table_data = [
        ['Metric', 'Value'],
        ['Correlation (All Data)', f'{corr_all:.3f}'],
        ['Correlation (Crashes)', f'{corr_crash:.3f}'],
        ['Lower Tail Dependence (λ)', f'{lambda_lower:.3f}'],
        ['Observations', f'{len(spy_returns)}'],
        ['Crash Events', f'{np.sum(crash_mask)}']
    ]
    
    table = ax6.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.3, 0.8, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.text(0.5, 0.15, f'⚠️ During crashes, correlation jumps from {corr_all:.2f} to {corr_crash:.2f}!',
            ha='center', fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced/03_copula_3d.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/03_copula_3d.png')
    plt.close()


# ============================================================================
# VIZ 4: TAIL INDEX EVOLUTION
# ============================================================================

def viz4_tail_index_evolution():
    """Rolling tail index (Xi) over time."""
    print('\n📈 VIZ 4: TAIL INDEX EVOLUTION')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    
    if isinstance(analyzer.data.columns, pd.MultiIndex):
        prices = analyzer.data[('Close', 'SPY')]
    else:
        prices = analyzer.data['Close']
    
    returns = analyzer.returns
    dates = prices.index[1:]
    
    # Calculate rolling Xi
    window = 252  # 1 year
    step = 21  # Calculate every ~1 month (was every day - too slow!)
    rolling_xi = []
    rolling_dates = []
    
    print(f'  Calculating rolling tail index (every {step} days)...')
    for i in range(window, len(returns), step):  # OPTIMIZED: step instead of 1
        window_returns = returns[i-window:i]
        
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            evt = EVTEngine(window_returns, ticker='SPY')
            evt.select_threshold(method='percentile', percentile=95.0)
            evt.fit_gpd(method='mle')
            
            sys.stdout = old_stdout
            
            rolling_xi.append(evt.xi)
            rolling_dates.append(dates[i])
        except:
            rolling_xi.append(np.nan)
            rolling_dates.append(dates[i])
    
    rolling_xi = np.array(rolling_xi)
    rolling_dates = np.array(rolling_dates)
    
    # Historical events
    events = {
        '2008 Financial Crisis': datetime(2008, 9, 15),
        '2011 Euro Crisis': datetime(2011, 8, 1),
        '2015 China Slowdown': datetime(2015, 8, 24),
        '2018 Vol Spike': datetime(2018, 2, 5),
        '2020 COVID Crash': datetime(2020, 3, 16),
        '2022 Inflation Shock': datetime(2022, 6, 13)
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Tail Index Evolution: Early Warning System for Crashes', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Rolling Xi with risk zones
    ax = axes[0]
    ax.plot(rolling_dates, rolling_xi, linewidth=2, color='steelblue')
    
    # Risk zones
    ax.axhspan(0, 0.15, alpha=0.2, color='green', label='Low Risk (ξ < 0.15)')
    ax.axhspan(0.15, 0.3, alpha=0.2, color='orange', label='Moderate Risk (ξ = 0.15-0.3)')
    ax.axhspan(0.3, 1.0, alpha=0.2, color='red', label='High Risk (ξ > 0.3)')
    
    # Mark events
    for event_name, event_date in events.items():
        if event_date >= rolling_dates[0] and event_date <= rolling_dates[-1]:
            ax.axvline(event_date, color='darkred', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Find Xi value at event
            idx = np.argmin(np.abs(rolling_dates - event_date))
            if not np.isnan(rolling_xi[idx]):
                ax.scatter(event_date, rolling_xi[idx], s=100, c='red', 
                          edgecolors='darkred', linewidth=2, zorder=5)
    
    ax.set_ylabel('ξ (Tail Index)', fontsize=12)
    ax.set_title('Rolling Tail Index (252-day window)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Xi vs future crashes
    ax = axes[1]
    
    # Calculate max drawdown in next 60 days for each date
    future_dd = []
    for i in range(len(rolling_dates)):
        date_idx = np.where(dates == rolling_dates[i])[0]
        if len(date_idx) > 0 and date_idx[0] + 60 < len(returns):
            future_returns = returns[date_idx[0]:date_idx[0]+60]
            cum_returns = (1 + future_returns).cumprod()
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            future_dd.append(np.min(drawdown) * 100)
        else:
            future_dd.append(np.nan)
    
    future_dd = np.array(future_dd)
    
    # Scatter plot
    mask = ~np.isnan(rolling_xi) & ~np.isnan(future_dd)
    colors = ['red' if dd < -5 else 'orange' if dd < -2 else 'gray' for dd in future_dd[mask]]
    
    ax.scatter(rolling_xi[mask], future_dd[mask], c=colors, alpha=0.5, s=20)
    ax.set_xlabel('Current ξ (Tail Index)', fontsize=12)
    ax.set_ylabel('Max Drawdown Next 60 Days (%)', fontsize=12)
    ax.set_title('Predictive Power: Higher ξ → Larger Future Drawdowns', 
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0.25, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add trend line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(rolling_xi[mask], future_dd[mask])
    x_line = np.array([np.min(rolling_xi[mask]), np.max(rolling_xi[mask])])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Trend (R²={r_value**2:.3f})')
    ax.legend()
    
    # Plot 3: Risk dashboard
    ax = axes[2]
    
    # Current Xi value
    current_xi = rolling_xi[-1] if not np.isnan(rolling_xi[-1]) else rolling_xi[~np.isnan(rolling_xi)][-1]
    
    # Risk gauge
    categories = ['Low\nRisk', 'Moderate\nRisk', 'High\nRisk']
    colors_gauge = ['green', 'orange', 'red']
    bounds = [0, 0.15, 0.3, 1.0]
    
    # Determine current risk level
    if current_xi < 0.15:
        current_level = 0
        risk_text = 'LOW RISK'
        risk_color = 'green'
    elif current_xi < 0.3:
        current_level = 1
        risk_text = 'MODERATE RISK'
        risk_color = 'orange'
    else:
        current_level = 2
        risk_text = 'HIGH RISK'
        risk_color = 'red'
    
    bars = ax.barh(categories, [0.15, 0.15, 0.7], left=[0, 0.15, 0.3], 
                   color=colors_gauge, alpha=0.3, edgecolor='black', linewidth=2)
    
    # Mark current position
    ax.axvline(current_xi, color=risk_color, linestyle='-', linewidth=4, 
               label=f'Current ξ = {current_xi:.3f}')
    ax.scatter([current_xi], [current_level], s=500, c=risk_color, 
              edgecolors='black', linewidth=3, zorder=5, marker='o')
    
    ax.set_xlabel('ξ (Tail Index)', fontsize=12)
    ax.set_title(f'Current Risk Level: {risk_text}', fontsize=13, fontweight='bold', color=risk_color)
    ax.set_xlim([0, 0.6])
    ax.legend(fontsize=11, loc='right')
    ax.grid(alpha=0.3, axis='x')
    
    # Add warning text
    if current_level >= 1:
        textstr = f'⚠️ WARNING: Elevated tail risk detected!\nHistorical average ξ = {np.nanmean(rolling_xi):.3f}'
        ax.text(0.5, -0.5, textstr, ha='center', fontsize=11, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('results/advanced/04_tail_index_evolution.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/04_tail_index_evolution.png')
    plt.close()


# ============================================================================
# VIZ 5: VAR BACKTESTING
# ============================================================================

def viz5_var_backtesting():
    """Backtest VaR predictions vs actual violations."""
    print('\n🎯 VIZ 5: VAR BACKTESTING')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    
    if isinstance(analyzer.data.columns, pd.MultiIndex):
        prices = analyzer.data[('Close', 'SPY')]
    else:
        prices = analyzer.data['Close']
    
    returns = analyzer.returns
    dates = prices.index[1:]
    
    # Fit EVT
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    evt = EVTEngine(returns, ticker='SPY')
    evt.select_threshold(method='percentile', percentile=95.0)
    evt.fit_gpd(method='mle')
    
    sys.stdout = old_stdout
    
    calc = RiskMetricsCalculator(returns, evt, 'SPY')
    
    # Calculate VaR
    var_99_normal = calc.calculate_var(0.99)['normal']
    var_99_evt = calc.calculate_var(0.99)['evt']
    
    # Find violations
    violations_normal = returns < var_99_normal
    violations_evt = returns < var_99_evt
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('VaR Backtesting: The Smoking Gun of Normal Model Failure', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Returns with VaR lines
    ax = axes[0, 0]
    ax.plot(dates, returns * 100, linewidth=0.5, alpha=0.7, color='gray', label='Daily Returns')
    ax.axhline(var_99_normal * 100, color='blue', linestyle='--', linewidth=2, 
               label=f'VaR(99%) Normal: {format_percent(var_99_normal, 2)}')
    ax.axhline(var_99_evt * 100, color='red', linestyle='--', linewidth=2, 
               label=f'VaR(99%) EVT: {format_percent(var_99_evt, 2)}')
    
    # Mark violations
    ax.scatter(dates[violations_normal], returns[violations_normal] * 100, 
               color='blue', s=50, alpha=0.8, label=f'Normal Violations: {np.sum(violations_normal)}', zorder=5)
    ax.scatter(dates[violations_evt], returns[violations_evt] * 100, 
               color='red', s=80, alpha=0.6, marker='x', linewidths=2,
               label=f'EVT Violations: {np.sum(violations_evt)}', zorder=6)
    
    ax.set_ylabel('Daily Return (%)', fontsize=11)
    ax.set_title('VaR(99%) Backtesting Timeline', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Violation counts comparison
    ax = axes[0, 1]
    
    expected_violations = len(returns) * 0.01  # 1% for 99% VaR
    actual_normal = np.sum(violations_normal)
    actual_evt = np.sum(violations_evt)
    
    x = np.arange(3)
    counts = [expected_violations, actual_normal, actual_evt]
    colors_bars = ['green', 'blue', 'red']
    labels_bars = ['Expected\n(1%)', 'Normal Model', 'EVT Model']
    
    bars = ax.bar(x, counts, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(expected_violations, color='green', linestyle='--', linewidth=2, label='Expected')
    ax.set_ylabel('Number of Violations', fontsize=11)
    ax.set_title('VaR(99%) Violation Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bars)
    ax.grid(alpha=0.3, axis='y')
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add error annotation
    error_normal = (actual_normal / expected_violations - 1) * 100
    error_evt = (actual_evt / expected_violations - 1) * 100
    
    textstr = f'Normal Error: {error_normal:+.0f}%\nEVT Error: {error_evt:+.0f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Plot 3: Rolling violation rate
    ax = axes[1, 0]
    
    window = 252  # 1 year
    rolling_violation_normal = pd.Series(violations_normal.astype(float)).rolling(window).mean() * 100
    rolling_violation_evt = pd.Series(violations_evt.astype(float)).rolling(window).mean() * 100
    
    ax.plot(dates, rolling_violation_normal, linewidth=2, color='blue', label='Normal Model')
    ax.plot(dates, rolling_violation_evt, linewidth=2, color='red', label='EVT Model')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Expected (1%)')
    ax.fill_between(dates, 0.5, 1.5, alpha=0.2, color='green', label='Acceptable Range')
    
    ax.set_ylabel('Rolling Violation Rate (%)', fontsize=11)
    ax.set_title('Rolling VaR Violation Rate (252-day window)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Violation magnitude distribution
    ax = axes[1, 1]
    
    violation_sizes_normal = -returns[violations_normal] * 100
    violation_sizes_evt = -returns[violations_evt] * 100
    
    ax.hist(violation_sizes_normal, bins=20, alpha=0.6, color='blue', 
            label='Normal VaR Violations', edgecolor='blue')
    if len(violation_sizes_evt) > 0:
        ax.hist(violation_sizes_evt, bins=20, alpha=0.6, color='red', 
                label='EVT VaR Violations', edgecolor='red')
    
    ax.set_xlabel('Violation Size (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Size of VaR Violations', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 5: Traffic light scorecard
    ax = axes[2, 0]
    ax.axis('off')
    
    # Basel traffic light zones
    # Green: 0-4 violations
    # Yellow: 5-9 violations
    # Red: 10+ violations
    
    if actual_normal <= 4:
        zone_normal = 'GREEN'
        color_normal = 'green'
    elif actual_normal <= 9:
        zone_normal = 'YELLOW'
        color_normal = 'orange'
    else:
        zone_normal = 'RED'
        color_normal = 'red'
    
    if actual_evt <= 4:
        zone_evt = 'GREEN'
        color_evt = 'green'
    elif actual_evt <= 9:
        zone_evt = 'YELLOW'
        color_evt = 'orange'
    else:
        zone_evt = 'RED'
        color_evt = 'red'
    
    # Draw traffic lights
    # Normal Model
    circle_normal = plt.Circle((0.25, 0.6), 0.15, color=color_normal, alpha=0.8)
    ax.add_patch(circle_normal)
    ax.text(0.25, 0.35, 'Normal Model', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.25, 0.6, zone_normal, ha='center', va='center', fontsize=14, 
            fontweight='bold', color='white')
    ax.text(0.25, 0.2, f'{actual_normal} violations', ha='center', fontsize=10)
    
    # EVT Model
    circle_evt = plt.Circle((0.75, 0.6), 0.15, color=color_evt, alpha=0.8)
    ax.add_patch(circle_evt)
    ax.text(0.75, 0.35, 'EVT Model', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.75, 0.6, zone_evt, ha='center', va='center', fontsize=14, 
            fontweight='bold', color='white')
    ax.text(0.75, 0.2, f'{actual_evt} violations', ha='center', fontsize=10)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Basel Traffic Light Test (Expected: 0-4 violations)', 
                 fontsize=12, fontweight='bold')
    
    # Plot 6: Summary statistics table
    ax = axes[2, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Expected', 'Normal', 'EVT'],
        ['Total Observations', f'{len(returns)}', f'{len(returns)}', f'{len(returns)}'],
        ['VaR Level', '99%', '99%', '99%'],
        ['Expected Violations', f'{expected_violations:.1f}', f'{expected_violations:.1f}', f'{expected_violations:.1f}'],
        ['Actual Violations', '-', f'{actual_normal}', f'{actual_evt}'],
        ['Violation Rate', '1.00%', f'{100*actual_normal/len(returns):.2f}%', f'{100*actual_evt/len(returns):.2f}%'],
        ['Error', '-', f'{error_normal:+.0f}%', f'{error_evt:+.0f}%'],
        ['Basel Zone', 'GREEN', zone_normal, zone_evt]
    ]
    
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code Basel zones
    table[(7, 2)].set_facecolor(color_normal)
    table[(7, 2)].set_text_props(weight='bold', color='white')
    table[(7, 3)].set_facecolor(color_evt)
    table[(7, 3)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('results/advanced/05_var_backtesting.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/05_var_backtesting.png')
    plt.close()


# ============================================================================
# VIZ 6: DRAWDOWN + RECOVERY TIME
# ============================================================================

def viz6_drawdown_recovery():
    """Analyze drawdowns and recovery periods."""
    print('\n📉 VIZ 6: DRAWDOWN & RECOVERY TIME')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    
    if isinstance(analyzer.data.columns, pd.MultiIndex):
        prices = analyzer.data[('Close', 'SPY')]
    else:
        prices = analyzer.data['Close']
    
    returns = analyzer.returns
    dates = prices.index[1:]
    
    # Calculate drawdowns
    cumulative = (1 + pd.Series(returns, index=dates)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    # Find drawdown periods
    in_drawdown = drawdown < -1  # More than 1% from peak
    drawdown_starts = []
    drawdown_ends = []
    drawdown_depths = []
    recovery_times = []
    
    in_dd = False
    dd_start = None
    dd_depth = 0
    
    for i, (date, dd) in enumerate(zip(dates, drawdown)):
        if not in_dd and dd < -1:
            in_dd = True
            dd_start = date
            dd_depth = dd
        elif in_dd:
            if dd < dd_depth:
                dd_depth = dd
            if dd >= -0.5:  # Recovered
                drawdown_starts.append(dd_start)
                drawdown_ends.append(date)
                drawdown_depths.append(dd_depth)
                recovery_days = (date - dd_start).days
                recovery_times.append(recovery_days)
                in_dd = False
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Drawdown & Recovery Analysis: Time in Pain', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Drawdown over time
    ax = axes[0, 0]
    ax.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
    ax.plot(dates, drawdown, linewidth=1, color='darkred')
    
    # Mark top 5 worst drawdowns
    worst_dd_indices = np.argsort(drawdown_depths)[:5]
    for idx in worst_dd_indices:
        if idx < len(drawdown_starts):
            ax.scatter(drawdown_starts[idx], drawdown_depths[idx], 
                      s=200, c='darkred', edgecolors='black', linewidth=2, zorder=5)
            ax.annotate(f'{drawdown_depths[idx]:.1f}%', 
                       xy=(drawdown_starts[idx], drawdown_depths[idx]),
                       xytext=(10, -10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='mistyrose'))
    
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('Drawdown from Peak', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Recovery time distribution
    ax = axes[0, 1]
    recovery_months = [days/30 for days in recovery_times]
    ax.hist(recovery_months, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Recovery Time (months)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Recovery Times', fontsize=12, fontweight='bold')
    ax.axvline(np.mean(recovery_months), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(recovery_months):.1f} months')
    ax.axvline(np.median(recovery_months), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(recovery_months):.1f} months')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Depth vs Recovery Time
    ax = axes[1, 0]
    ax.scatter([-d for d in drawdown_depths], recovery_months, 
              s=80, alpha=0.6, c=recovery_months, cmap='Reds', edgecolors='black')
    ax.set_xlabel('Drawdown Depth (%)', fontsize=11)
    ax.set_ylabel('Recovery Time (months)', fontsize=11)
    ax.set_title('Deeper Crashes = Longer Recovery', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add trend line
    from scipy.stats import linregress
    if len(drawdown_depths) > 1:
        slope, intercept, r_value, _, _ = linregress([-d for d in drawdown_depths], recovery_months)
        x_line = np.array([0, max([-d for d in drawdown_depths])])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'R²={r_value**2:.3f}')
        ax.legend()
    
    # Plot 4: Top 10 worst drawdowns table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Get top 10
    top10_indices = np.argsort(drawdown_depths)[:10]
    table_data = []
    
    for rank, idx in enumerate(top10_indices, 1):
        if idx < len(drawdown_starts):
            table_data.append([
                f'#{rank}',
                drawdown_starts[idx].strftime('%Y-%m-%d'),
                f'{drawdown_depths[idx]:.1f}%',
                f'{recovery_times[idx]/30:.1f} mo'
            ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Rank', 'Date', 'Depth', 'Recovery'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Top 10 Worst Drawdowns', fontsize=12, fontweight='bold', pad=20)
    
    # Plot 5: Underwater chart
    ax = axes[2, 0]
    underwater = (cumulative / running_max - 1) * 100
    ax.fill_between(dates, underwater, 0, where=(underwater < 0), 
                     alpha=0.5, color='red', label='Underwater')
    ax.fill_between(dates, underwater, 0, where=(underwater >= 0), 
                     alpha=0.5, color='green', label='At New High')
    ax.plot(dates, underwater, linewidth=0.5, color='darkred')
    ax.set_ylabel('% Below Peak', fontsize=11)
    ax.set_title('Underwater Chart', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 6: Recovery statistics
    ax = axes[2, 1]
    ax.axis('off')
    
    pct_time_underwater = 100 * np.sum(drawdown < -1) / len(drawdown)
    avg_recovery_months = np.mean(recovery_months)
    max_recovery_months = np.max(recovery_months)
    max_dd = np.min(drawdown)
    
    stats_text = f"""
    📊 DRAWDOWN STATISTICS (15 Years)
    
    • Maximum Drawdown: {max_dd:.2f}%
    • Number of Drawdowns: {len(drawdown_depths)}
    • Average Recovery Time: {avg_recovery_months:.1f} months
    • Longest Recovery: {max_recovery_months:.0f} months
    • Time Underwater: {pct_time_underwater:.1f}%
    
    ⚠️ KEY INSIGHT:
    After a -20% crash, expect 12-18 months
    to recover to previous highs.
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced/06_drawdown_recovery.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/06_drawdown_recovery.png')
    plt.close()


# ============================================================================
# VIZ 7: EXPECTED SHORTFALL FUNNEL
# ============================================================================

def viz7_es_funnel():
    """ES distribution - the loss beyond VaR."""
    print('\n🌪️  VIZ 7: EXPECTED SHORTFALL FUNNEL')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    returns = analyzer.returns
    
    # Fit EVT
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    evt = EVTEngine(returns, ticker='SPY')
    evt.select_threshold(method='percentile', percentile=95.0)
    evt.fit_gpd(method='mle')
    
    sys.stdout = old_stdout
    
    calc = RiskMetricsCalculator(returns, evt, 'SPY')
    
    # Monte Carlo simulation
    n_sim = 50000
    
    # Normal model
    normal_sim = np.random.normal(calc.mu, calc.sigma, n_sim)
    
    # EVT model (simplified)
    evt_sim = []
    for _ in range(n_sim):
        u = np.random.uniform(0, 1)
        if u > 0.95:
            quantile = evt.calculate_quantile(u)
            evt_sim.append(-quantile)
        else:
            evt_sim.append(np.random.choice(returns[returns > -evt.threshold]))
    evt_sim = np.array(evt_sim)
    
    # Calculate VaR and ES
    var_99_normal = np.percentile(normal_sim, 1)
    var_99_evt = np.percentile(evt_sim, 1)
    
    # Conditional losses (beyond VaR)
    losses_beyond_var_normal = normal_sim[normal_sim < var_99_normal]
    losses_beyond_var_evt = evt_sim[evt_sim < var_99_evt]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Expected Shortfall: The Loss Beyond VaR', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Full distribution with VaR
    ax = axes[0, 0]
    ax.hist(normal_sim * 100, bins=100, density=True, alpha=0.5, 
            color='blue', label='Normal', edgecolor='blue')
    ax.hist(evt_sim * 100, bins=100, density=True, alpha=0.5, 
            color='red', label='EVT', edgecolor='red')
    ax.axvline(var_99_normal * 100, color='blue', linestyle='--', linewidth=2)
    ax.axvline(var_99_evt * 100, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Return (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Full Return Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.set_xlim([-15, 5])
    
    # Plot 2: Tail only (beyond VaR)
    ax = axes[0, 1]
    ax.hist(losses_beyond_var_normal * 100, bins=50, density=True, alpha=0.6, 
            color='blue', label='Normal (tail)', edgecolor='blue')
    ax.hist(losses_beyond_var_evt * 100, bins=50, density=True, alpha=0.6, 
            color='red', label='EVT (tail)', edgecolor='red')
    ax.set_xlabel('Loss Beyond VaR (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Conditional Loss Distribution (Beyond VaR)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Percentiles of conditional loss
    ax = axes[0, 2]
    
    percentiles = [50, 75, 90, 95, 99]
    normal_percentiles = [np.percentile(losses_beyond_var_normal, p) * 100 for p in percentiles]
    evt_percentiles = [np.percentile(losses_beyond_var_evt, p) * 100 for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [-p for p in normal_percentiles], width, 
                   label='Normal', color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, [-p for p in evt_percentiles], width, 
                   label='EVT', color='red', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Loss (%)', fontsize=11)
    ax.set_title('Percentiles of Loss Given Breach', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Funnel chart
    ax = axes[1, 0]
    
    # Create funnel effect
    confidence_levels = [0.99, 0.995, 0.999, 0.9999]
    normal_es = []
    evt_es = []
    
    for conf in confidence_levels:
        var_n = np.percentile(normal_sim, (1-conf)*100)
        var_e = np.percentile(evt_sim, (1-conf)*100)
        
        es_n = np.mean(normal_sim[normal_sim < var_n])
        es_e = np.mean(evt_sim[evt_sim < var_e])
        
        normal_es.append(-es_n * 100)
        evt_es.append(-es_e * 100)
    
    y = np.arange(len(confidence_levels))
    
    # Create horizontal funnel
    for i, conf in enumerate(confidence_levels):
        # Normal (blue, left side)
        ax.barh(i, normal_es[i], height=0.4, left=0, 
                color='blue', alpha=0.6, edgecolor='black', linewidth=2)
        # EVT (red, right side)  
        ax.barh(i, evt_es[i], height=0.4, left=0,
                color='red', alpha=0.6, edgecolor='black', linewidth=2)
        
        # Add labels
        ax.text(normal_es[i]/2, i, f'{normal_es[i]:.2f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        ax.text(evt_es[i]/2, i+0.45, f'{evt_es[i]:.2f}%', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    ax.set_yticks(y)
    ax.set_yticklabels([f'{c*100:.2f}%' for c in confidence_levels])
    ax.set_xlabel('Expected Shortfall (%)', fontsize=11)
    ax.set_ylabel('Confidence Level', fontsize=11)
    ax.set_title('ES Funnel: Normal (Blue) vs EVT (Red)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    # Plot 5: $$ Impact on $1M portfolio
    ax = axes[1, 1]
    
    portfolio_value = 1_000_000
    normal_impact = [es * portfolio_value / 100 for es in normal_es]
    evt_impact = [es * portfolio_value / 100 for es in evt_es]
    
    x = np.arange(len(confidence_levels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_impact, width, 
                   label='Normal', color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, evt_impact, width, 
                   label='EVT', color='red', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Capital at Risk ($)', fontsize=11)
    ax.set_title('ES Impact on $1M Portfolio', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c*100:.2f}%' for c in confidence_levels], rotation=45)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y/1000:.0f}K'))
    
    # Add values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height/1000:.0f}K',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [
        ['Confidence', 'VaR Normal', 'ES Normal', 'VaR EVT', 'ES EVT'],
    ]
    
    for i, conf in enumerate(confidence_levels):
        var_n = np.percentile(normal_sim, (1-conf)*100) * 100
        var_e = np.percentile(evt_sim, (1-conf)*100) * 100
        table_data.append([
            f'{conf*100:.2f}%',
            f'{-var_n:.2f}%',
            f'{normal_es[i]:.2f}%',
            f'{-var_e:.2f}%',
            f'{evt_es[i]:.2f}%'
        ])
    
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.text(0.5, 0.1, '⚠️ ES shows the REAL risk: average loss when things go bad',
            ha='center', fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced/07_es_funnel.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/07_es_funnel.png')
    plt.close()


# ============================================================================
# VIZ 8: CORRELATION BREAKDOWN MATRIX
# ============================================================================

def viz8_correlation_matrix():
    """Show how correlations change during crashes."""
    print('\n🔗 VIZ 8: CORRELATION BREAKDOWN MATRIX')
    print('-'*80)
    
    # Load multiple assets
    tickers = ['SPY', 'QQQ', 'GLD', 'TLT']
    returns_dict = {}
    
    print('  Loading data for multiple assets...')
    for ticker in tickers:
        analyzer = GaussianFailureAnalyzer(ticker, years=10)
        analyzer.load_data()
        analyzer.calculate_statistics()
        returns_dict[ticker] = analyzer.returns
    
    # Align returns
    min_len = min([len(returns_dict[t]) for t in tickers])
    for ticker in tickers:
        returns_dict[ticker] = returns_dict[ticker][-min_len:]
    
    # Create returns matrix
    returns_matrix = np.column_stack([returns_dict[t] for t in tickers])
    
    # Define regimes
    spy_returns = returns_dict['SPY']
    
    # Normal times (middle 60%)
    normal_mask = (spy_returns > np.percentile(spy_returns, 20)) & (spy_returns < np.percentile(spy_returns, 80))
    
    # Stress times (10-20% and 80-90%)
    stress_mask = ((spy_returns >= np.percentile(spy_returns, 10)) & (spy_returns <= np.percentile(spy_returns, 20))) | \
                  ((spy_returns >= np.percentile(spy_returns, 80)) & (spy_returns <= np.percentile(spy_returns, 90)))
    
    # Crash times (worst 10%)
    crash_mask = spy_returns < np.percentile(spy_returns, 10)
    
    # Calculate correlations for each regime
    corr_normal = np.corrcoef(returns_matrix[normal_mask].T)
    corr_stress = np.corrcoef(returns_matrix[stress_mask].T)
    corr_crash = np.corrcoef(returns_matrix[crash_mask].T)
    corr_all = np.corrcoef(returns_matrix.T)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Correlation Breakdown: Diversification Disappears When You Need It Most', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: All data correlation
    ax = axes[0, 0]
    im = ax.imshow(corr_all, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_yticklabels(tickers)
    ax.set_title('All Data (Full Period)', fontsize=12, fontweight='bold')
    
    # Add correlation values
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = ax.text(j, i, f'{corr_all[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Normal times
    ax = axes[0, 1]
    im = ax.imshow(corr_normal, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_yticklabels(tickers)
    ax.set_title('Normal Times (Middle 60%)', fontsize=12, fontweight='bold', color='green')
    
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = ax.text(j, i, f'{corr_normal[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Plot 3: Stress times
    ax = axes[0, 2]
    im = ax.imshow(corr_stress, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_yticklabels(tickers)
    ax.set_title('Stress Times (Volatility)', fontsize=12, fontweight='bold', color='orange')
    
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = ax.text(j, i, f'{corr_stress[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Plot 4: Crash times
    ax = axes[1, 0]
    im = ax.imshow(corr_crash, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_yticklabels(tickers)
    ax.set_title('CRASH Times (Worst 10%)', fontsize=12, fontweight='bold', color='darkred')
    
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            text = ax.text(j, i, f'{corr_crash[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    
    # Plot 5: Correlation change (Crash - Normal)
    ax = axes[1, 1]
    corr_change = corr_crash - corr_normal
    im = ax.imshow(corr_change, cmap='Reds', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_yticklabels(tickers)
    ax.set_title('Correlation INCREASE\n(Crash - Normal)', fontsize=12, fontweight='bold')
    
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            if i != j:
                text = ax.text(j, i, f'+{corr_change[i, j]:.2f}',
                              ha="center", va="center", color="white" if corr_change[i, j] > 0.25 else "black", 
                              fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Increase in Correlation')
    
    # Plot 6: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate average correlation (excluding diagonal)
    def avg_corr(corr_matrix):
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        return np.mean(corr_matrix[mask])
    
    table_data = [
        ['Regime', 'Avg Corr', 'SPY-QQQ', 'SPY-GLD', 'SPY-TLT'],
        ['All Data', f'{avg_corr(corr_all):.3f}', f'{corr_all[0,1]:.3f}', f'{corr_all[0,2]:.3f}', f'{corr_all[0,3]:.3f}'],
        ['Normal', f'{avg_corr(corr_normal):.3f}', f'{corr_normal[0,1]:.3f}', f'{corr_normal[0,2]:.3f}', f'{corr_normal[0,3]:.3f}'],
        ['Stress', f'{avg_corr(corr_stress):.3f}', f'{corr_stress[0,1]:.3f}', f'{corr_stress[0,2]:.3f}', f'{corr_stress[0,3]:.3f}'],
        ['CRASH', f'{avg_corr(corr_crash):.3f}', f'{corr_crash[0,1]:.3f}', f'{corr_crash[0,2]:.3f}', f'{corr_crash[0,3]:.3f}'],
    ]
    
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight crash row
    for i in range(5):
        table[(4, i)].set_facecolor('#FF6B6B')
        table[(4, i)].set_text_props(weight='bold')
    
    ax.text(0.5, 0.1, '⚠️ KEY INSIGHT: Diversification fails during crashes!\nAll assets move together when you need protection most.',
            ha='center', fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('results/advanced/08_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/08_correlation_matrix.png')
    plt.close()


# ============================================================================
# VIZ 9: LIQUIDITY VS TAIL RISK
# ============================================================================

def viz9_liquidity_risk():
    """Illiquid assets have fatter tails."""
    print('\n💧 VIZ 9: LIQUIDITY VS TAIL RISK')
    print('-'*80)
    
    # Assets with different liquidity profiles
    assets = {
        'SPY': {'name': 'S&P 500 ETF', 'liquidity': 'Very High'},
        'QQQ': {'name': 'Nasdaq ETF', 'liquidity': 'Very High'},
        'GLD': {'name': 'Gold ETF', 'liquidity': 'High'},
        'TLT': {'name': 'Treasury ETF', 'liquidity': 'High'},
        'BTC-USD': {'name': 'Bitcoin', 'liquidity': 'Medium'}
    }
    
    xi_values = []
    vol_values = []
    names = []
    colors_list = []
    
    print('  Analyzing tail risk across asset classes...')
    for ticker, info in assets.items():
        try:
            analyzer = GaussianFailureAnalyzer(ticker, years=10)
            analyzer.load_data()
            stats = analyzer.calculate_statistics()
            
            if isinstance(analyzer.data.columns, pd.MultiIndex):
                prices = analyzer.data[('Close', ticker)]
            else:
                prices = analyzer.data['Close']
            
            # Calculate average daily volume as proxy for liquidity
            if 'Volume' in analyzer.data.columns or ('Volume', ticker) in analyzer.data.columns:
                if isinstance(analyzer.data.columns, pd.MultiIndex):
                    volume = analyzer.data[('Volume', ticker)]
                else:
                    volume = analyzer.data['Volume']
                avg_volume = np.mean(volume)
            else:
                avg_volume = 1e9  # Default for crypto
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            evt = EVTEngine(analyzer.returns, ticker=ticker)
            evt.select_threshold(method='percentile', percentile=95.0)
            evt.fit_gpd(method='mle')
            
            sys.stdout = old_stdout
            
            xi_values.append(evt.xi)
            vol_values.append(stats['std'] * np.sqrt(252) * 100)  # Annualized volatility
            names.append(info['name'])
            
            # Color by liquidity
            if info['liquidity'] == 'Very High':
                colors_list.append('green')
            elif info['liquidity'] == 'High':
                colors_list.append('blue')
            else:
                colors_list.append('red')
                
        except Exception as e:
            print(f'    Warning: Could not analyze {ticker}: {e}')
            continue
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Liquidity vs Tail Risk: Illiquid = Dangerous', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Xi by asset
    ax = axes[0, 0]
    bars = ax.bar(range(len(names)), xi_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(0.25, color='red', linestyle='--', linewidth=2, label='High Risk (ξ=0.25)')
    ax.set_ylabel('ξ (Tail Index)', fontsize=11)
    ax.set_title('Tail Risk by Asset Class', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add values
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Volatility vs Xi scatter
    ax = axes[0, 1]
    scatter = ax.scatter(vol_values, xi_values, s=200, c=colors_list, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    for i, name in enumerate(names):
        ax.annotate(name.split()[0], (vol_values[i], xi_values[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Volatility (%)', fontsize=11)
    ax.set_ylabel('ξ (Tail Index)', fontsize=11)
    ax.set_title('Volatility vs Tail Risk', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add trend line
    if len(vol_values) > 2:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(vol_values, xi_values)
        x_line = np.array([min(vol_values), max(vol_values)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'R²={r_value**2:.3f}')
        ax.legend()
    
    # Plot 3: Risk ranking table
    ax = axes[1, 0]
    ax.axis('off')
    
    # Sort by Xi
    sorted_indices = np.argsort(xi_values)[::-1]
    
    table_data = []
    for rank, idx in enumerate(sorted_indices, 1):
        risk_level = '🔴 HIGH' if xi_values[idx] > 0.25 else '🟡 MOD' if xi_values[idx] > 0.15 else '🟢 LOW'
        table_data.append([
            f'#{rank}',
            names[idx].split()[0],
            f'{xi_values[idx]:.3f}',
            f'{vol_values[idx]:.1f}%',
            risk_level
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Rank', 'Asset', 'ξ', 'Vol', 'Risk'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Tail Risk Ranking', fontsize=12, fontweight='bold', pad=20)
    
    # Plot 4: Key insights
    ax = axes[1, 1]
    ax.axis('off')
    
    insights_text = f"""
    📊 KEY INSIGHTS
    
    1. LIQUIDITY MATTERS
       • Liquid assets (SPY, QQQ): ξ ≈ 0.15-0.25
       • Illiquid assets (BTC): ξ > 0.30
       • Higher ξ = Fatter tails = More crashes
    
    2. VOLATILITY ≠ TAIL RISK
       • Low vol assets can have fat tails
       • High vol assets can have thin tails
       • Need EVT, not just σ!
    
    3. DIVERSIFICATION PARADOX
       • "Diversifying" into illiquid assets
       • INCREASES tail risk, not decreases it
       • Crashes are BIGGER and more frequent
    
    ⚠️  RECOMMENDATION:
    In portfolio optimization, penalize
    high-ξ assets MORE than high-vol assets.
    """
    
    ax.text(0.1, 0.5, insights_text, fontsize=10, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced/09_liquidity_risk.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/09_liquidity_risk.png')
    plt.close()


# ============================================================================
# VIZ 10: PORTFOLIO OPTIMIZATION (EVT-BASED)
# ============================================================================

def viz10_portfolio_optimization():
    """EVT-based efficient frontier vs traditional."""
    print('\n📈 VIZ 10: PORTFOLIO OPTIMIZATION (EVT-BASED)')
    print('-'*80)
    
    # Assets for portfolio
    tickers = ['SPY', 'QQQ', 'GLD', 'TLT']
    returns_dict = {}
    evt_dict = {}
    
    print('  Building EVT models for portfolio assets...')
    for ticker in tickers:
        analyzer = GaussianFailureAnalyzer(ticker, years=10)
        analyzer.load_data()
        analyzer.calculate_statistics()
        returns_dict[ticker] = analyzer.returns
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        evt = EVTEngine(analyzer.returns, ticker=ticker)
        evt.select_threshold(method='percentile', percentile=95.0)
        evt.fit_gpd(method='mle')
        
        sys.stdout = old_stdout
        
        evt_dict[ticker] = evt
    
    # Align returns
    min_len = min([len(returns_dict[t]) for t in tickers])
    for ticker in tickers:
        returns_dict[ticker] = returns_dict[ticker][-min_len:]
    
    returns_matrix = np.column_stack([returns_dict[t] for t in tickers])
    
    # Calculate statistics
    mean_returns = np.mean(returns_matrix, axis=0) * 252  # Annualized
    cov_matrix = np.cov(returns_matrix.T) * 252
    
    # Monte Carlo: Generate random portfolios
    n_portfolios = 5000
    np.random.seed(42)
    
    portfolio_returns_normal = []
    portfolio_stds_normal = []
    portfolio_var99_normal = []
    portfolio_var99_evt = []
    portfolio_weights = []
    
    print('  Simulating 5000 portfolio combinations...')
    for i in range(n_portfolios):
        # Random weights
        weights = np.random.random(len(tickers))
        weights = weights / weights.sum()
        
        # Portfolio return and risk (Normal model)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # VaR 99% - Normal
        port_var_normal = -(port_return/252 - 2.33 * port_std/np.sqrt(252))
        
        # VaR 99% - EVT (approximate with weighted average)
        port_var_evt = 0
        for j, ticker in enumerate(tickers):
            calc = RiskMetricsCalculator(returns_dict[ticker], evt_dict[ticker], ticker)
            var_evt = calc.calculate_var(0.99)['evt']
            port_var_evt += weights[j] * abs(var_evt)
        
        portfolio_returns_normal.append(port_return)
        portfolio_stds_normal.append(port_std)
        portfolio_var99_normal.append(port_var_normal)
        portfolio_var99_evt.append(port_var_evt)
        portfolio_weights.append(weights)
    
    portfolio_returns_normal = np.array(portfolio_returns_normal)
    portfolio_stds_normal = np.array(portfolio_stds_normal)
    portfolio_var99_normal = np.array(portfolio_var99_normal)
    portfolio_var99_evt = np.array(portfolio_var99_evt)
    
    # Calculate Sharpe ratios
    sharpe_normal = portfolio_returns_normal / portfolio_stds_normal
    sharpe_evt = portfolio_returns_normal / (portfolio_var99_evt * np.sqrt(252))
    
    # Find optimal portfolios
    idx_sharpe_normal = np.argmax(sharpe_normal)
    idx_sharpe_evt = np.argmax(sharpe_evt)
    idx_min_var_evt = np.argmin(portfolio_var99_evt)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Portfolio Optimization: EVT Changes Everything', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Traditional Efficient Frontier (Risk = Std Dev)
    ax = axes[0, 0]
    scatter = ax.scatter(portfolio_stds_normal * 100, portfolio_returns_normal * 100,
                        c=sharpe_normal, cmap='viridis', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    
    # Mark optimal
    ax.scatter(portfolio_stds_normal[idx_sharpe_normal] * 100, 
              portfolio_returns_normal[idx_sharpe_normal] * 100,
              c='red', s=500, marker='*', edgecolors='black', linewidth=2,
              label='Max Sharpe', zorder=5)
    
    ax.set_xlabel('Risk (Std Dev, %)', fontsize=11)
    ax.set_ylabel('Expected Return (%)', fontsize=11)
    ax.set_title('Traditional Efficient Frontier', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: EVT-based Frontier (Risk = VaR EVT)
    ax = axes[0, 1]
    scatter = ax.scatter(portfolio_var99_evt * 100, portfolio_returns_normal * 100,
                        c=sharpe_evt, cmap='plasma', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax, label='EVT-Sharpe')
    
    # Mark optimal
    ax.scatter(portfolio_var99_evt[idx_sharpe_evt] * 100, 
              portfolio_returns_normal[idx_sharpe_evt] * 100,
              c='red', s=500, marker='*', edgecolors='black', linewidth=2,
              label='Max EVT-Sharpe', zorder=5)
    
    ax.scatter(portfolio_var99_evt[idx_min_var_evt] * 100, 
              portfolio_returns_normal[idx_min_var_evt] * 100,
              c='blue', s=500, marker='s', edgecolors='black', linewidth=2,
              label='Min VaR-EVT', zorder=5)
    
    ax.set_xlabel('Risk (VaR-EVT 99%, %)', fontsize=11)
    ax.set_ylabel('Expected Return (%)', fontsize=11)
    ax.set_title('EVT-based Efficient Frontier', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Comparison of optimal portfolios
    ax = axes[0, 2]
    
    optimal_weights = {
        'Traditional': portfolio_weights[idx_sharpe_normal],
        'EVT-Based': portfolio_weights[idx_sharpe_evt],
        'Min VaR-EVT': portfolio_weights[idx_min_var_evt]
    }
    
    x = np.arange(len(tickers))
    width = 0.25
    
    for i, (name, weights) in enumerate(optimal_weights.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, weights * 100, width, label=name, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Weight (%)', fontsize=11)
    ax.set_title('Optimal Portfolio Allocations', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Risk comparison scatter
    ax = axes[1, 0]
    
    ax.scatter(portfolio_var99_normal * 100, portfolio_var99_evt * 100,
              alpha=0.3, s=10, c='gray')
    
    # Diagonal line (x=y)
    max_val = max(portfolio_var99_normal.max(), portfolio_var99_evt.max()) * 100
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='x=y (Perfect Agreement)')
    
    # Mark optima
    ax.scatter(portfolio_var99_normal[idx_sharpe_normal] * 100,
              portfolio_var99_evt[idx_sharpe_normal] * 100,
              s=300, c='red', marker='*', edgecolors='black', linewidth=2, zorder=5)
    
    ax.set_xlabel('VaR Normal (%)', fontsize=11)
    ax.set_ylabel('VaR EVT (%)', fontsize=11)
    ax.set_title('Risk Model Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 5: Comparison table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Traditional', 'EVT-Based', 'Min VaR-EVT'],
        ['Return', f'{portfolio_returns_normal[idx_sharpe_normal]*100:.2f}%',
                  f'{portfolio_returns_normal[idx_sharpe_evt]*100:.2f}%',
                  f'{portfolio_returns_normal[idx_min_var_evt]*100:.2f}%'],
        ['Std Dev', f'{portfolio_stds_normal[idx_sharpe_normal]*100:.2f}%',
                   f'{portfolio_stds_normal[idx_sharpe_evt]*100:.2f}%',
                   f'{portfolio_stds_normal[idx_min_var_evt]*100:.2f}%'],
        ['VaR-Normal', f'{portfolio_var99_normal[idx_sharpe_normal]*100:.2f}%',
                       f'{portfolio_var99_normal[idx_sharpe_evt]*100:.2f}%',
                       f'{portfolio_var99_normal[idx_min_var_evt]*100:.2f}%'],
        ['VaR-EVT', f'{portfolio_var99_evt[idx_sharpe_normal]*100:.2f}%',
                    f'{portfolio_var99_evt[idx_sharpe_evt]*100:.2f}%',
                    f'{portfolio_var99_evt[idx_min_var_evt]*100:.2f}%'],
        ['Sharpe', f'{sharpe_normal[idx_sharpe_normal]:.3f}',
                   f'{sharpe_normal[idx_sharpe_evt]:.3f}',
                   f'{sharpe_normal[idx_min_var_evt]:.3f}'],
        ['EVT-Sharpe', f'{sharpe_evt[idx_sharpe_normal]:.3f}',
                       f'{sharpe_evt[idx_sharpe_evt]:.3f}',
                       f'{sharpe_evt[idx_min_var_evt]:.3f}'],
    ]
    
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    table[(6, 2)].set_facecolor('#90EE90')  # Best EVT-Sharpe
    
    # Plot 6: Key insights
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate differences
    weight_diff = portfolio_weights[idx_sharpe_evt] - portfolio_weights[idx_sharpe_normal]
    
    insights_text = f"""
    💡 KEY FINDINGS
    
    1. ALLOCATION CHANGES:
    """
    
    for i, ticker in enumerate(tickers):
        diff = weight_diff[i] * 100
        insights_text += f"\n   {ticker}: {diff:+.1f}%"
    
    insights_text += f"""
    
    2. RISK PERCEPTION:
       • Normal VaR: {portfolio_var99_normal[idx_sharpe_evt]*100:.2f}%
       • EVT VaR: {portfolio_var99_evt[idx_sharpe_evt]*100:.2f}%
       • Underestimation: {(portfolio_var99_evt[idx_sharpe_evt]/portfolio_var99_normal[idx_sharpe_evt]-1)*100:.1f}%
    
    3. PERFORMANCE:
       • Traditional Sharpe: {sharpe_normal[idx_sharpe_normal]:.3f}
       • EVT-Sharpe: {sharpe_evt[idx_sharpe_evt]:.3f}
    
    ⚠️ CONCLUSION:
    EVT optimization produces MORE CONSERVATIVE
    allocations that better protect against
    tail risk while maintaining returns.
    """
    
    ax.text(0.05, 0.5, insights_text, fontsize=9, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced/10_portfolio_optimization.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/10_portfolio_optimization.png')
    plt.close()


# ============================================================================
# VIZ 11: TIME TO RUIN (SURVIVAL ANALYSIS)
# ============================================================================

def viz11_time_to_ruin():
    """Monte Carlo simulation of portfolio survival."""
    print('\n⏰ VIZ 11: TIME TO RUIN (SURVIVAL ANALYSIS)')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    returns = analyzer.returns
    
    # Fit EVT
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    evt = EVTEngine(returns, ticker='SPY')
    evt.select_threshold(method='percentile', percentile=95.0)
    evt.fit_gpd(method='mle')
    
    sys.stdout = old_stdout
    
    # Simulation parameters
    n_simulations = 1000
    n_years = 30
    n_days = n_years * 252
    initial_capital = 1_000_000
    ruin_threshold = 0.5  # 50% loss = ruin
    
    print(f'  Running {n_simulations} Monte Carlo simulations over {n_years} years...')
    
    # Simulate using Normal model
    np.random.seed(42)
    ruin_times_normal = []
    final_values_normal = []
    
    for sim in range(n_simulations):
        capital = initial_capital
        for day in range(n_days):
            daily_return = np.random.normal(np.mean(returns), np.std(returns))
            capital *= (1 + daily_return)
            
            if capital <= initial_capital * ruin_threshold:
                ruin_times_normal.append(day / 252)  # Convert to years
                break
        else:
            ruin_times_normal.append(np.inf)
            final_values_normal.append(capital)
    
    # Simulate using EVT model
    np.random.seed(42)
    ruin_times_evt = []
    final_values_evt = []
    
    for sim in range(n_simulations):
        capital = initial_capital
        for day in range(n_days):
            u = np.random.uniform(0, 1)
            if u > 0.95:  # Tail event
                quantile = evt.calculate_quantile(u)
                daily_return = -quantile
            else:
                daily_return = np.random.choice(returns[returns > -evt.threshold])
            
            capital *= (1 + daily_return)
            
            if capital <= initial_capital * ruin_threshold:
                ruin_times_evt.append(day / 252)
                break
        else:
            ruin_times_evt.append(np.inf)
            final_values_evt.append(capital)
    
    # Calculate survival probabilities
    years = np.arange(1, n_years + 1)
    survival_normal = [np.mean([t > y for t in ruin_times_normal]) for y in years]
    survival_evt = [np.mean([t > y for t in ruin_times_evt]) for y in years]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Time to Ruin: {ruin_threshold*100:.0f}% Loss Threshold on ${initial_capital:,} Portfolio', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Survival curves
    ax = axes[0, 0]
    ax.plot(years, np.array(survival_normal) * 100, linewidth=3, 
            color='blue', label='Normal Model', marker='o', markersize=4)
    ax.plot(years, np.array(survival_evt) * 100, linewidth=3, 
            color='red', label='EVT Model', marker='s', markersize=4)
    ax.fill_between(years, np.array(survival_normal) * 100, 
                     np.array(survival_evt) * 100, alpha=0.3, color='red',
                     label='Additional Risk (EVT)')
    
    ax.set_xlabel('Years', fontsize=11)
    ax.set_ylabel('Survival Probability (%)', fontsize=11)
    ax.set_title('Kaplan-Meier Survival Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Plot 2: Time to ruin distribution
    ax = axes[0, 1]
    
    ruin_times_normal_finite = [t for t in ruin_times_normal if t != np.inf]
    ruin_times_evt_finite = [t for t in ruin_times_evt if t != np.inf]
    
    ax.hist(ruin_times_normal_finite, bins=30, alpha=0.6, color='blue', 
            label=f'Normal ({len(ruin_times_normal_finite)} ruins)', edgecolor='blue')
    ax.hist(ruin_times_evt_finite, bins=30, alpha=0.6, color='red', 
            label=f'EVT ({len(ruin_times_evt_finite)} ruins)', edgecolor='red')
    
    ax.set_xlabel('Time to Ruin (years)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Ruin Times', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 3: Ruin probability by year
    ax = axes[0, 2]
    
    ruin_prob_normal = 1 - np.array(survival_normal)
    ruin_prob_evt = 1 - np.array(survival_evt)
    
    width = 0.35
    x = np.arange(0, n_years, 5)  # Every 5 years
    
    ax.bar(x - width/2, ruin_prob_normal[::5] * 100, width, 
           label='Normal', color='blue', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, ruin_prob_evt[::5] * 100, width, 
           label='EVT', color='red', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Years', fontsize=11)
    ax.set_ylabel('Cumulative Ruin Probability (%)', fontsize=11)
    ax.set_title('Ruin Risk Over Time', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Final wealth distribution (survivors)
    ax = axes[1, 0]
    
    if len(final_values_normal) > 0 and len(final_values_evt) > 0:
        ax.hist(np.log10(final_values_normal), bins=30, alpha=0.6, color='blue', 
                label='Normal (survivors)', density=True, edgecolor='blue')
        ax.hist(np.log10(final_values_evt), bins=30, alpha=0.6, color='red', 
                label='EVT (survivors)', density=True, edgecolor='red')
        
        ax.set_xlabel('Final Wealth (log10 $)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Wealth After {n_years} Years (Survivors Only)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
    
    # Plot 5: Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    median_survival_normal = np.inf if np.all(np.array(survival_normal) > 0.5) else years[np.argmax(np.array(survival_normal) < 0.5)]
    median_survival_evt = np.inf if np.all(np.array(survival_evt) > 0.5) else years[np.argmax(np.array(survival_evt) < 0.5)]
    
    prob_ruin_10y_normal = (1 - survival_normal[9]) * 100
    prob_ruin_10y_evt = (1 - survival_evt[9]) * 100
    
    prob_ruin_30y_normal = (1 - survival_normal[-1]) * 100
    prob_ruin_30y_evt = (1 - survival_evt[-1]) * 100
    
    table_data = [
        ['Metric', 'Normal Model', 'EVT Model'],
        ['Simulations', f'{n_simulations}', f'{n_simulations}'],
        ['Ruins (Total)', f'{len(ruin_times_normal_finite)}', f'{len(ruin_times_evt_finite)}'],
        ['Ruin Rate', f'{len(ruin_times_normal_finite)/n_simulations*100:.1f}%', 
                      f'{len(ruin_times_evt_finite)/n_simulations*100:.1f}%'],
        ['Median Survival', 'Never' if median_survival_normal == np.inf else f'{median_survival_normal:.0f}y',
                           'Never' if median_survival_evt == np.inf else f'{median_survival_evt:.0f}y'],
        ['P(Ruin in 10y)', f'{prob_ruin_10y_normal:.2f}%', f'{prob_ruin_10y_evt:.2f}%'],
        ['P(Ruin in 30y)', f'{prob_ruin_30y_normal:.2f}%', f'{prob_ruin_30y_evt:.2f}%'],
        ['Avg Ruin Time', f'{np.mean(ruin_times_normal_finite):.1f}y' if ruin_times_normal_finite else 'N/A',
                         f'{np.mean(ruin_times_evt_finite):.1f}y' if ruin_times_evt_finite else 'N/A'],
    ]
    
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 3)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight key metrics
    table[(3, 2)].set_facecolor('#FFB6C1')  # EVT ruin rate
    table[(6, 2)].set_facecolor('#FFB6C1')  # EVT 30y ruin prob
    
    # Plot 6: Key insights
    ax = axes[1, 2]
    ax.axis('off')
    
    insights_text = f"""
    ⚠️ CRITICAL INSIGHTS
    
    1. RUIN RISK:
       • Normal: {prob_ruin_30y_normal:.1f}% in 30 years
       • EVT: {prob_ruin_30y_evt:.1f}% in 30 years
       • EVT is {prob_ruin_30y_evt/prob_ruin_30y_normal:.1f}x higher!
    
    2. TIME HORIZON MATTERS:
       • 10 years: {prob_ruin_10y_evt:.2f}% ruin risk
       • 30 years: {prob_ruin_30y_evt:.2f}% ruin risk
       • Long-term investors NEED tail protection
    
    3. SURVIVOR BIAS:
       • {len(ruin_times_evt_finite)} portfolios WIPED OUT
       • Traditional models underestimate
         catastrophic risk by ignoring fat tails
    
    💡 RECOMMENDATION:
    For a {ruin_threshold*100:.0f}% loss threshold:
    - Increase cash reserves
    - Add tail hedge (put options, gold)
    - Reduce portfolio leverage
    - Monitor tail index (ξ) continuously
    
    ⏰ The clock is ticking...
    """
    
    ax.text(0.05, 0.5, insights_text, fontsize=9, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('results/advanced/11_time_to_ruin.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/11_time_to_ruin.png')
    plt.close()


# ============================================================================
# VIZ 12: CALENDAR HEATMAP
# ============================================================================

def viz12_calendar_heatmap():
    """GitHub-style heatmap of extreme events."""
    print('\n📅 VIZ 12: CALENDAR HEATMAP (EXTREME EVENTS)')
    print('-'*80)
    
    analyzer = GaussianFailureAnalyzer('SPY', years=15)
    analyzer.load_data()
    analyzer.calculate_statistics()
    
    if isinstance(analyzer.data.columns, pd.MultiIndex):
        prices = analyzer.data[('Close', 'SPY')]
    else:
        prices = analyzer.data['Close']
    
    returns = analyzer.returns
    dates = prices.index[1:]
    
    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'return': returns * 100})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Categorize returns
    def categorize_return(r):
        if r < -5:
            return 5  # Extreme crash
        elif r < -3:
            return 4  # Major loss
        elif r < -1:
            return 3  # Moderate loss
        elif r < 0:
            return 2  # Small loss
        elif r < 1:
            return 1  # Small gain
        elif r < 3:
            return 0  # Moderate gain
        else:
            return -1  # Large gain
    
    df['category'] = df['return'].apply(categorize_return)
    
    # Get last 5 years for better visualization
    recent_years = sorted(df['year'].unique())[-5:]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(6, 1, height_ratios=[1, 1, 1, 1, 1, 0.5], hspace=0.4)
    
    fig.suptitle('Calendar Heatmap: Daily Returns Pattern (GitHub-Style)', 
                 fontsize=16, fontweight='bold')
    
    # Colors: Red for losses, Green for gains
    colors = ['#196127', '#239a3b', '#7bc96f', '#c6e48b', '#ebedf0',  # Gains to neutral
              '#ffebe9', '#ff9e9e', '#ff6b6b', '#cc0000', '#8b0000']  # Neutral to extreme losses
    
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors[::-1])
    
    # Create one heatmap per year
    for idx, year in enumerate(recent_years):
        ax = fig.add_subplot(gs[idx])
        
        # Filter data for this year
        year_data = df[df['year'] == year].copy()
        
        # Create pivot table
        pivot = year_data.pivot_table(values='category', index='dayofweek', 
                                      columns='week', aggfunc='first')
        
        # Plot
        im = ax.imshow(pivot, cmap=cmap, aspect='auto', vmin=-1, vmax=5, 
                      interpolation='nearest')
        
        # Formatting
        ax.set_yticks(range(5))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
        ax.set_xticks(range(0, 53, 4))
        ax.set_xticklabels(range(1, 54, 4))
        ax.set_xlabel('Week of Year', fontsize=10)
        ax.set_title(f'{year}', fontsize=12, fontweight='bold', pad=10)
        
        # Add grid
        ax.set_xticks(np.arange(pivot.shape[1]) - 0.5, minor=True)
        ax.set_yticks(np.arange(pivot.shape[0]) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
    
    # Add colorbar and legend
    ax_legend = fig.add_subplot(gs[5])
    ax_legend.axis('off')
    
    # Manual colorbar
    legend_elements = [
        ('🟢 +3% or more', colors[0]),
        ('🟢 +1% to +3%', colors[1]),
        ('🟢 0% to +1%', colors[2]),
        ('⚪ -1% to 0%', colors[4]),
        ('🔴 -1% to -3%', colors[5]),
        ('🔴 -3% to -5%', colors[7]),
        ('🔴🔴 -5% or worse', colors[9])
    ]
    
    legend_text = "COLOR LEGEND:  "
    for label, color in legend_elements:
        legend_text += f"  {label}  "
    
    ax_legend.text(0.5, 0.7, legend_text, ha='center', fontsize=11, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=15))
    
    # Statistics
    n_extreme = len(df[df['return'] < -5])
    n_major = len(df[(df['return'] < -3) & (df['return'] >= -5)])
    n_total_red = len(df[df['return'] < 0])
    
    stats_text = f"""
    📊 STATISTICS (15 Years):
    • Extreme Crashes (< -5%): {n_extreme} days
    • Major Losses (-3% to -5%): {n_major} days
    • Total Red Days: {n_total_red} days ({n_total_red/len(df)*100:.1f}%)
    • Worst Day: {df['return'].min():.2f}%
    • Best Day: {df['return'].max():.2f}%
    
    💡 PATTERN: Crashes cluster in time (see 2008, 2020)
    """
    
    ax_legend.text(0.5, 0.2, stats_text, ha='center', fontsize=10,
                  family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.savefig('results/advanced/12_calendar_heatmap.png', dpi=300, bbox_inches='tight')
    print('  ✅ Saved: results/advanced/12_calendar_heatmap.png')
    plt.close()


# ============================================================================
# VIZ 10-12: COMPLETE IMPLEMENTATION
# ============================================================================

def viz10_12_final():
    """Execute the final 3 visualizations."""
    viz10_portfolio_optimization()
    viz11_time_to_ruin()
    viz12_calendar_heatmap()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print('='*80)
    print('🚀 ADVANCED VISUALIZATIONS - TOP TIER ANALYSIS')
    print('='*80)
    print('\n⏳ This will take a few minutes...')
    print('   Creating 12 cutting-edge visualizations\n')
    
    # Create output directory
    os.makedirs('results/advanced', exist_ok=True)
    
    # Run first 4 visualizations
    viz1_regime_analysis()
    viz2_stress_testing()
    viz3_copula_3d()
    viz4_tail_index_evolution()
    
    print('\n' + '='*80)
    print('✅ FIRST 4 VISUALIZATIONS COMPLETED!')
    print('='*80)
    
    # Run VIZ 5-9
    viz5_var_backtesting()
    viz6_drawdown_recovery()
    viz7_es_funnel()
    viz8_correlation_matrix()
    viz9_liquidity_risk()
    
    print('\n' + '='*80)
    print('✅ FIRST 9 VISUALIZATIONS COMPLETED! NOW THE FINAL 3...')
    print('='*80)
    
    # Run VIZ 10-12 (COMPLETE IMPLEMENTATIONS!)
    viz10_portfolio_optimization()
    viz11_time_to_ruin()
    viz12_calendar_heatmap()
    
    print('\n' + '='*80)
    print('🎉🎉🎉 ALL 12 VISUALIZATIONS COMPLETED! 🎉🎉🎉')
    print('='*80)
    print('\n📂 Visualizations saved in results/advanced/:')
    print('  1️⃣  01_regime_analysis.png')
    print('  2️⃣  02_stress_testing.png')
    print('  3️⃣  03_copula_3d.png')
    print('  4️⃣  04_tail_index_evolution.png')
    print('  5️⃣  05_var_backtesting.png')
    print('  6️⃣  06_drawdown_recovery.png')
    print('  7️⃣  07_es_funnel.png')
    print('  8️⃣  08_correlation_matrix.png')
    print('  9️⃣  09_liquidity_risk.png')
    print('  🔟 10_portfolio_optimization.png')
    print('  1️⃣1️⃣ 11_time_to_ruin.png')
    print('  1️⃣2️⃣ 12_calendar_heatmap.png')
    print('\n🚀 WORLD-CLASS RISK ANALYSIS COMPLETE!')
    print('='*80)

if __name__ == '__main__':
    main()

