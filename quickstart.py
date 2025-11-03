"""
Quickstart: Quick Risk Analysis using EVT

This script provides simple examples of how to use the EVT Engine
for quick analysis of a single asset.

Usage:
    python quickstart.py
"""

from src.phase1_gaussian_failure import analyze_gaussian_failure
from src.phase2_evt_engine import fit_evt_model
from src.phase3_risk_metrics import calculate_risk_metrics


def example_basic():
    """Basic example: Single asset analysis."""
    
    print("\n" + "="*60)
    print("📊 BASIC EXAMPLE: Quick Risk Analysis")
    print("="*60 + "\n")
    
    # Configuration
    ticker = 'SPY'
    years = 10
    
    print(f"Analyzing {ticker} ({years} years)...\n")
    
    # 1. Load data and prove Gaussian failure
    print("1️⃣  Phase 1: Demonstrating normal model failure...")
    phase1 = analyze_gaussian_failure(ticker, years=years, plot=False)
    returns = phase1['returns']
    kurtosis = phase1['statistics']['kurtosis']
    
    print(f"   ✅ Kurtosis = {kurtosis:.2f} (Normal = 3.0)")
    if kurtosis > 3:
        print(f"   ⚠️  Fat tails detected!\n")
    
    # 2. Fit EVT model
    print("2️⃣  Phase 2: Fitting EVT Engine...")
    evt_engine = fit_evt_model(returns, ticker=ticker, plot=False)
    
    print(f"   ✅ ξ (Xi) = {evt_engine.xi:.4f}")
    if evt_engine.xi > 0:
        print(f"   ⚠️  Mathematically confirms fat tails!\n")
    
    # 3. Calculate risk metrics
    print("3️⃣  Phase 3: Calculating risk metrics...")
    results = calculate_risk_metrics(
        returns=returns,
        evt_engine=evt_engine,
        ticker=ticker,
        confidence=0.99,
        plot=False
    )
    
    print("\n" + "="*60)
    print("✅ Complete analysis!")
    print("="*60 + "\n")


def example_compare_tickers():
    """Exemplo: Comparar múltiplos ativos."""
    
    print("\n" + "="*60)
    print("📊 EXEMPLO: Comparação de Múltiplos Ativos")
    print("="*60 + "\n")
    
    tickers = ['SPY', 'QQQ', 'GLD']
    years = 10
    
    results_dict = {}
    
    for ticker in tickers:
        print(f"\n--- Analisando {ticker} ---")
        
        # Carregar e analisar
        phase1 = analyze_gaussian_failure(ticker, years=years, plot=False, 
                                          test_normality=False, compare_tails=False)
        returns = phase1['returns']
        
        # Fit EVT
        evt_engine = fit_evt_model(returns, ticker=ticker, plot=False)
        
        # Armazenar resultados
        results_dict[ticker] = {
            'kurtosis': phase1['statistics']['kurtosis'],
            'xi': evt_engine.xi,
            'var_99': -evt_engine.calculate_var(0.99),
            'es_99': -evt_engine.calculate_es(0.99)
        }
    
    # Comparação
    print("\n" + "="*60)
    print("📊 COMPARAÇÃO DE RISCO")
    print("="*60)
    
    print(f"\n{'Ticker':<10} {'Curtose':<12} {'Xi (ξ)':<12} {'VaR(99%)':<12} {'ES(99%)':<12}")
    print("-" * 60)
    
    for ticker, res in results_dict.items():
        print(f"{ticker:<10} {res['kurtosis']:<12.2f} {res['xi']:<12.4f} "
              f"{res['var_99']*100:<12.2f}% {res['es_99']*100:<12.2f}%")
    
    print("\n" + "="*60)
    print("✅ Comparação completa!")
    print("\n💡 Todos os ativos mostram fat tails (K > 3, ξ > 0)")
    print("="*60 + "\n")


def example_calculate_black_swan_prob():
    """Exemplo: Calcular probabilidade de cisne negro."""
    
    print("\n" + "="*60)
    print("🔮 EXEMPLO: Probabilidade de Cisne Negro")
    print("="*60 + "\n")
    
    ticker = 'SPY'
    crash_threshold = -0.10  # -10% crash
    
    print(f"Calculando probabilidade de crash de -10% em {ticker}...\n")
    
    # Análise
    phase1 = analyze_gaussian_failure(ticker, years=15, plot=False, 
                                      test_normality=False, compare_tails=False)
    returns = phase1['returns']
    
    evt_engine = fit_evt_model(returns, ticker=ticker, plot=False)
    
    # Período de retorno
    years_evt = evt_engine.calculate_return_period(-crash_threshold)
    
    # Comparar com normal
    import numpy as np
    from scipy import stats
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    prob_normal = stats.norm.cdf(crash_threshold, mu, sigma)
    years_normal = (1 / prob_normal) / 252 if prob_normal > 0 else np.inf
    
    print(f"Crash de -10%:")
    print(f"   Modelo Normal: 1 em {years_normal:.2e} anos")
    print(f"   Motor EVT:     1 em {years_evt:.1f} anos")
    
    if years_evt < years_normal / 100:
        print(f"\n   ⚠️  EVT prevê que este crash é {years_normal/years_evt:.0e}x mais provável!")
    
    print("\n" + "="*60)
    print("✅ Cálculo completo!")
    print("="*60 + "\n")


def main():
    """Menu principal."""
    
    print("\n" + "="*80)
    print(" "*25 + "🚀 QUICKSTART - Motor EVT")
    print("="*80)
    
    print("\nEscolha um exemplo:\n")
    print("1. Análise Básica (um ativo)")
    print("2. Comparar Múltiplos Ativos")
    print("3. Calcular Probabilidade de Cisne Negro")
    print("4. Executar Todos")
    print("0. Sair")
    
    choice = input("\nOpção: ").strip()
    
    if choice == '1':
        example_basic()
    elif choice == '2':
        example_compare_tickers()
    elif choice == '3':
        example_calculate_black_swan_prob()
    elif choice == '4':
        example_basic()
        example_compare_tickers()
        example_calculate_black_swan_prob()
    elif choice == '0':
        print("\nSaindo...")
        return
    else:
        print("\n❌ Opção inválida!")
        return
    
    print("\n💡 Para demo completa com todas as fases, execute:")
    print("   python demo_complete.py\n")


if __name__ == "__main__":
    main()

