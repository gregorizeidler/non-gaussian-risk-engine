"""
Complete Demo Script for the Non-Gaussian Risk Engine

This script runs all 4 phases of the project in sequence,
demonstrating complete risk analysis using EVT.

Usage:
    python demo_complete.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Import project modules
from src.phase1_gaussian_failure import analyze_gaussian_failure
from src.phase2_evt_engine import fit_evt_model
from src.phase3_risk_metrics import calculate_risk_metrics
from src.phase4_copulas import analyze_portfolio_with_copulas


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def demo_phase_1(ticker: str = 'SPY', years: int = 15):
    """
    Demonstrate Phase 1: Gaussian Model Failure.
    
    Parameters
    ----------
    ticker : str
        Asset ticker
    years : int
        Years of history
        
    Returns
    -------
    dict
        Analysis results
    """
    print_header("PHASE 1: THE GAUSSIAN MODEL FAILURE")
    
    print("📖 Objective:")
    print("   Prove visually and mathematically that the normal distribution")
    print("   fails to model financial returns.\n")
    
    input("Press ENTER to continue...")
    
    # Run analysis
    results = analyze_gaussian_failure(
        ticker=ticker,
        years=years,
        plot=True,
        test_normality=True,
        compare_tails=True
    )
    
    # Extrair insights
    kurtosis = results['statistics']['kurtosis']
    
    print("\n✅ FASE 1 COMPLETA!")
    print(f"\n💡 Insight Principal:")
    print(f"   Curtose = {kurtosis:.2f} (Normal = 3.0)")
    
    if kurtosis > 3:
        print(f"   ⚠️  CAUDAS GORDAS detectadas!")
        print(f"   O modelo normal SUBESTIMA o risco de crashes.")
    
    input("\n\nPressione ENTER para ir para Fase 2...")
    
    return results


def demo_phase_2(returns: np.ndarray, ticker: str = 'SPY'):
    """
    Demonstra Fase 2: Motor EVT.
    
    Parameters
    ----------
    returns : np.ndarray
        Array de retornos
    ticker : str
        Ticker do ativo
        
    Returns
    -------
    EVTEngine
        Motor EVT fitted
    """
    print_header("FASE 2: O MOTOR EVT (EXTREME VALUE THEORY)")
    
    print("📖 Objetivo:")
    print("   Implementar a solução correta: modelar apenas os EXTREMOS")
    print("   usando a Distribuição Generalizada de Pareto (GPD).\n")
    
    input("Pressione ENTER para continuar...")
    
    # Fit EVT
    evt_engine = fit_evt_model(
        returns=returns,
        ticker=ticker,
        threshold_percentile=95.0,
        plot=True
    )
    
    print("\n✅ FASE 2 COMPLETA!")
    print(f"\n💡 Insight Principal:")
    print(f"   ξ (Xi) = {evt_engine.xi:.4f}")
    
    if evt_engine.xi > 0:
        print(f"   ⚠️  Parâmetro Xi > 0 confirma FAT TAILS!")
        print(f"   Eventos extremos são mais prováveis que o modelo normal prevê.")
    
    input("\n\nPressione ENTER para ir para Fase 3...")
    
    return evt_engine


def demo_phase_3(returns: np.ndarray, evt_engine, ticker: str = 'SPY'):
    """
    Demonstra Fase 3: Métricas de Risco Práticas.
    
    Parameters
    ----------
    returns : np.ndarray
        Array de retornos
    evt_engine : EVTEngine
        Motor EVT fitted
    ticker : str
        Ticker do ativo
        
    Returns
    -------
    dict
        Resultados das métricas
    """
    print_header("FASE 3: MÉTRICAS DE RISCO PRÁTICAS")
    
    print("📖 Objetivo:")
    print("   Calcular VaR, Expected Shortfall e probabilidades de 'Cisnes Negros'")
    print("   comparando Modelo Normal vs Motor EVT.\n")
    
    input("Pressione ENTER para continuar...")
    
    # Calcular métricas
    results = calculate_risk_metrics(
        returns=returns,
        evt_engine=evt_engine,
        ticker=ticker,
        confidence=0.99,
        plot=True
    )
    
    print("\n✅ FASE 3 COMPLETA!")
    print(f"\n💡 Insight Principal:")
    print(f"   O Motor EVT prevê riscos significativamente MAIORES")
    print(f"   que o modelo normal para eventos extremos.")
    print(f"   Isso tem impacto direto em alocação de capital e stop-loss.")
    
    input("\n\nPressione ENTER para ir para Fase 4...")
    
    return results


def demo_phase_4(tickers: list = None, weights: list = None, years: int = 15):
    """
    Demonstra Fase 4: Análise de Portfólio Multivariado.
    
    Parameters
    ----------
    tickers : list, optional
        Lista de tickers do portfólio
    weights : list, optional
        Pesos do portfólio
    years : int
        Anos de histórico
        
    Returns
    -------
    dict
        Resultados da análise de portfólio
    """
    print_header("FASE 4: PORTFÓLIO MULTIVARIADO (EVT + CÓPULAS)")
    
    print("📖 Objetivo:")
    print("   Modelar risco de portfólio usando EVT para cada ativo")
    print("   e Cópulas para capturar dependências durante crashes.\n")
    
    if tickers is None:
        tickers = ['SPY', 'GLD', 'TLT']  # Ações, Ouro, Títulos
        weights = [0.6, 0.2, 0.2]  # 60-20-20
    
    print(f"Portfólio:")
    for ticker, weight in zip(tickers, weights):
        print(f"   {ticker}: {weight*100:.0f}%")
    
    input("\nPressione ENTER para continuar...")
    
    # Análise de portfólio
    results = analyze_portfolio_with_copulas(
        tickers=tickers,
        weights=weights,
        years=years,
        copula_type='t',  # Cópula-t captura dependência de cauda
        n_simulations=10000,
        plot=True
    )
    
    print("\n✅ FASE 4 COMPLETA!")
    print(f"\n💡 Insight Principal:")
    print(f"   Durante crashes, as correlações AUMENTAM (correlação vai para 1).")
    print(f"   Diversificação falha exatamente quando você mais precisa.")
    print(f"   Cópulas modelam corretamente este fenômeno.")
    
    return results


def main():
    """
    Executa demonstração completa de todas as 4 fases.
    """
    print("\n" + "="*80)
    print(" "*20 + "🚀 MOTOR DE RISCO NÃO-GAUSSIANO")
    print(" "*15 + "The Non-Gaussian Risk Engine")
    print(" "*10 + "(Extreme Value Theory + Copulas)")
    print("="*80)
    
    print("\n📚 Este script demonstra todas as 4 fases do projeto:")
    print("\n   Fase 1: Prova da Falha do Modelo Gaussiano")
    print("   Fase 2: Implementação do Motor EVT")
    print("   Fase 3: Métricas de Risco Práticas (VaR, ES, Cisnes Negros)")
    print("   Fase 4: Portfólio Multivariado com Cópulas")
    
    print("\n⏱️  Tempo estimado: 5-10 minutos")
    print("📊 Gráficos serão exibidos para cada fase")
    
    proceed = input("\n\nDeseja continuar? (s/n): ").strip().lower()
    
    if proceed not in ['s', 'sim', 'y', 'yes']:
        print("\nDemo cancelada pelo usuário.")
        return
    
    # Configuração
    TICKER = 'SPY'
    YEARS = 15
    
    try:
        # FASE 1
        phase1_results = demo_phase_1(ticker=TICKER, years=YEARS)
        returns = phase1_results['returns']
        
        # FASE 2
        evt_engine = demo_phase_2(returns=returns, ticker=TICKER)
        
        # FASE 3
        phase3_results = demo_phase_3(
            returns=returns,
            evt_engine=evt_engine,
            ticker=TICKER
        )
        
        # FASE 4
        phase4_results = demo_phase_4(
            tickers=['SPY', 'GLD', 'TLT'],
            weights=[0.6, 0.2, 0.2],
            years=YEARS
        )
        
        # CONCLUSÃO
        print_header("✅ DEMONSTRAÇÃO COMPLETA!")
        
        print("🎓 O que você aprendeu:\n")
        print("1. A distribuição normal FALHA em modelar retornos financeiros")
        print("2. EVT é a solução matemática correta para eventos extremos")
        print("3. VaR e ES calculados com EVT são muito mais realistas")
        print("4. Cópulas modelam corretamente dependências durante crashes")
        print("5. Este framework é aplicável a qualquer ativo ou portfólio")
        
        print("\n📁 Resultados salvos em: results/")
        print("📊 Todas as figuras foram salvas para referência futura")
        
        print("\n🚀 Próximos passos:")
        print("   • Explore os módulos em src/")
        print("   • Teste com seus próprios ativos")
        print("   • Adapte para seu caso de uso específico")
        print("   • Integre em sistemas de gestão de risco")
        
        print("\n📚 Para mais informações, consulte o README.md")
        
        print("\n" + "="*80)
        print(" "*25 + "Obrigado por usar o Motor EVT!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Erro durante execução: {e}")
        print("Por favor, verifique se todas as dependências estão instaladas:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()

