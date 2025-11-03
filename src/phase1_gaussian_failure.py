"""
Phase 1: The "Failure" of the Classical Model (Proof of the Problem)

This module visually and mathematically demonstrates that the normal distribution
fails to model financial returns, especially extreme events.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings

from src.utils import (
    download_data,
    calculate_log_returns,
    print_statistics,
    format_percent,
    save_figure
)

# Style configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
warnings.filterwarnings('ignore')


class GaussianFailureAnalyzer:
    """
    Analyzes and demonstrates the failure of the Gaussian model on financial data.
    """
    
    def __init__(self, ticker: str, years: int = 15):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        ticker : str
            Asset symbol (e.g. 'SPY', 'AAPL')
        years : int, default=15
            Number of years of history
        """
        self.ticker = ticker
        self.years = years
        self.data = None
        self.returns = None
        self.stats = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load historical data from Yahoo Finance.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with OHLCV data
        """
        self.data = download_data(self.ticker, years=self.years)
        return self.data
    
    def calculate_returns(self) -> np.ndarray:
        """
        Calculate logarithmic returns.
        
        Returns
        -------
        np.ndarray
            Array of logarithmic returns
        """
        if self.data is None:
            self.load_data()
        
        # Handle yfinance multi-index
        if isinstance(self.data.columns, pd.MultiIndex):
            prices = self.data[('Close', self.ticker)]
        else:
            prices = self.data['Close']
        
        self.returns = calculate_log_returns(prices)
        
        print(f"\n📈 Returns calculated: {len(self.returns)} observations")
        
        return self.returns
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate descriptive statistics of returns.
        
        Returns
        -------
        dict
            Dicionário com estatísticas
        """
        if self.returns is None:
            self.calculate_returns()
        
        self.stats = print_statistics(self.returns, self.ticker)
        
        return self.stats
    
    def test_normality(self) -> Dict:
        """
        Executa testes estatísticos de normalidade.
        
        Returns
        -------
        dict
            Resultados dos testes de normalidade
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Teste de Jarque-Bera (baseado em skewness e kurtosis)
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)
        
        # Teste de Kolmogorov-Smirnov
        ks_stat, ks_pvalue = stats.kstest(
            self.returns,
            'norm',
            args=(np.mean(self.returns), np.std(self.returns))
        )
        
        # Teste de Shapiro-Wilk (mais poderoso, mas limitado a n < 5000)
        if len(self.returns) <= 5000:
            sw_stat, sw_pvalue = stats.shapiro(self.returns)
        else:
            sw_stat, sw_pvalue = None, None
        
        results = {
            'jarque_bera': {'statistic': jb_stat, 'pvalue': jb_pvalue},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'pvalue': ks_pvalue},
            'shapiro_wilk': {'statistic': sw_stat, 'pvalue': sw_pvalue}
        }
        
        print(f"\n{'='*60}")
        print(f"🔬 Testes de Normalidade: {self.ticker}")
        print(f"{'='*60}")
        print(f"Jarque-Bera:")
        print(f"  Estatística: {jb_stat:.2f}")
        print(f"  P-valor: {jb_pvalue:.2e}")
        print(f"  Resultado: {'❌ NÃO NORMAL' if jb_pvalue < 0.05 else '✅ Normal'}")
        
        print(f"\nKolmogorov-Smirnov:")
        print(f"  Estatística: {ks_stat:.4f}")
        print(f"  P-valor: {ks_pvalue:.2e}")
        print(f"  Resultado: {'❌ NÃO NORMAL' if ks_pvalue < 0.05 else '✅ Normal'}")
        
        if sw_stat is not None:
            print(f"\nShapiro-Wilk:")
            print(f"  Estatística: {sw_stat:.4f}")
            print(f"  P-valor: {sw_pvalue:.2e}")
            print(f"  Resultado: {'❌ NÃO NORMAL' if sw_pvalue < 0.05 else '✅ Normal'}")
        
        print(f"{'='*60}\n")
        
        return results
    
    def calculate_tail_probability(self, threshold: float, model: str = 'normal') -> float:
        """
        Calcula a probabilidade de um evento extremo sob um modelo.
        
        Parameters
        ----------
        threshold : float
            Limiar de perda (ex: -0.10 para -10%)
        model : str, default='normal'
            Modelo a usar ('normal' ou 'empirical')
            
        Returns
        -------
        float
            Probabilidade do evento
        """
        if self.returns is None:
            self.calculate_returns()
        
        if model == 'normal':
            # Modelo Normal
            mu = np.mean(self.returns)
            sigma = np.std(self.returns)
            prob = stats.norm.cdf(threshold, loc=mu, scale=sigma)
        else:
            # Modelo Empírico
            prob = np.mean(self.returns <= threshold)
        
        return prob
    
    def compare_tail_probabilities(self, thresholds: list = None) -> pd.DataFrame:
        """
        Compara probabilidades de cauda entre modelo normal e empírico.
        
        Parameters
        ----------
        thresholds : list, optional
            Lista de thresholds de perda (ex: [-0.05, -0.07, -0.10])
            
        Returns
        -------
        pd.DataFrame
            DataFrame comparativo
        """
        if thresholds is None:
            thresholds = [-0.03, -0.05, -0.07, -0.10, -0.15]
        
        results = []
        
        for thresh in thresholds:
            prob_normal = self.calculate_tail_probability(thresh, model='normal')
            prob_empirical = self.calculate_tail_probability(thresh, model='empirical')
            
            # Converte probabilidades em "1 em N" eventos
            if prob_normal > 0:
                days_normal = 1 / prob_normal
                years_normal = days_normal / 252  # Trading days
            else:
                days_normal = np.inf
                years_normal = np.inf
            
            if prob_empirical > 0:
                days_empirical = 1 / prob_empirical
                years_empirical = days_empirical / 252
            else:
                days_empirical = np.inf
                years_empirical = np.inf
            
            results.append({
                'Perda': format_percent(thresh),
                'Prob Normal': f'{prob_normal:.2e}',
                'Prob Empírica': f'{prob_empirical:.2e}',
                'Normal (anos)': f'1 em {years_normal:.0f}' if years_normal < 1e6 else '> 1M',
                'Empírico (anos)': f'1 em {years_empirical:.1f}' if years_empirical < 1e3 else '> 1K'
            })
        
        df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print(f"⚖️  Comparação: Probabilidades de Cauda (Modelo Normal vs Realidade)")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        return df
    
    def plot_failure(self, save: bool = True) -> plt.Figure:
        """
        Cria visualização completa da falha do modelo gaussiano.
        
        Parameters
        ----------
        save : bool, default=True
            Se True, salva a figura em disco
            
        Returns
        -------
        matplotlib.figure.Figure
            Figura com os plots
        """
        if self.returns is None:
            self.calculate_returns()
        
        if self.stats is None:
            self.calculate_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'Gaussian Model Failure: {self.ticker}\n'
            f'Data: {self.years} years ({len(self.returns)} observations)',
            fontsize=16,
            fontweight='bold'
        )
        
        # Plot 1: Histograma vs Normal
        ax1 = axes[0, 0]
        self._plot_histogram_vs_normal(ax1)
        
        # Plot 2: Q-Q Plot
        ax2 = axes[0, 1]
        self._plot_qq(ax2)
        
        # Plot 3: Caudas (Log Scale)
        ax3 = axes[1, 0]
        self._plot_tail_comparison(ax3)
        
        # Plot 4: Série Temporal com Eventos Extremos
        ax4 = axes[1, 1]
        self._plot_extreme_events(ax4)
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, f'phase1_gaussian_failure_{self.ticker}')
        
        return fig
    
    def _plot_histogram_vs_normal(self, ax: plt.Axes):
        """Plot histograma dos retornos vs distribuição normal."""
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        
        # Histograma
        ax.hist(self.returns, bins=100, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black', label='Real Data')
        
        # Curva Normal
        x = np.linspace(self.returns.min(), self.returns.max(), 1000)
        normal_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal Model')
        
        # Destacar eventos extremos
        extreme_threshold = mu - 3 * sigma
        ax.axvline(extreme_threshold, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'3σ ({format_percent(extreme_threshold)})')
        
        ax.set_xlabel('Daily Return', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Histogram vs Normal Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Adicionar texto com curtose
        kurtosis = self.stats['kurtosis']
        textstr = f'Kurtosis = {kurtosis:.2f}\n(Normal = 3.0)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    def _plot_qq(self, ax: plt.Axes):
        """Plot Q-Q (Quantile-Quantile) para comparar com normal."""
        stats.probplot(self.returns, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot: Returns vs Normal', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Adicionar texto explicativo
        textstr = 'If the data were normal,\npoints would be on the red line'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    def _plot_tail_comparison(self, ax: plt.Axes):
        """Plot comparação de caudas em escala log."""
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        
        # Cauda esquerda (perdas)
        sorted_returns = np.sort(self.returns)
        empirical_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        
        # Probabilidades teóricas da normal
        theoretical_prob = stats.norm.cdf(sorted_returns, mu, sigma)
        
        # Plot apenas a cauda esquerda (20% menores)
        n_tail = int(0.2 * len(sorted_returns))
        
        ax.semilogy(sorted_returns[:n_tail], empirical_prob[:n_tail], 
                    'o', markersize=4, alpha=0.6, label='Empirical')
        ax.semilogy(sorted_returns[:n_tail], theoretical_prob[:n_tail], 
                    'r-', linewidth=2, label='Normal')
        
        ax.set_xlabel('Daily Return', fontsize=11)
        ax.set_ylabel('Cumulative Probability (log scale)', fontsize=11)
        ax.set_title('Lower Tail Comparison (20% worst days)', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Destacar discrepância
        textstr = '⚠️ Normal Model\nunderestimates tail risk'
        props = dict(boxstyle='round', facecolor='mistyrose', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='darkred', fontweight='bold')
    
    def _plot_extreme_events(self, ax: plt.Axes):
        """Plot série temporal com eventos extremos destacados."""
        if self.data is None:
            return
        
        # Criar série temporal de retornos
        returns_series = pd.Series(self.returns, index=self.data.index[1:])
        
        # Identificar eventos extremos (> 3 desvios padrão)
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        threshold_3sigma = mu - 3 * sigma
        threshold_4sigma = mu - 4 * sigma
        
        extreme_3sigma = returns_series[returns_series < threshold_3sigma]
        extreme_4sigma = returns_series[returns_series < threshold_4sigma]
        
        # Plot série temporal
        ax.plot(returns_series.index, returns_series.values, 
                linewidth=0.5, alpha=0.7, color='gray', label='Daily Returns')
        
        # Destacar eventos extremos
        ax.scatter(extreme_3sigma.index, extreme_3sigma.values, 
                   color='orange', s=50, alpha=0.8, label=f'> 3σ ({len(extreme_3sigma)} events)', zorder=5)
        ax.scatter(extreme_4sigma.index, extreme_4sigma.values, 
                   color='red', s=100, alpha=0.9, label=f'> 4σ ({len(extreme_4sigma)} events)', zorder=6)
        
        # Linhas de threshold
        ax.axhline(threshold_3sigma, color='orange', linestyle='--', 
                   linewidth=1, alpha=0.7)
        ax.axhline(threshold_4sigma, color='red', linestyle='--', 
                   linewidth=1, alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Daily Return', fontsize=11)
        ax.set_title('Extreme Events Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        # Formatar eixo y como percentual
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: format_percent(y, 1)))
        
        # Rotacionar labels do eixo x
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def analyze_gaussian_failure(
    ticker: str,
    years: int = 15,
    plot: bool = True,
    test_normality: bool = True,
    compare_tails: bool = True
) -> Dict:
    """
    Pipeline completo de análise da falha gaussiana.
    
    Parameters
    ----------
    ticker : str
        Símbolo do ativo
    years : int, default=15
        Anos de histórico
    plot : bool, default=True
        Se True, cria visualizações
    test_normality : bool, default=True
        Se True, executa testes de normalidade
    compare_tails : bool, default=True
        Se True, compara probabilidades de cauda
        
    Returns
    -------
    dict
        Resultados da análise
    """
    print(f"\n{'='*80}")
    print(f"🔍 FASE 1: Análise da Falha do Modelo Gaussiano")
    print(f"Ticker: {ticker} | Período: {years} anos")
    print(f"{'='*80}\n")
    
    analyzer = GaussianFailureAnalyzer(ticker, years)
    
    # Carregar dados e calcular retornos
    analyzer.load_data()
    analyzer.calculate_returns()
    
    # Estatísticas descritivas
    stats = analyzer.calculate_statistics()
    
    results = {
        'ticker': ticker,
        'returns': analyzer.returns,
        'statistics': stats
    }
    
    # Testes de normalidade
    if test_normality:
        normality_tests = analyzer.test_normality()
        results['normality_tests'] = normality_tests
    
    # Comparação de probabilidades de cauda
    if compare_tails:
        tail_comparison = analyzer.compare_tail_probabilities()
        results['tail_comparison'] = tail_comparison
    
    # Visualizações
    if plot:
        fig = analyzer.plot_failure(save=True)
        results['figure'] = fig
        plt.show()
    
    print(f"\n{'='*80}")
    print(f"✅ Análise Fase 1 completa!")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    results = analyze_gaussian_failure('SPY', years=15)

