"""
Phase 3: Practical Investment Applications (The "So What?")

Implements risk metrics using EVT:
- True Value-at-Risk (VaR)
- Expected Shortfall (ES) / CVaR
- "Black Swan" event probabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
import warnings

from src.phase2_evt_engine import EVTEngine
from src.utils import format_percent, save_figure

# Configuration
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')


class RiskMetricsCalculator:
    """
    Risk metrics calculator comparing Normal model vs EVT.
    """
    
    def __init__(self, returns: np.ndarray, evt_engine: EVTEngine, ticker: str = ""):
        """
        Initialize calculator.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns
        evt_engine : EVTEngine
            Fitted EVT engine
        ticker : str, optional
            Ticker name
        """
        self.returns = returns
        self.evt = evt_engine
        self.ticker = ticker
        
        # Basic statistics
        self.mu = np.mean(returns)
        self.sigma = np.std(returns)
        
    def calculate_var(
        self,
        confidence: float = 0.99,
        method: str = 'both'
    ) -> Dict[str, float]:
        """
        Calcula Value-at-Risk (VaR).
        
        Parameters
        ----------
        confidence : float, default=0.99
            Nível de confiança (ex: 0.99 para 99%)
        method : str, default='both'
            Método: 'normal', 'evt', 'empirical', ou 'both'
            
        Returns
        -------
        dict
            VaRs calculados por diferentes métodos
        """
        results = {}
        
        if method in ['normal', 'both']:
            # VaR Normal (Gaussiano)
            z_score = stats.norm.ppf(1 - confidence)
            var_normal = -(self.mu + z_score * self.sigma)  # Negativo = perda
            results['normal'] = var_normal
        
        if method in ['evt', 'both']:
            # VaR EVT
            var_evt_loss = self.evt.calculate_var(confidence)
            var_evt = -var_evt_loss  # Negativo = perda
            results['evt'] = var_evt
        
        if method in ['empirical', 'both']:
            # VaR Empírico (histórico)
            var_empirical = -np.percentile(-self.returns, confidence * 100)
            results['empirical'] = var_empirical
        
        return results
    
    def calculate_es(
        self,
        confidence: float = 0.99,
        method: str = 'both'
    ) -> Dict[str, float]:
        """
        Calcula Expected Shortfall (ES) / Conditional VaR (CVaR).
        
        Parameters
        ----------
        confidence : float, default=0.99
            Nível de confiança
        method : str, default='both'
            Método: 'normal', 'evt', 'empirical', ou 'both'
            
        Returns
        -------
        dict
            ES calculados por diferentes métodos
        """
        results = {}
        
        if method in ['normal', 'both']:
            # ES Normal
            z_score = stats.norm.ppf(1 - confidence)
            pdf_z = stats.norm.pdf(z_score)
            es_normal = -(self.mu + self.sigma * pdf_z / (1 - confidence))
            results['normal'] = es_normal
        
        if method in ['evt', 'both']:
            # ES EVT
            es_evt_loss = self.evt.calculate_es(confidence)
            es_evt = -es_evt_loss if es_evt_loss != np.inf else -np.inf
            results['evt'] = es_evt
        
        if method in ['empirical', 'both']:
            # ES Empírico
            var_emp = self.calculate_var(confidence, method='empirical')['empirical']
            losses_beyond_var = -self.returns[self.returns <= var_emp]
            es_empirical = -np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var_emp
            results['empirical'] = es_empirical
        
        return results
    
    def calculate_black_swan_probability(
        self,
        crash_threshold: float = -0.10,
        method: str = 'both'
    ) -> Dict[str, Dict]:
        """
        Calcula a probabilidade de um evento "Cisne Negro" (crash extremo).
        
        Parameters
        ----------
        crash_threshold : float, default=-0.10
            Threshold do crash (ex: -0.10 = perda de 10%)
        method : str, default='both'
            Método: 'normal', 'evt', ou 'both'
            
        Returns
        -------
        dict
            Probabilidades e períodos de retorno para cada método
        """
        results = {}
        loss_threshold = -crash_threshold
        
        if method in ['normal', 'both']:
            # Modelo Normal
            z_score = (crash_threshold - self.mu) / self.sigma
            prob_normal = stats.norm.cdf(z_score)
            
            if prob_normal > 0:
                days_normal = 1 / prob_normal
                years_normal = days_normal / 252
            else:
                days_normal = np.inf
                years_normal = np.inf
            
            results['normal'] = {
                'probability': prob_normal,
                'days': days_normal,
                'years': years_normal
            }
        
        if method in ['evt', 'both']:
            # Modelo EVT
            years_evt = self.evt.calculate_return_period(loss_threshold)
            
            if years_evt != np.inf:
                days_evt = years_evt * 252
                prob_evt = 1 / days_evt
            else:
                days_evt = np.inf
                prob_evt = 0
            
            results['evt'] = {
                'probability': prob_evt,
                'days': days_evt,
                'years': years_evt
            }
        
        # Empírico
        prob_empirical = np.mean(self.returns <= crash_threshold)
        if prob_empirical > 0:
            days_empirical = 1 / prob_empirical
            years_empirical = days_empirical / 252
        else:
            days_empirical = np.inf
            years_empirical = np.inf
        
        results['empirical'] = {
            'probability': prob_empirical,
            'days': days_empirical,
            'years': years_empirical
        }
        
        return results
    
    def compare_all_metrics(
        self,
        confidence_levels: List[float] = None,
        crash_thresholds: List[float] = None
    ) -> Dict:
        """
        Compara todas as métricas de risco entre os métodos.
        
        Parameters
        ----------
        confidence_levels : list, optional
            Níveis de confiança para VaR/ES (default: [0.95, 0.99])
        crash_thresholds : list, optional
            Thresholds de crash (default: [-0.05, -0.07, -0.10])
            
        Returns
        -------
        dict
            Comparação completa das métricas
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        if crash_thresholds is None:
            crash_thresholds = [-0.05, -0.07, -0.10]
        
        results = {
            'var': {},
            'es': {},
            'black_swan': {}
        }
        
        # VaR e ES para cada nível de confiança
        for conf in confidence_levels:
            results['var'][conf] = self.calculate_var(conf, method='both')
            results['es'][conf] = self.calculate_es(conf, method='both')
        
        # Probabilidades de Cisne Negro
        for threshold in crash_thresholds:
            results['black_swan'][threshold] = self.calculate_black_swan_probability(
                threshold, method='both'
            )
        
        return results
    
    def print_comparison(self, confidence: float = 0.99):
        """
        Imprime comparação formatada entre Normal e EVT.
        
        Parameters
        ----------
        confidence : float, default=0.99
            Nível de confiança
        """
        # VaR
        var_results = self.calculate_var(confidence)
        
        # ES
        es_results = self.calculate_es(confidence)
        
        # Cisnes Negros
        crashes = [-0.05, -0.07, -0.10]
        bs_results = {}
        for crash in crashes:
            bs_results[crash] = self.calculate_black_swan_probability(crash)
        
        # Imprimir
        print(f"\n{'='*80}")
        print(f"📊 COMPARAÇÃO: Modelo Normal vs Motor EVT")
        print(f"Ticker: {self.ticker} | Confiança: {confidence*100:.0f}%")
        print(f"{'='*80}\n")
        
        # Tabela VaR
        print(f"1️⃣  Value-at-Risk (VaR) - {confidence*100:.0f}%")
        print(f"   {'Method':<15} {'VaR':<12} {'Diff vs Normal'}")
        print(f"   {'-'*50}")
        var_normal = var_results['normal']
        print(f"   {'Normal':<15} {format_percent(var_normal, 2):<12} {'-'}")
        print(f"   {'EVT':<15} {format_percent(var_results['evt'], 2):<12} "
              f"{format_percent(var_results['evt'] - var_normal, 2)} "
              f"({(var_results['evt']/var_normal - 1)*100:+.1f}%)")
        print(f"   {'Empírico':<15} {format_percent(var_results['empirical'], 2):<12} "
              f"{format_percent(var_results['empirical'] - var_normal, 2)}")
        
        # Tabela ES
        print(f"\n2️⃣  Expected Shortfall (ES) - {confidence*100:.0f}%")
        print(f"   {'Method':<15} {'ES':<12} {'Diff vs Normal'}")
        print(f"   {'-'*50}")
        es_normal = es_results['normal']
        print(f"   {'Normal':<15} {format_percent(es_normal, 2):<12} {'-'}")
        es_evt_val = es_results['evt']
        if es_evt_val != -np.inf:
            print(f"   {'EVT':<15} {format_percent(es_evt_val, 2):<12} "
                  f"{format_percent(es_evt_val - es_normal, 2)} "
                  f"({(es_evt_val/es_normal - 1)*100:+.1f}%)")
        else:
            print(f"   {'EVT':<15} {'∞ (ξ≥1)':<12}")
        print(f"   {'Empírico':<15} {format_percent(es_results['empirical'], 2):<12} "
              f"{format_percent(es_results['empirical'] - es_normal, 2)}")
        
        # Tabela Cisne Negro
        print(f"\n3️⃣  Probabilidade de Eventos 'Cisne Negro'")
        print(f"   {'Crash':<10} {'Normal':<25} {'EVT':<25} {'Empírico':<25}")
        print(f"   {'-'*85}")
        
        for crash in crashes:
            bs = bs_results[crash]
            
            normal_str = f"1 em {bs['normal']['years']:.0f} anos" if bs['normal']['years'] < 1e6 else "> 1M anos"
            evt_str = f"1 em {bs['evt']['years']:.1f} anos" if bs['evt']['years'] < 1e3 else "> 1K anos"
            emp_str = f"1 em {bs['empirical']['years']:.1f} anos" if bs['empirical']['years'] < 1e3 else "Nunca observado"
            
            print(f"   {format_percent(crash, 1):<10} {normal_str:<25} {evt_str:<25} {emp_str:<25}")
        
        print(f"\n{'='*80}")
        
        # Mensagem de alerta
        if var_results['evt'] < 1.3 * var_results['normal']:
            print(f"⚠️  ALERTA: EVT prevê risco {(var_results['evt']/var_results['normal']-1)*100:.0f}% maior que Normal!")
        
        print(f"{'='*80}\n")
    
    def plot_comparison(self, save: bool = True) -> plt.Figure:
        """
        Cria visualizações comparativas entre Normal e EVT.
        
        Parameters
        ----------
        save : bool, default=True
            Se True, salva a figura
            
        Returns
        -------
        matplotlib.figure.Figure
            Figura com as comparações
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'Comparison: Normal Model vs EVT Engine - {self.ticker}',
            fontsize=16,
            fontweight='bold'
        )
        
        # Plot 1: Distribuições com VaR e ES
        ax1 = axes[0, 0]
        self._plot_distributions_with_risk_metrics(ax1)
        
        # Plot 2: VaR Comparison Across Confidence Levels
        ax2 = axes[0, 1]
        self._plot_var_comparison(ax2)
        
        # Plot 3: Black Swan Probabilities
        ax3 = axes[1, 0]
        self._plot_black_swan_probabilities(ax3)
        
        # Plot 4: Capital Allocation Impact
        ax4 = axes[1, 1]
        self._plot_capital_impact(ax4)
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, f'phase3_risk_comparison_{self.ticker}')
        
        return fig
    
    def _plot_distributions_with_risk_metrics(self, ax: plt.Axes):
        """Plot distribuições com VaR e ES marcados."""
        # Histograma
        ax.hist(self.returns, bins=100, density=True, alpha=0.5, 
                color='gray', edgecolor='black', label='Real Data')
        
        # Curva Normal
        x = np.linspace(self.returns.min(), self.returns.max(), 1000)
        normal_pdf = stats.norm.pdf(x, self.mu, self.sigma)
        ax.plot(x, normal_pdf, 'b-', linewidth=2, label='Normal Model')
        
        # VaR e ES (99%)
        var_normal = self.calculate_var(0.99)['normal']
        var_evt = self.calculate_var(0.99)['evt']
        es_normal = self.calculate_es(0.99)['normal']
        es_evt = self.calculate_es(0.99)['evt']
        
        # Linhas verticais
        ax.axvline(var_normal, color='blue', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'VaR Normal: {format_percent(var_normal, 2)}')
        ax.axvline(var_evt, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'VaR EVT: {format_percent(var_evt, 2)}')
        
        if es_evt != -np.inf:
            ax.axvline(es_normal, color='blue', linestyle=':', linewidth=2, alpha=0.7)
            ax.axvline(es_evt, color='red', linestyle=':', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Daily Return', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Distributions with VaR (99%) and ES', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=self.returns.min())
    
    def _plot_var_comparison(self, ax: plt.Axes):
        """Plot VaR comparison across confidence levels."""
        confidence_levels = np.linspace(0.90, 0.99, 20)
        var_normal = []
        var_evt = []
        
        for conf in confidence_levels:
            vars = self.calculate_var(conf)
            var_normal.append(-vars['normal'])  # Como perda positiva
            var_evt.append(-vars['evt'])
        
        ax.plot(confidence_levels * 100, var_normal, 'b-', linewidth=2, 
                marker='o', markersize=4, label='Normal Model')
        ax.plot(confidence_levels * 100, var_evt, 'r-', linewidth=2, 
                marker='s', markersize=4, label='EVT Engine')
        
        # Área de diferença
        ax.fill_between(confidence_levels * 100, var_normal, var_evt, 
                        alpha=0.2, color='red', label='Normal Underestimation')
        
        ax.set_xlabel('Confidence Level (%)', fontsize=11)
        ax.set_ylabel('VaR (Loss)', fontsize=11)
        ax.set_title('VaR: Normal vs EVT', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: format_percent(y, 1)))
    
    def _plot_black_swan_probabilities(self, ax: plt.Axes):
        """Plot probabilidades de cisne negro."""
        crashes = np.array([-0.03, -0.05, -0.07, -0.10, -0.15, -0.20])
        years_normal = []
        years_evt = []
        
        for crash in crashes:
            bs = self.calculate_black_swan_probability(crash)
            years_normal.append(bs['normal']['years'])
            years_evt.append(bs['evt']['years'])
        
        x = np.arange(len(crashes))
        width = 0.35
        
        # Limitar valores para visualização
        years_normal_plot = [min(y, 1e6) for y in years_normal]
        years_evt_plot = [min(y, 1e3) for y in years_evt]
        
        bars1 = ax.bar(x - width/2, years_normal_plot, width, label='Normal Model', 
                       color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, years_evt_plot, width, label='EVT Engine', 
                       color='red', alpha=0.7)
        
        ax.set_xlabel('Crash Magnitude', fontsize=11)
        ax.set_ylabel('Return Period (years, log scale)', fontsize=11)
        ax.set_title('Probability of "Black Swans"', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([format_percent(c, 0) for c in crashes])
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        
        # Adicionar texto
        textstr = '⚠️ Normal Model\ndrastically underestimates!'
        props = dict(boxstyle='round', facecolor='mistyrose', alpha=0.8)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='darkred', fontweight='bold')
    
    def _plot_capital_impact(self, ax: plt.Axes):
        """Plot impacto no capital necessário."""
        # Suponha um portfólio de $1M
        portfolio_value = 1_000_000
        confidence_levels = [0.95, 0.99, 0.995]
        
        capital_normal = []
        capital_evt = []
        labels = []
        
        for conf in confidence_levels:
            vars = self.calculate_var(conf)
            capital_normal.append(portfolio_value * (-vars['normal']))
            capital_evt.append(portfolio_value * (-vars['evt']))
            labels.append(f"{conf*100:.1f}%")
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, capital_normal, width, label='Normal Model', 
                       color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, capital_evt, width, label='EVT Engine', 
                       color='red', alpha=0.7)
        
        # Adicionar valores nos bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'${height/1000:.0f}K',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Confidence Level', fontsize=11)
        ax.set_ylabel('Capital at Risk ($)', fontsize=11)
        ax.set_title(f'Capital Impact (Portfolio of ${portfolio_value/1e6:.1f}M)', 
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Formatar eixo y
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y/1000:.0f}K'))


def calculate_risk_metrics(
    returns: np.ndarray = None,
    evt_engine: EVTEngine = None,
    ticker: str = "",
    confidence: float = 0.99,
    plot: bool = True
) -> Dict:
    """
    Pipeline completo para cálculo de métricas de risco.
    
    Parameters
    ----------
    returns : np.ndarray, optional
        Array de retornos
    evt_engine : EVTEngine, optional
        Motor EVT fitted (se None, será criado)
    ticker : str, optional
        Nome do ticker
    confidence : float, default=0.99
        Nível de confiança principal
    plot : bool, default=True
        Se True, cria visualizações
        
    Returns
    -------
    dict
        Métricas de risco calculadas
    """
    print(f"\n{'='*80}")
    print(f"📈 FASE 3: Cálculo de Métricas de Risco")
    print(f"Ticker: {ticker}")
    print(f"{'='*80}\n")
    
    # Se EVT engine não fornecido, criar um
    if evt_engine is None:
        from src.phase2_evt_engine import fit_evt_model
        evt_engine = fit_evt_model(returns, ticker=ticker, plot=False)
    
    # Criar calculadora
    calculator = RiskMetricsCalculator(returns, evt_engine, ticker)
    
    # Calcular e imprimir comparação
    calculator.print_comparison(confidence)
    
    # Comparação completa
    all_metrics = calculator.compare_all_metrics()
    
    # Plots
    if plot:
        fig = calculator.plot_comparison(save=True)
        plt.show()
    
    print(f"\n{'='*80}")
    print(f"✅ Métricas de risco calculadas com sucesso!")
    print(f"{'='*80}\n")
    
    return {
        'calculator': calculator,
        'metrics': all_metrics
    }


if __name__ == "__main__":
    # Exemplo de uso
    from src.phase1_gaussian_failure import analyze_gaussian_failure
    from src.phase2_evt_engine import fit_evt_model
    
    # Carregar dados
    results = analyze_gaussian_failure('SPY', years=15, plot=False)
    returns = results['returns']
    
    # Fit EVT
    evt_engine = fit_evt_model(returns, ticker='SPY', plot=False)
    
    # Calcular métricas
    risk_results = calculate_risk_metrics(returns, evt_engine, ticker='SPY')

