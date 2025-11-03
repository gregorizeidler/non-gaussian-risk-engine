"""
Phase 2: The EVT Engine (The Econophysics Solution)

Implements Extreme Value Theory (EVT) using the Peaks-Over-Threshold (POT) 
method and the Generalized Pareto Distribution (GPD).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional
import warnings

from src.utils import (
    format_percent,
    save_figure,
    get_losses
)

# Style configuration
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')


class EVTEngine:
    """
    Extreme Value Analysis Engine using Peaks-Over-Threshold (POT) method.
    """
    
    def __init__(self, returns: np.ndarray, ticker: str = ""):
        """
        Initialize the EVT engine.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of returns (can be positive or negative)
        ticker : str, optional
            Ticker name for identification
        """
        self.ticker = ticker
        self.returns = returns
        self.losses = -returns  # Convert to losses (positive)
        
        self.threshold = None
        self.exceedances = None
        self.xi = None  # Shape parameter
        self.sigma = None  # Scale parameter
        self.sigma = None  # Parâmetro de escala (scale)
        self.n_exceedances = None
        
    def select_threshold(
        self,
        method: str = 'percentile',
        percentile: float = 95.0,
        threshold_value: float = None
    ) -> float:
        """
        Seleciona o limiar (threshold) para o método POT.
        
        Parameters
        ----------
        method : str, default='percentile'
            Método de seleção ('percentile' ou 'value')
        percentile : float, default=95.0
            Percentil a usar (ex: 95 = 5% piores perdas)
        threshold_value : float, optional
            Valor específico de threshold (se method='value')
            
        Returns
        -------
        float
            Valor do threshold selecionado
        """
        if method == 'percentile':
            self.threshold = np.percentile(self.losses, percentile)
        elif method == 'value':
            if threshold_value is None:
                raise ValueError("threshold_value deve ser especificado quando method='value'")
            self.threshold = threshold_value
        else:
            raise ValueError(f"Method '{method}' not recognized")
        
        # Calculate exceedances
        self.exceedances = self.losses[self.losses > self.threshold] - self.threshold
        self.n_exceedances = len(self.exceedances)
        
        print(f"\n{'='*60}")
        print(f"🎯 Threshold Selection")
        print(f"{'='*60}")
        print(f"Method: {method}")
        if method == 'percentile':
            print(f"Percentile: {percentile}%")
        print(f"Threshold: {format_percent(self.threshold)}")
        print(f"Exceedances: {self.n_exceedances} ({self.n_exceedances/len(self.losses)*100:.1f}% of data)")
        print(f"Mean exceedance: {format_percent(np.mean(self.exceedances))}")
        print(f"Max exceedance: {format_percent(np.max(self.exceedances))}")
        print(f"{'='*60}\n")
        
        return self.threshold
    
    def fit_gpd(self, method: str = 'mle') -> Tuple[float, float]:
        """
        Fit Generalized Pareto Distribution (GPD) to tail data.
        
        The GPD has the form:
        F(x) = 1 - (1 + ξ * x / σ)^(-1/ξ)  if ξ ≠ 0
        F(x) = 1 - exp(-x / σ)              if ξ = 0
        
        Parameters
        ----------
        method : str, default='mle'
            Método de estimação ('mle' para Maximum Likelihood)
            
        Returns
        -------
        tuple
            (xi, sigma) - parâmetros de forma e escala
        """
        if self.exceedances is None:
            raise ValueError("Execute select_threshold() primeiro")
        
        if len(self.exceedances) < 10:
            raise ValueError(f"Poucas excedências ({len(self.exceedances)}). Reduza o threshold.")
        
        if method == 'mle':
            # Maximum Likelihood Estimation
            self.xi, _, self.sigma = self._fit_gpd_mle()
        else:
            raise ValueError(f"Método '{method}' não implementado")
        
        print(f"\n{'='*60}")
        print(f"📊 Generalized Pareto Distribution (GPD) Fit")
        print(f"{'='*60}")
        print(f"Method: Maximum Likelihood Estimation (MLE)")
        print(f"\nEstimated parameters:")
        print(f"  ξ (Xi - Shape): {self.xi:.4f}")
        print(f"  σ (Sigma - Scale): {self.sigma:.6f} ({format_percent(self.sigma)})")
        print(f"\nInterpretation of ξ:")
        
        if self.xi > 0.5:
            print(f"  ⚠️  ξ > 0.5: VERY FAT TAIL!")
            print(f"      Infinite variance. Extreme crash risk.")
        elif self.xi > 0:
            print(f"  ⚠️  ξ > 0: FAT TAIL (Fat Tail)")
            print(f"      Extreme events are more likely than the normal model predicts.")
        elif self.xi == 0:
            print(f"  ✓  ξ = 0: Exponential tail (moderate risk)")
        else:
            print(f"  ✓  ξ < 0: Finite tail (rare in finance)")
        
        print(f"{'='*60}\n")
        
        return self.xi, self.sigma
    
    def _fit_gpd_mle(self) -> Tuple[float, float, float]:
        """
        Fit GPD usando Maximum Likelihood Estimation.
        
        Returns
        -------
        tuple
            (shape, loc, scale) compatível com scipy.stats.genpareto
        """
        # Log-likelihood negativa para GPD
        def neg_log_likelihood(params):
            xi, sigma = params
            
            # Restrições: sigma > 0
            if sigma <= 0:
                return np.inf
            
            n = len(self.exceedances)
            
            # Caso ξ ≈ 0 (distribuição exponencial)
            if np.abs(xi) < 1e-10:
                ll = -n * np.log(sigma) - np.sum(self.exceedances) / sigma
            else:
                # Verificar se os dados são válidos para os parâmetros
                z = 1 + xi * self.exceedances / sigma
                if np.any(z <= 0):
                    return np.inf
                
                ll = -n * np.log(sigma) - (1 + 1/xi) * np.sum(np.log(z))
            
            return -ll
        
        # Estimativa inicial
        mean_exc = np.mean(self.exceedances)
        var_exc = np.var(self.exceedances)
        
        # Método dos momentos como inicial
        xi_init = 0.5 * (1 - (mean_exc**2 / var_exc))
        sigma_init = 0.5 * mean_exc * (1 + xi_init)
        
        # Otimização
        result = minimize(
            neg_log_likelihood,
            x0=[xi_init, sigma_init],
            method='Nelder-Mead',
            options={'maxiter': 10000}
        )
        
        if not result.success:
            warnings.warn("Otimização MLE pode não ter convergido completamente")
        
        xi, sigma = result.x
        
        # Retorna no formato scipy (shape, loc=0, scale)
        return xi, 0, sigma
    
    def calculate_quantile(self, probability: float) -> float:
        """
        Calcula o quantil da distribuição de cauda usando GPD.
        
        Parameters
        ----------
        probability : float
            Probabilidade (ex: 0.99 para 99º percentil)
            
        Returns
        -------
        float
            Valor do quantil (como perda positiva)
        """
        if self.xi is None or self.sigma is None:
            raise ValueError("Execute fit_gpd() primeiro")
        
        # Probabilidade de exceder o threshold
        n_total = len(self.losses)
        p_exceed = self.n_exceedances / n_total
        
        # Ajuste da probabilidade para a cauda
        if probability <= (1 - p_exceed):
            # Usa dados empíricos abaixo do threshold
            quantile = np.percentile(self.losses, probability * 100)
        else:
            # Usa GPD acima do threshold
            p_cond = (probability - (1 - p_exceed)) / p_exceed
            
            if np.abs(self.xi) < 1e-10:
                # Caso exponencial
                exceedance = -self.sigma * np.log(1 - p_cond)
            else:
                # Caso geral GPD
                exceedance = (self.sigma / self.xi) * ((1 - p_cond)**(-self.xi) - 1)
            
            quantile = self.threshold + exceedance
        
        return quantile
    
    def calculate_var(self, confidence: float = 0.99) -> float:
        """
        Calcula Value-at-Risk (VaR) usando EVT.
        
        Parameters
        ----------
        confidence : float, default=0.99
            Nível de confiança (ex: 0.99 para VaR de 99%)
            
        Returns
        -------
        float
            VaR como perda positiva (converta para negativo para retorno)
        """
        var = self.calculate_quantile(confidence)
        return var
    
    def calculate_es(self, confidence: float = 0.99) -> float:
        """
        Calcula Expected Shortfall (ES) / CVaR usando EVT.
        
        ES é a perda média condicionada a exceder o VaR.
        
        Parameters
        ----------
        confidence : float, default=0.99
            Nível de confiança
            
        Returns
        -------
        float
            Expected Shortfall como perda positiva
        """
        var = self.calculate_var(confidence)
        
        if np.abs(self.xi) < 1e-10:
            # Caso exponencial
            es = var + self.sigma
        elif self.xi < 1:
            # Caso geral (ξ < 1 garante que ES existe)
            es = (var + self.sigma - self.xi * self.threshold) / (1 - self.xi)
        else:
            # ξ >= 1: ES não existe (variância infinita)
            warnings.warn(f"ξ = {self.xi:.4f} >= 1: ES pode não existir (variância infinita)")
            es = np.inf
        
        return es
    
    def calculate_return_period(self, loss_threshold: float) -> float:
        """
        Calcula o período de retorno (em anos) para uma perda específica.
        
        Parameters
        ----------
        loss_threshold : float
            Limiar de perda (ex: 0.10 para perda de 10%)
            
        Returns
        -------
        float
            Período de retorno em anos
        """
        if loss_threshold <= self.threshold:
            # Abaixo do threshold: usa distribuição empírica
            prob = np.mean(self.losses >= loss_threshold)
        else:
            # Acima do threshold: usa GPD
            exceedance = loss_threshold - self.threshold
            
            if np.abs(self.xi) < 1e-10:
                prob_exceed_given_threshold = np.exp(-exceedance / self.sigma)
            else:
                prob_exceed_given_threshold = (1 + self.xi * exceedance / self.sigma)**(-1/self.xi)
            
            # Probabilidade total
            p_threshold = self.n_exceedances / len(self.losses)
            prob = p_threshold * prob_exceed_given_threshold
        
        if prob <= 0:
            return np.inf
        
        # Converte para período em anos (252 trading days)
        days = 1 / prob
        years = days / 252
        
        return years
    
    def plot_diagnostics(self, save: bool = True) -> plt.Figure:
        """
        Cria plots de diagnóstico do fit GPD.
        
        Parameters
        ----------
        save : bool, default=True
            Se True, salva a figura
            
        Returns
        -------
        matplotlib.figure.Figure
            Figura com os diagnósticos
        """
        if self.xi is None:
            raise ValueError("Execute fit_gpd() primeiro")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f'EVT Diagnostics - GPD Fit: {self.ticker}\n'
            f'ξ = {self.xi:.4f}, σ = {self.sigma:.6f}, Threshold = {format_percent(self.threshold)}',
            fontsize=14,
            fontweight='bold'
        )
        
        # Plot 1: Density Plot
        ax1 = axes[0, 0]
        self._plot_density(ax1)
        
        # Plot 2: Q-Q Plot
        ax2 = axes[0, 1]
        self._plot_qq_gpd(ax2)
        
        # Plot 3: Mean Excess Plot
        ax3 = axes[1, 0]
        self._plot_mean_excess(ax3)
        
        # Plot 4: Return Level Plot
        ax4 = axes[1, 1]
        self._plot_return_level(ax4)
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, f'phase2_evt_diagnostics_{self.ticker}')
        
        return fig
    
    def _plot_density(self, ax: plt.Axes):
        """Plot densidade empírica vs GPD fitted."""
        # Histograma das excedências
        ax.hist(self.exceedances, bins=30, density=True, alpha=0.6, 
                color='steelblue', edgecolor='black', label='Empirical Data')
        
        # GPD fitted
        x = np.linspace(0, self.exceedances.max(), 1000)
        pdf = stats.genpareto.pdf(x, c=self.xi, loc=0, scale=self.sigma)
        ax.plot(x, pdf, 'r-', linewidth=2, label='GPD Fitted')
        
        ax.set_xlabel('Exceedance over Threshold', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Density: Empirical vs GPD', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_qq_gpd(self, ax: plt.Axes):
        """Q-Q plot para GPD."""
        # Quantis teóricos
        n = len(self.exceedances)
        sorted_data = np.sort(self.exceedances)
        empirical_probs = np.arange(1, n + 1) / (n + 1)
        theoretical_quantiles = stats.genpareto.ppf(empirical_probs, c=self.xi, loc=0, scale=self.sigma)
        
        # Plot
        ax.plot(theoretical_quantiles, sorted_data, 'o', markersize=4, alpha=0.6)
        
        # Linha de referência
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2)
        
        ax.set_xlabel('Theoretical Quantiles (GPD)', fontsize=10)
        ax.set_ylabel('Empirical Quantiles', fontsize=10)
        ax.set_title('Q-Q Plot: GPD Fit', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_mean_excess(self, ax: plt.Axes):
        """Mean Excess Plot para validar threshold."""
        # Para vários thresholds, calcular média das excedências
        thresholds = np.percentile(self.losses, np.linspace(80, 99, 50))
        mean_excesses = []
        
        for u in thresholds:
            excesses = self.losses[self.losses > u] - u
            if len(excesses) > 10:
                mean_excesses.append(np.mean(excesses))
            else:
                mean_excesses.append(np.nan)
        
        ax.plot(thresholds, mean_excesses, 'o-', markersize=4)
        ax.axvline(self.threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Selected Threshold')
        
        ax.set_xlabel('Threshold', fontsize=10)
        ax.set_ylabel('Mean Excess', fontsize=10)
        ax.set_title('Mean Excess Plot', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Texto explicativo
        textstr = 'If GPD is appropriate,\nthe plot should be ~linear'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    def _plot_return_level(self, ax: plt.Axes):
        """Return Level Plot."""
        # Períodos de retorno
        return_periods = np.array([1, 2, 5, 10, 20, 50, 100, 250, 500])  # em anos
        return_levels = []
        
        for years in return_periods:
            days = years * 252
            prob = 1 - 1/days
            quantile = self.calculate_quantile(prob)
            return_levels.append(quantile)
        
        # Plot
        ax.semilogx(return_periods, return_levels, 'o-', linewidth=2, markersize=8)
        
        ax.set_xlabel('Return Period (years)', fontsize=10)
        ax.set_ylabel('Return Level (Loss)', fontsize=10)
        ax.set_title('Return Level Plot', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        # Formatar eixo y como percentual
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: format_percent(y, 1)))
        
        # Adicionar alguns labels
        for i, (period, level) in enumerate(zip(return_periods, return_levels)):
            if i % 2 == 0:  # Mostrar apenas alguns labels
                ax.annotate(f'{format_percent(level, 1)}', 
                           xy=(period, level), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)


def fit_evt_model(
    returns: np.ndarray,
    ticker: str = "",
    threshold_percentile: float = 95.0,
    plot: bool = True
) -> EVTEngine:
    """
    Pipeline completo para fit do modelo EVT.
    
    Parameters
    ----------
    returns : np.ndarray
        Array de retornos
    ticker : str, optional
        Nome do ticker
    threshold_percentile : float, default=95.0
        Percentil para seleção do threshold
    plot : bool, default=True
        Se True, cria plots de diagnóstico
        
    Returns
    -------
    EVTEngine
        Instância do motor EVT com modelo fitted
    """
    print(f"\n{'='*80}")
    print(f"🔧 FASE 2: Motor EVT (Extreme Value Theory)")
    print(f"Ticker: {ticker}")
    print(f"{'='*80}\n")
    
    # Inicializa engine
    engine = EVTEngine(returns, ticker)
    
    # Seleciona threshold
    engine.select_threshold(method='percentile', percentile=threshold_percentile)
    
    # Fit GPD
    engine.fit_gpd(method='mle')
    
    # Diagnósticos
    if plot:
        fig = engine.plot_diagnostics(save=True)
        plt.show()
    
    print(f"\n{'='*80}")
    print(f"✅ Modelo EVT fitted com sucesso!")
    print(f"{'='*80}\n")
    
    return engine


if __name__ == "__main__":
    # Exemplo de uso
    from src.phase1_gaussian_failure import analyze_gaussian_failure
    
    # Primeiro, obter retornos
    results = analyze_gaussian_failure('SPY', years=15, plot=False)
    returns = results['returns']
    
    # Fit EVT
    evt_engine = fit_evt_model(returns, ticker='SPY', threshold_percentile=95)

