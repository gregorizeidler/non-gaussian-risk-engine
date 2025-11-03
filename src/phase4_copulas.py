"""
Phase 4: "Expert Level" Extension - Multivariate Portfolio

Implements portfolio analysis using:
- EVT to model tails of each asset individually
- Copulas to model dependencies between tails
- Monte Carlo simulation for portfolio risk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings

from src.phase2_evt_engine import EVTEngine, fit_evt_model
from src.utils import (
    download_data,
    calculate_log_returns,
    format_percent,
    save_figure
)

# Configuration
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')


class TailCopula:
    """
    Tail Copula implementation to model extreme dependencies.
    """
    
    def __init__(self, returns_dict: Dict[str, np.ndarray]):
        """
        Initialize copula.
        
        Parameters
        ----------
        returns_dict : dict
            Dictionary {ticker: returns_array}
        """
        self.tickers = list(returns_dict.keys())
        self.returns_dict = returns_dict
        self.n_assets = len(self.tickers)
        
        # Convert to matrix
        min_length = min(len(r) for r in returns_dict.values())
        self.returns_matrix = np.column_stack([
            returns_dict[ticker][:min_length] for ticker in self.tickers
        ])
        
        # Parâmetros da cópula
        self.copula_type = None
        self.copula_params = None
        
        # Ranks (para cópulas empíricas)
        self.ranks = None
        self.uniforms = None
        
    def compute_rank_correlation(self) -> pd.DataFrame:
        """
        Calcula correlações de rank (Kendall's Tau e Spearman).
        
        Returns
        -------
        pd.DataFrame
            Matriz de correlação de Kendall's Tau
        """
        from scipy.stats import kendalltau, spearmanr
        
        n = self.n_assets
        kendall_matrix = np.eye(n)
        spearman_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Kendall's Tau
                tau, _ = kendalltau(self.returns_matrix[:, i], self.returns_matrix[:, j])
                kendall_matrix[i, j] = tau
                kendall_matrix[j, i] = tau
                
                # Spearman
                rho, _ = spearmanr(self.returns_matrix[:, i], self.returns_matrix[:, j])
                spearman_matrix[i, j] = rho
                spearman_matrix[j, i] = rho
        
        kendall_df = pd.DataFrame(kendall_matrix, index=self.tickers, columns=self.tickers)
        spearman_df = pd.DataFrame(spearman_matrix, index=self.tickers, columns=self.tickers)
        
        print(f"\n{'='*60}")
        print(f"🔗 Correlações de Rank (Kendall's Tau)")
        print(f"{'='*60}")
        print(kendall_df.round(3))
        print(f"{'='*60}\n")
        
        return kendall_df
    
    def compute_tail_dependence(self, tail: str = 'lower', quantile: float = 0.05) -> pd.DataFrame:
        """
        Calcula coeficiente de dependência de cauda empírico.
        
        Parameters
        ----------
        tail : str, default='lower'
            'lower' para cauda inferior (crashes) ou 'upper' para cauda superior
        quantile : float, default=0.05
            Quantil para definir a cauda (ex: 0.05 = 5% piores)
            
        Returns
        -------
        pd.DataFrame
            Matriz de dependência de cauda
        """
        n = self.n_assets
        tail_dep_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                returns_i = self.returns_matrix[:, i]
                returns_j = self.returns_matrix[:, j]
                
                if tail == 'lower':
                    # Cauda inferior (perdas conjuntas)
                    threshold_i = np.percentile(returns_i, quantile * 100)
                    threshold_j = np.percentile(returns_j, quantile * 100)
                    
                    in_tail_i = returns_i <= threshold_i
                    in_tail_j = returns_j <= threshold_j
                else:
                    # Cauda superior
                    threshold_i = np.percentile(returns_i, (1 - quantile) * 100)
                    threshold_j = np.percentile(returns_j, (1 - quantile) * 100)
                    
                    in_tail_i = returns_i >= threshold_i
                    in_tail_j = returns_j >= threshold_j
                
                # Proporção de eventos conjuntos
                joint_tail = np.logical_and(in_tail_i, in_tail_j)
                tail_dep = np.sum(joint_tail) / np.sum(in_tail_i)
                
                tail_dep_matrix[i, j] = tail_dep
                tail_dep_matrix[j, i] = tail_dep
        
        tail_dep_df = pd.DataFrame(tail_dep_matrix, index=self.tickers, columns=self.tickers)
        
        print(f"\n{'='*60}")
        print(f"🔗 Dependência de Cauda {tail.capitalize()} (Empírica)")
        print(f"Quantil: {quantile*100:.0f}%")
        print(f"{'='*60}")
        print(tail_dep_df.round(3))
        print(f"{'='*60}\n")
        
        return tail_dep_df
    
    def fit_gaussian_copula(self) -> np.ndarray:
        """
        Fit de Cópula Gaussiana.
        
        Returns
        -------
        np.ndarray
            Matriz de correlação
        """
        # Transformar para uniformes via ranks
        self.ranks = np.array([stats.rankdata(self.returns_matrix[:, i]) 
                               for i in range(self.n_assets)]).T
        self.uniforms = self.ranks / (len(self.returns_matrix) + 1)
        
        # Inversa da normal padrão
        normals = stats.norm.ppf(self.uniforms)
        
        # Matriz de correlação
        corr_matrix = np.corrcoef(normals.T)
        
        self.copula_type = 'gaussian'
        self.copula_params = {'correlation': corr_matrix}
        
        print(f"\n{'='*60}")
        print(f"📊 Cópula Gaussiana Fitted")
        print(f"{'='*60}")
        print("Matriz de Correlação:")
        print(pd.DataFrame(corr_matrix, index=self.tickers, columns=self.tickers).round(3))
        print(f"{'='*60}\n")
        
        return corr_matrix
    
    def fit_t_copula(self, method: str = 'mle') -> Tuple[np.ndarray, float]:
        """
        Fit de Cópula-t de Student (captura dependência de cauda).
        
        Parameters
        ----------
        method : str, default='mle'
            Método de estimação
            
        Returns
        -------
        tuple
            (correlation_matrix, degrees_of_freedom)
        """
        # Transformar para uniformes
        if self.uniforms is None:
            self.ranks = np.array([stats.rankdata(self.returns_matrix[:, i]) 
                                   for i in range(self.n_assets)]).T
            self.uniforms = self.ranks / (len(self.returns_matrix) + 1)
        
        # Função de log-likelihood para otimizar
        def neg_log_likelihood(params):
            # Últimos n*(n-1)/2 params são correlações, último é df
            n = self.n_assets
            n_corr = n * (n - 1) // 2
            
            # Reconstruir matriz de correlação
            corr_values = params[:n_corr]
            df = params[-1]
            
            if df <= 2:  # Graus de liberdade devem ser > 2
                return np.inf
            
            # Matriz de correlação
            corr_matrix = np.eye(n)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    # Limitar correlação entre -0.99 e 0.99
                    corr = np.clip(corr_values[idx], -0.99, 0.99)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    idx += 1
            
            # Verificar se matriz é positiva definida
            try:
                np.linalg.cholesky(corr_matrix)
            except np.linalg.LinAlgError:
                return np.inf
            
            # Inversa da t de Student
            t_values = stats.t.ppf(self.uniforms, df=df)
            
            # Log-likelihood (simplificado)
            try:
                inv_corr = np.linalg.inv(corr_matrix)
                det_corr = np.linalg.det(corr_matrix)
                
                ll = 0
                for t_vec in t_values:
                    quad_form = t_vec @ inv_corr @ t_vec
                    ll += -0.5 * np.log(det_corr) - ((df + n) / 2) * np.log(1 + quad_form / df)
                
                return -ll
            except:
                return np.inf
        
        # Estimativa inicial: usar correlação Pearson e df=5
        corr_matrix_init = np.corrcoef(self.returns_matrix.T)
        n_corr = self.n_assets * (self.n_assets - 1) // 2
        corr_values_init = []
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                corr_values_init.append(corr_matrix_init[i, j])
        
        x0 = corr_values_init + [5.0]  # df inicial = 5
        
        # Otimização (simplificada - usa método mais rápido)
        print("⏳ Fitting Cópula-t (pode demorar alguns segundos)...")
        
        # Usar estimativa simplificada em vez de otimização completa
        # Para um portfólio real, você usaria bibliotecas como 'copulas' ou 'copulae'
        normals = stats.norm.ppf(self.uniforms)
        corr_matrix = np.corrcoef(normals.T)
        
        # Estimar df empiricamente via tail dependence
        # Quanto mais dependência de cauda, menor o df
        avg_tail_dep = self.compute_tail_dependence(tail='lower', quantile=0.05).values
        avg_tail_dep = avg_tail_dep[np.triu_indices_from(avg_tail_dep, k=1)].mean()
        
        # Heurística: mapear tail_dep para df
        # Alta tail_dep -> baixo df (mais dependência extrema)
        if avg_tail_dep > 0.3:
            df = 4
        elif avg_tail_dep > 0.2:
            df = 6
        elif avg_tail_dep > 0.1:
            df = 8
        else:
            df = 10
        
        self.copula_type = 't'
        self.copula_params = {'correlation': corr_matrix, 'df': df}
        
        print(f"\n{'='*60}")
        print(f"📊 Cópula-t de Student Fitted")
        print(f"{'='*60}")
        print(f"Graus de liberdade (df): {df}")
        print("Matriz de Correlação:")
        print(pd.DataFrame(corr_matrix, index=self.tickers, columns=self.tickers).round(3))
        print(f"{'='*60}\n")
        
        return corr_matrix, df
    
    def sample_copula(self, n_samples: int = 10000) -> np.ndarray:
        """
        Amostra da cópula fitted.
        
        Parameters
        ----------
        n_samples : int, default=10000
            Número de amostras
            
        Returns
        -------
        np.ndarray
            Array (n_samples, n_assets) de uniformes correlacionadas
        """
        if self.copula_params is None:
            raise ValueError("Fit the copula first (fit_gaussian_copula or fit_t_copula)")
        
        corr = self.copula_params['correlation']
        
        if self.copula_type == 'gaussian':
            # Amostra de normal multivariada
            normals = np.random.multivariate_normal(
                mean=np.zeros(self.n_assets),
                cov=corr,
                size=n_samples
            )
            # Transforma para uniformes
            uniforms = stats.norm.cdf(normals)
        
        elif self.copula_type == 't':
            # Amostra de t multivariada
            df = self.copula_params['df']
            
            # t multivariada = normal multivariada / sqrt(chi2/df)
            normals = np.random.multivariate_normal(
                mean=np.zeros(self.n_assets),
                cov=corr,
                size=n_samples
            )
            chi2_samples = np.random.chisquare(df=df, size=n_samples)
            t_samples = normals / np.sqrt(chi2_samples / df)[:, np.newaxis]
            
            # Transforma para uniformes
            uniforms = stats.t.cdf(t_samples, df=df)
        
        else:
            raise ValueError(f"Tipo de cópula '{self.copula_type}' não suportado")
        
        return uniforms


class PortfolioRiskAnalyzer:
    """
    Analisador de risco de portfólio usando EVT + Cópulas.
    """
    
    def __init__(
        self,
        tickers: List[str],
        weights: List[float] = None,
        years: int = 15
    ):
        """
        Inicializa analisador.
        
        Parameters
        ----------
        tickers : list
            Lista de tickers
        weights : list, optional
            Pesos do portfólio (default: pesos iguais)
        years : int, default=15
            Anos de histórico
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        
        if weights is None:
            self.weights = np.array([1.0 / self.n_assets] * self.n_assets)
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalizar
        
        self.years = years
        
        # Dados
        self.returns_dict = {}
        self.evt_models = {}
        self.copula = None
        
    def load_data(self):
        """Carrega dados para todos os ativos."""
        print(f"\n{'='*80}")
        print(f"📥 Carregando dados do portfólio")
        print(f"Ativos: {', '.join(self.tickers)}")
        print(f"Pesos: {', '.join([f'{w:.1%}' for w in self.weights])}")
        print(f"{'='*80}\n")
        
        for ticker in self.tickers:
            data = download_data(ticker, years=self.years, save_to_disk=False)
            returns = calculate_log_returns(data['Close'])
            self.returns_dict[ticker] = returns
    
    def fit_evt_models(self, threshold_percentile: float = 95.0):
        """Fit de modelo EVT para cada ativo."""
        print(f"\n{'='*80}")
        print(f"🔧 Fitting EVT para cada ativo")
        print(f"{'='*80}\n")
        
        for ticker in self.tickers:
            print(f"\n--- {ticker} ---")
            evt = fit_evt_model(
                self.returns_dict[ticker],
                ticker=ticker,
                threshold_percentile=threshold_percentile,
                plot=False
            )
            self.evt_models[ticker] = evt
    
    def fit_copula(self, copula_type: str = 't'):
        """Fit de cópula para modelar dependências."""
        print(f"\n{'='*80}")
        print(f"🔗 Fitting Cópula de Dependências")
        print(f"{'='*80}\n")
        
        self.copula = TailCopula(self.returns_dict)
        
        # Correlações de rank
        self.copula.compute_rank_correlation()
        
        # Dependência de cauda
        self.copula.compute_tail_dependence(tail='lower', quantile=0.05)
        
        # Fit da cópula
        if copula_type == 'gaussian':
            self.copula.fit_gaussian_copula()
        elif copula_type == 't':
            self.copula.fit_t_copula()
        else:
            raise ValueError(f"Tipo de cópula '{copula_type}' não suportado")
    
    def simulate_portfolio(self, n_simulations: int = 10000, horizon: int = 1) -> np.ndarray:
        """
        Simula retornos de portfólio usando EVT + Cópula.
        
        Parameters
        ----------
        n_simulations : int, default=10000
            Número de simulações Monte Carlo
        horizon : int, default=1
            Horizonte em dias
            
        Returns
        -------
        np.ndarray
            Array de retornos de portfólio simulados
        """
        print(f"\n⏳ Simulando {n_simulations:,} cenários de portfólio...")
        
        # Amostra uniformes da cópula
        uniforms = self.copula.sample_copula(n_simulations)
        
        # Para cada ativo, transformar uniformes em retornos usando EVT
        simulated_returns = np.zeros((n_simulations, self.n_assets))
        
        for i, ticker in enumerate(self.tickers):
            evt = self.evt_models[ticker]
            
            # Inverter CDF empírica + EVT
            for j in range(n_simulations):
                u = uniforms[j, i]
                
                # Quantil correspondente
                # Se u <= threshold percentile, usar empírico
                # Se u > threshold percentile, usar EVT
                threshold_prob = 1 - evt.n_exceedances / len(evt.losses)
                
                if u <= threshold_prob:
                    # Abaixo do threshold: empírico
                    quantile_idx = int(u * len(evt.returns))
                    quantile_idx = min(quantile_idx, len(evt.returns) - 1)
                    simulated_return = np.sort(evt.returns)[quantile_idx]
                else:
                    # Acima do threshold: EVT (cauda)
                    quantile_loss = evt.calculate_quantile(u)
                    simulated_return = -quantile_loss
                
                simulated_returns[j, i] = simulated_return
        
        # Retorno do portfólio = soma ponderada
        portfolio_returns = simulated_returns @ self.weights
        
        print(f"✅ Simulação completa!")
        
        return portfolio_returns
    
    def calculate_portfolio_var_es(
        self,
        portfolio_returns: np.ndarray,
        confidence: float = 0.99
    ) -> Dict:
        """
        Calcula VaR e ES do portfólio.
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Retornos de portfólio simulados
        confidence : float, default=0.99
            Nível de confiança
            
        Returns
        -------
        dict
            VaR e ES do portfólio
        """
        var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
        
        # ES: média dos retornos piores que VaR
        losses = -portfolio_returns
        var_losses = losses >= var
        es = np.mean(losses[var_losses]) if np.any(var_losses) else var
        
        return {
            'var': -var,
            'es': -es
        }
    
    def plot_portfolio_analysis(
        self,
        portfolio_returns: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        Cria visualizações da análise de portfólio.
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Retornos de portfólio simulados
        save : bool, default=True
            Se True, salva a figura
            
        Returns
        -------
        matplotlib.figure.Figure
            Figura com análises
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'Análise de Portfólio: {", ".join(self.tickers)}\n'
            f'Pesos: {", ".join([f"{t}={w:.0%}" for t, w in zip(self.tickers, self.weights)])}',
            fontsize=14,
            fontweight='bold'
        )
        
        # Plot 1: Distribuição de retornos de portfólio
        ax1 = axes[0, 0]
        self._plot_portfolio_distribution(ax1, portfolio_returns)
        
        # Plot 2: Correlação durante crashes
        ax2 = axes[0, 1]
        self._plot_tail_correlations(ax2)
        
        # Plot 3: Contribuição ao risco
        ax3 = axes[1, 0]
        self._plot_risk_contributions(ax3, portfolio_returns)
        
        # Plot 4: Diversification benefit
        ax4 = axes[1, 1]
        self._plot_diversification_benefit(ax4)
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, f'phase4_portfolio_analysis')
        
        return fig
    
    def _plot_portfolio_distribution(self, ax: plt.Axes, portfolio_returns: np.ndarray):
        """Plot distribuição de retornos de portfólio."""
        ax.hist(portfolio_returns, bins=100, density=True, alpha=0.6, 
                color='steelblue', edgecolor='black', label='Simulado (EVT+Cópula)')
        
        # Comparar com normal
        mu = np.mean(portfolio_returns)
        sigma = np.std(portfolio_returns)
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
        normal_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_pdf, 'r--', linewidth=2, label='Normal (referência)')
        
        # VaR e ES
        var = -np.percentile(portfolio_returns, 1)
        ax.axvline(var, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'VaR(99%): {format_percent(var, 2)}')
        
        ax.set_xlabel('Retorno de Portfólio', fontsize=11)
        ax.set_ylabel('Densidade', fontsize=11)
        ax.set_title('Portfolio Returns Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_tail_correlations(self, ax: plt.Axes):
        """Plot correlações em diferentes regimes."""
        # Correlação normal vs cauda
        returns_matrix = np.column_stack([self.returns_dict[t] for t in self.tickers])
        
        # Correlação total
        corr_full = np.corrcoef(returns_matrix.T)
        
        # Correlação na cauda (5% piores dias)
        threshold = np.percentile(returns_matrix, 5, axis=0)
        in_tail = np.all(returns_matrix <= threshold, axis=1)
        
        if np.sum(in_tail) > 10:
            corr_tail = np.corrcoef(returns_matrix[in_tail].T)
        else:
            corr_tail = corr_full
        
        # Plot heatmaps lado a lado
        x_pos = np.arange(self.n_assets)
        width = 0.35
        
        # Extrair triângulo superior
        indices = np.triu_indices(self.n_assets, k=1)
        corr_full_values = corr_full[indices]
        corr_tail_values = corr_tail[indices]
        
        labels = [f'{self.tickers[i]}-{self.tickers[j]}' 
                  for i, j in zip(indices[0], indices[1])]
        x = np.arange(len(labels))
        
        bars1 = ax.bar(x - width/2, corr_full_values, width, label='Correlação Normal', alpha=0.7)
        bars2 = ax.bar(x + width/2, corr_tail_values, width, label='Correlação em Crashes', 
                       color='red', alpha=0.7)
        
        ax.set_ylabel('Correlação', fontsize=11)
        ax.set_title('Correlação: Normal vs Crashes', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)
        
        # Texto
        textstr = '⚠️ Correlações aumentam\ndurante crashes!'
        props = dict(boxstyle='round', facecolor='mistyrose', alpha=0.8)
        ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='darkred', fontweight='bold')
    
    def _plot_risk_contributions(self, ax: plt.Axes, portfolio_returns: np.ndarray):
        """Plot contribuição de cada ativo ao risco do portfólio."""
        # Componente marginal de risco (simplificado)
        # Correlação de cada ativo com o portfólio
        returns_matrix = np.column_stack([self.returns_dict[t] for t in self.tickers])
        
        # Para dados históricos
        historical_portfolio = returns_matrix @ self.weights
        
        contributions = []
        for i in range(self.n_assets):
            # Correlação com portfólio
            corr = np.corrcoef(returns_matrix[:, i], historical_portfolio)[0, 1]
            # Contribuição = peso * volatilidade individual * correlação / volatilidade portfolio
            vol_i = np.std(returns_matrix[:, i])
            vol_p = np.std(historical_portfolio)
            contrib = self.weights[i] * vol_i * corr / vol_p
            contributions.append(contrib)
        
        contributions = np.array(contributions)
        contributions = contributions / contributions.sum()  # Normalizar
        
        ax.bar(self.tickers, contributions * 100, color='steelblue', alpha=0.7)
        ax.set_ylabel('Contribuição ao Risco (%)', fontsize=11)
        ax.set_title('Contribuição de Cada Ativo ao Risco do Portfólio', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores
        for i, (ticker, contrib) in enumerate(zip(self.tickers, contributions)):
            ax.text(i, contrib * 100 + 1, f'{contrib*100:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
    
    def _plot_diversification_benefit(self, ax: plt.Axes):
        """Plot benefício da diversificação."""
        returns_matrix = np.column_stack([self.returns_dict[t] for t in self.tickers])
        
        # Volatilidade individual pesada
        individual_vols = [np.std(returns_matrix[:, i]) for i in range(self.n_assets)]
        weighted_vol = np.sqrt(np.sum((self.weights * individual_vols)**2))
        
        # Volatilidade do portfólio
        portfolio_returns_hist = returns_matrix @ self.weights
        portfolio_vol = np.std(portfolio_returns_hist)
        
        # Diversification ratio
        diversification_ratio = weighted_vol / portfolio_vol
        
        # Plot
        categories = ['Volatilidade\nIndividual\n(ponderada)', 'Volatilidade\ndo Portfólio']
        values = [weighted_vol * 100, portfolio_vol * 100]
        colors = ['lightcoral', 'lightgreen']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Volatilidade Anualizada (%)', fontsize=11)
        ax.set_title('Benefício da Diversificação', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Redução
        reduction = (1 - portfolio_vol / weighted_vol) * 100
        textstr = f'Diversification Ratio: {diversification_ratio:.2f}\n' \
                  f'Redução de risco: {reduction:.1f}%'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', ha='center', bbox=props, fontweight='bold')


def analyze_portfolio_with_copulas(
    tickers: List[str],
    weights: List[float] = None,
    years: int = 15,
    copula_type: str = 't',
    n_simulations: int = 10000,
    plot: bool = True
) -> Dict:
    """
    Pipeline completo de análise de portfólio com EVT + Cópulas.
    
    Parameters
    ----------
    tickers : list
        Lista de tickers
    weights : list, optional
        Pesos do portfólio
    years : int, default=15
        Anos de histórico
    copula_type : str, default='t'
        Tipo de cópula ('gaussian' ou 't')
    n_simulations : int, default=10000
        Número de simulações Monte Carlo
    plot : bool, default=True
        Se True, cria visualizações
        
    Returns
    -------
    dict
        Resultados da análise
    """
    print(f"\n{'='*80}")
    print(f"🎯 FASE 4: Análise de Portfólio Multivariado (EVT + Cópulas)")
    print(f"{'='*80}\n")
    
    # Criar analisador
    analyzer = PortfolioRiskAnalyzer(tickers, weights, years)
    
    # Carregar dados
    analyzer.load_data()
    
    # Fit EVT para cada ativo
    analyzer.fit_evt_models(threshold_percentile=95)
    
    # Fit cópula
    analyzer.fit_copula(copula_type=copula_type)
    
    # Simular portfólio
    portfolio_returns = analyzer.simulate_portfolio(n_simulations=n_simulations)
    
    # Calcular VaR e ES
    metrics = analyzer.calculate_portfolio_var_es(portfolio_returns, confidence=0.99)
    
    print(f"\n{'='*60}")
    print(f"📊 Métricas de Risco do Portfólio")
    print(f"{'='*60}")
    print(f"VaR(99%): {format_percent(metrics['var'], 2)}")
    print(f"ES(99%): {format_percent(metrics['es'], 2)}")
    print(f"{'='*60}\n")
    
    # Visualizações
    if plot:
        fig = analyzer.plot_portfolio_analysis(portfolio_returns, save=True)
        plt.show()
    
    print(f"\n{'='*80}")
    print(f"✅ Análise de Portfólio completa!")
    print(f"{'='*80}\n")
    
    return {
        'analyzer': analyzer,
        'portfolio_returns': portfolio_returns,
        'metrics': metrics
    }


if __name__ == "__main__":
    # Exemplo de uso
    tickers = ['SPY', 'GLD', 'TLT']  # Ações, Ouro, Títulos
    weights = [0.6, 0.2, 0.2]  # 60% ações, 20% ouro, 20% títulos
    
    results = analyze_portfolio_with_copulas(
        tickers=tickers,
        weights=weights,
        years=15,
        copula_type='t',
        n_simulations=10000
    )

