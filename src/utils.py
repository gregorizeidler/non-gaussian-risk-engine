"""
Funções auxiliares para o Motor de Risco Não-Gaussiano
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Union, List, Tuple
import os


def download_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    years: int = 15,
    save_to_disk: bool = True
) -> pd.DataFrame:
    """
    Baixa dados históricos do Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Símbolo do ticker (ex: 'SPY', 'AAPL')
    start_date : str, optional
        Data inicial no formato 'YYYY-MM-DD'
    end_date : str, optional
        Data final no formato 'YYYY-MM-DD'
    years : int, default=15
        Número de anos de histórico (usado se start_date não especificado)
    save_to_disk : bool, default=True
        Se True, salva os dados em CSV
        
    Returns
    -------
    pd.DataFrame
        DataFrame com dados OHLCV
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    
    print(f"📥 Downloading {ticker} data ({start_date} to {end_date})...")
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"Unable to download data for {ticker}")
    
    print(f"✅ {len(df)} days of data downloaded")
    
    if save_to_disk:
        os.makedirs('data', exist_ok=True)
        filepath = f'data/{ticker}_{start_date}_{end_date}.csv'
        df.to_csv(filepath)
        print(f"💾 Data saved to: {filepath}")
    
    return df


def calculate_log_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Calculate logarithmic returns: log(P_t / P_{t-1})
    
    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Time series of prices
        
    Returns
    -------
    np.ndarray
        Array of log returns (excluding first NaN)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Remove possible NaN or Inf
    log_returns = log_returns[np.isfinite(log_returns)]
    
    return log_returns


def calculate_simple_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
    
    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Time series of prices
        
    Returns
    -------
    np.ndarray
        Array of simple returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    simple_returns = (prices[1:] - prices[:-1]) / prices[:-1]
    simple_returns = simple_returns[np.isfinite(simple_returns)]
    
    return simple_returns


def get_losses(returns: np.ndarray) -> np.ndarray:
    """
    Converte retornos em perdas (valores negativos tornam-se positivos).
    Útil para análise de cauda inferior (downside risk).
    
    Parameters
    ----------
    returns : np.ndarray
        Array de retornos
        
    Returns
    -------
    np.ndarray
        Array de perdas (valores positivos)
    """
    return -returns


def calculate_drawdown(prices: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, float, int]:
    """
    Calcula o drawdown de uma série de preços.
    
    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Série temporal de preços
        
    Returns
    -------
    tuple
        (drawdown_series, max_drawdown, max_drawdown_duration)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calcula o máximo acumulado (peak)
    cummax = np.maximum.accumulate(prices)
    
    # Drawdown em cada ponto
    drawdown = (prices - cummax) / cummax
    
    # Máximo drawdown
    max_dd = np.min(drawdown)
    
    # Duração do drawdown (simplificado)
    dd_duration = 0
    current_duration = 0
    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            dd_duration = max(dd_duration, current_duration)
        else:
            current_duration = 0
    
    return drawdown, max_dd, dd_duration


def format_percent(value: float, decimals: int = 2) -> str:
    """
    Formata um valor decimal como percentual.
    
    Parameters
    ----------
    value : float
        Valor decimal (ex: 0.05 para 5%)
    decimals : int, default=2
        Número de casas decimais
        
    Returns
    -------
    str
        String formatada (ex: '5.00%')
    """
    # Converter para float se for pandas Series/array
    if hasattr(value, 'item'):
        value = value.item()
    elif hasattr(value, '__iter__') and not isinstance(value, str):
        value = float(list(value)[0]) if hasattr(value, '__iter__') else float(value)
    return f"{float(value) * 100:.{decimals}f}%"


def print_statistics(returns: np.ndarray, ticker: str = "") -> dict:
    """
    Print descriptive statistics of returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    ticker : str, optional
        Ticker name for title
        
    Returns
    -------
    dict
        Dictionary with calculated statistics
    """
    from scipy import stats
    
    # Ensure returns is 1D numpy array
    returns = np.asarray(returns).flatten()
    
    stats_dict = {
        'count': int(len(returns)),
        'mean': float(np.mean(returns)),
        'std': float(np.std(returns)),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'skewness': float(stats.skew(returns)),
        'kurtosis': float(stats.kurtosis(returns, fisher=False)),  # Pearson (normal = 3)
        'excess_kurtosis': float(stats.kurtosis(returns, fisher=True)),  # Excess (normal = 0)
        'percentile_1': float(np.percentile(returns, 1)),
        'percentile_5': float(np.percentile(returns, 5)),
        'percentile_95': float(np.percentile(returns, 95)),
        'percentile_99': float(np.percentile(returns, 99)),
    }
    
    if ticker:
        print(f"\n{'='*60}")
        print(f"📊 Descriptive Statistics: {ticker}")
        print(f"{'='*60}")
    else:
        print(f"\n📊 Descriptive Statistics")
        print(f"{'='*60}")
    
    print(f"Observations: {stats_dict['count']}")
    print(f"Mean: {format_percent(stats_dict['mean'])}")
    print(f"Std Deviation: {format_percent(stats_dict['std'])}")
    print(f"Minimum: {format_percent(stats_dict['min'])}")
    print(f"Maximum: {format_percent(stats_dict['max'])}")
    print(f"\nSkewness: {stats_dict['skewness']:.4f}")
    print(f"Kurtosis: {stats_dict['kurtosis']:.4f} (Normal = 3.0)")
    print(f"Excess Kurtosis: {stats_dict['excess_kurtosis']:.4f} (Normal = 0.0)")
    
    if stats_dict['kurtosis'] > 3:
        print(f"  ⚠️  FAT TAILS DETECTED! (K = {stats_dict['kurtosis']:.2f} > 3)")
    
    print(f"\nPercentiles:")
    print(f"  1%: {format_percent(stats_dict['percentile_1'])}")
    print(f"  5%: {format_percent(stats_dict['percentile_5'])}")
    print(f"  95%: {format_percent(stats_dict['percentile_95'])}")
    print(f"  99%: {format_percent(stats_dict['percentile_99'])}")
    print(f"{'='*60}\n")
    
    return stats_dict


def save_figure(fig, filename: str, dpi: int = 300):
    """
    Save a figure to disk.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (without extension)
    dpi : int, default=300
        Resolution in DPI
    """
    os.makedirs('results', exist_ok=True)
    filepath = f'results/{filename}.png'
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"💾 Figure saved: {filepath}")

