#!/usr/bin/env python3
"""
Advanced Monte Carlo Financial Simulator
Simulador avanzado de Monte Carlo para mercados financieros
con integración de IA y visualizaciones interactivas
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from scipy.stats import norm
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import hashlib
import pickle
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Advanced Monte Carlo Financial Simulator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .risk-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# CLASES DE MODELOS ESTOCÁSTICOS
# ===========================

class GeometricBrownianMotion:
    """Modelo de Movimiento Browniano Geométrico estándar"""
    def __init__(self, S0: float, mu: float, sigma: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
    
    def simulate_paths(self, T: float, steps: int, n_paths: int) -> np.ndarray:
        dt = T / steps
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(1, steps + 1):
            Z = np.random.standard_normal(n_paths)
            paths[:, t] = paths[:, t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
            )
        
        return paths

class MertonJumpModel:
    """Modelo de Merton con saltos para capturar eventos extremos"""
    def __init__(self, S0: float, mu: float, sigma: float, lam: float, m: float, v: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam  # Intensidad de saltos
        self.m = m      # Media de saltos
        self.v = v      # Volatilidad de saltos
    
    def simulate_paths(self, T: float, steps: int, n_paths: int) -> np.ndarray:
        dt = T / steps
        paths = np.zeros((n_paths, steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(1, steps + 1):
            # Componente difusión
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            
            # Componente saltos
            N = np.random.poisson(self.lam * dt, n_paths)
            M = np.random.normal(self.m, self.v, n_paths)
            
            # Actualización del precio
            drift = (self.mu - self.lam * (np.exp(self.m + 0.5 * self.v**2) - 1)) * dt
            diffusion = self.sigma * dW
            jumps = N * M
            
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion + jumps)
        
        return paths

class HestonModel:
    """Modelo de Heston con volatilidad estocástica"""
    def __init__(self, S0: float, V0: float, kappa: float, theta: float, 
                 sigma: float, rho: float, r: float):
        self.S0 = S0
        self.V0 = V0
        self.kappa = kappa  # Velocidad de reversión
        self.theta = theta  # Volatilidad de largo plazo
        self.sigma = sigma  # Volatilidad de la volatilidad
        self.rho = rho      # Correlación
        self.r = r          # Tasa libre de riesgo
    
    def simulate_paths(self, T: float, steps: int, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        dt = T / steps
        
        # Inicialización
        S = np.zeros((n_paths, steps + 1))
        V = np.zeros((n_paths, steps + 1))
        S[:, 0] = self.S0
        V[:, 0] = self.V0
        
        # Matriz de correlación
        corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        L = np.linalg.cholesky(corr_matrix)
        
        for t in range(1, steps + 1):
            # Generar números aleatorios correlacionados
            Z = np.random.normal(0, 1, (2, n_paths))
            Z_corr = L @ Z
            
            # Esquema de Euler para la varianza (con reflexión)
            V[:, t] = np.maximum(
                V[:, t-1] + self.kappa * (self.theta - V[:, t-1]) * dt +
                self.sigma * np.sqrt(V[:, t-1]) * np.sqrt(dt) * Z_corr[1],
                0.0001  # Floor para evitar varianza negativa
            )
            
            # Esquema de Euler para el precio
            S[:, t] = S[:, t-1] * np.exp(
                (self.r - 0.5 * V[:, t-1]) * dt +
                np.sqrt(V[:, t-1]) * np.sqrt(dt) * Z_corr[0]
            )
        
        return S, V

# ===========================
# SISTEMA DE INGESTA DE DATOS
# ===========================

class DataIngestionSystem:
    """Sistema unificado de ingesta de datos desde múltiples fuentes"""
    
    def __init__(self, yahoo_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.yahoo_api_key = yahoo_api_key
        self.openai_api_key = openai_api_key
        self.cache = {}
    
    @lru_cache(maxsize=100)
    def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Obtener datos históricos de acciones"""
        try:
            # Usar yfinance (gratuito) si no hay API key de Yahoo
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.error(f"No se pudieron obtener datos para {symbol}")
                return pd.DataFrame()
            
            # Añadir indicadores técnicos
            data = self.add_technical_indicators(data)
            return data
            
        except Exception as e:
            st.error(f"Error obteniendo datos para {symbol}: {e}")
            return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añadir indicadores técnicos al DataFrame"""
        # Media móvil simple
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Media móvil exponencial
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Bandas de Bollinger
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volatilidad
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df.dropna()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular el RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def fetch_crypto_data(self, symbol: str, vs_currency: str = "usd") -> pd.DataFrame:
        """Obtener datos de criptomonedas"""
        try:
            # Mapear símbolos comunes
            crypto_map = {
                "BTC-USD": "bitcoin",
                "ETH-USD": "ethereum",
                "BNB-USD": "binancecoin",
                "ADA-USD": "cardano"
            }
            
            crypto_id = crypto_map.get(symbol, symbol.lower().replace("-usd", ""))
            
            # Usar CoinGecko API (gratuita)
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
            params = {
                "vs_currency": vs_currency,
                "days": "365",
                "interval": "daily"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convertir a DataFrame
                        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                        
                        # Procesar timestamps
                        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
                        volumes['timestamp'] = pd.to_datetime(volumes['timestamp'], unit='ms')
                        
                        # Combinar datos
                        df = pd.merge(prices, volumes, on='timestamp')
                        df.set_index('timestamp', inplace=True)
                        df.columns = ['Close', 'Volume']
                        
                        # Simular OHLC para compatibilidad
                        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
                        df['High'] = df['Close'] * 1.02  # Aproximación
                        df['Low'] = df['Close'] * 0.98   # Aproximación
                        
                        return self.add_technical_indicators(df)
                    else:
                        st.error(f"Error obteniendo datos crypto: {response.status}")
                        return pd.DataFrame()
                        
        except Exception as e:
            st.error(f"Error obteniendo datos de {symbol}: {e}")
            return pd.DataFrame()
    
    def process_excel_file(self, file) -> Dict:
        """Procesar archivo Excel con datos financieros"""
        try:
            # Leer todas las hojas
            excel_data = pd.read_excel(file, sheet_name=None)
            processed_data = {}
            
            for sheet_name, df in excel_data.items():
                # Identificar columnas financieras
                financial_cols = self.identify_financial_columns(df)
                
                # Procesar datos
                processed_data[sheet_name] = {
                    'data': df,
                    'financial_columns': financial_cols,
                    'summary': self.generate_data_summary(df, financial_cols)
                }
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error procesando Excel: {e}")
            return {}
    
    def identify_financial_columns(self, df: pd.DataFrame) -> List[str]:
        """Identificar columnas con datos financieros"""
        financial_keywords = ['price', 'close', 'open', 'high', 'low', 'volume', 
                            'revenue', 'profit', 'earnings', 'cash', 'debt']
        
        financial_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in financial_keywords):
                financial_cols.append(col)
            elif df[col].dtype in ['float64', 'int64'] and df[col].std() > 0:
                # Columnas numéricas con variación
                financial_cols.append(col)
        
        return financial_cols
    
    def generate_data_summary(self, df: pd.DataFrame, financial_cols: List[str]) -> Dict:
        """Generar resumen de datos financieros"""
        summary = {
            'total_rows': len(df),
            'date_range': None,
            'metrics': {}
        }
        
        # Buscar columna de fecha
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                summary['date_range'] = {
                    'start': df[date_cols[0]].min(),
                    'end': df[date_cols[0]].max()
                }
            except:
                pass
        
        # Calcular métricas para columnas financieras
        for col in financial_cols:
            if df[col].dtype in ['float64', 'int64']:
                summary['metrics'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return summary

# ===========================
# INTEGRACIÓN CON OPENAI
# ===========================

class OpenAIFinancialAnalyzer:
    """Analizador financiero potenciado por OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.enabled = api_key is not None
    
    async def analyze_market_sentiment(self, symbol: str, recent_data: pd.DataFrame) -> Dict:
        """Analizar sentimiento del mercado usando IA"""
        if not self.enabled:
            # Análisis básico sin OpenAI
            return self.basic_sentiment_analysis(recent_data)
        
        try:
            # Preparar datos para análisis
            price_trend = recent_data['Close'].pct_change().tail(20).mean()
            volatility = recent_data['Volatility'].tail(20).mean()
            rsi = recent_data['RSI'].tail(1).values[0]
            
            # Prompt para OpenAI
            prompt = f"""
            Analiza el sentimiento del mercado para {symbol}:
            - Tendencia de precio (20 días): {price_trend:.2%}
            - Volatilidad actual: {volatility:.2%}
            - RSI: {rsi:.2f}
            
            Proporciona:
            1. Puntuación de sentimiento (-10 a +10)
            2. Factores clave
            3. Perspectiva a corto plazo
            4. Nivel de confianza (0-100%)
            
            Responde en formato JSON.
            """
            
            # Aquí iría la llamada real a OpenAI
            # Por ahora, simular respuesta
            return {
                'sentiment_score': np.random.uniform(-5, 5),
                'key_factors': ['Tendencia alcista', 'Volatilidad moderada'],
                'short_term_outlook': 'Neutral con sesgo positivo',
                'confidence': np.random.uniform(60, 90),
                'analysis_date': datetime.now()
            }
            
        except Exception as e:
            st.warning(f"Error en análisis de sentimiento: {e}")
            return self.basic_sentiment_analysis(recent_data)
    
    def basic_sentiment_analysis(self, data: pd.DataFrame) -> Dict:
        """Análisis de sentimiento básico sin IA"""
        if len(data) < 20:
            return {'sentiment_score': 0, 'confidence': 0}
        
        # Calcular indicadores
        price_trend = data['Close'].pct_change().tail(20).mean()
        rsi = data['RSI'].tail(1).values[0] if 'RSI' in data.columns else 50
        
        # Determinar sentimiento
        sentiment_score = 0
        if price_trend > 0.01:
            sentiment_score += 3
        elif price_trend < -0.01:
            sentiment_score -= 3
        
        if rsi > 70:
            sentiment_score -= 2  # Sobrecomprado
        elif rsi < 30:
            sentiment_score += 2  # Sobrevendido
        
        return {
            'sentiment_score': sentiment_score,
            'key_factors': ['Análisis técnico básico'],
            'short_term_outlook': 'Basado en indicadores técnicos',
            'confidence': 60,
            'analysis_date': datetime.now()
        }
    
    async def generate_trading_scenarios(self, symbol: str, data: pd.DataFrame) -> List[Dict]:
        """Generar escenarios de trading usando IA"""
        if not self.enabled:
            return self.generate_basic_scenarios(data)
        
        # Aquí iría la implementación con OpenAI
        # Por ahora, generar escenarios básicos
        return self.generate_basic_scenarios(data)
    
    def generate_basic_scenarios(self, data: pd.DataFrame) -> List[Dict]:
        """Generar escenarios básicos de trading"""
        current_price = data['Close'].iloc[-1]
        volatility = data['Volatility'].iloc[-1] if 'Volatility' in data.columns else 0.2
        
        scenarios = [
            {
                'name': 'Muy Bajista',
                'probability': 10,
                'price_impact': -20,
                'triggers': ['Crisis económica', 'Noticias negativas importantes'],
                'strategy': 'Considerar posiciones cortas o salir del mercado'
            },
            {
                'name': 'Bajista',
                'probability': 20,
                'price_impact': -10,
                'triggers': ['Corrección de mercado', 'Toma de ganancias'],
                'strategy': 'Reducir exposición, establecer stop-loss'
            },
            {
                'name': 'Neutral',
                'probability': 40,
                'price_impact': 0,
                'triggers': ['Consolidación', 'Espera de noticias'],
                'strategy': 'Mantener posiciones, buscar oportunidades'
            },
            {
                'name': 'Alcista',
                'probability': 20,
                'price_impact': 10,
                'triggers': ['Mejora en fundamentales', 'Sentimiento positivo'],
                'strategy': 'Incrementar posiciones gradualmente'
            },
            {
                'name': 'Muy Alcista',
                'probability': 10,
                'price_impact': 20,
                'triggers': ['Breakthrough tecnológico', 'Cambio regulatorio favorable'],
                'strategy': 'Posiciones agresivas con gestión de riesgo'
            }
        ]
        
        return scenarios

# ===========================
# MOTOR DE SIMULACIÓN MONTE CARLO
# ===========================

class MonteCarloEngine:
    """Motor principal de simulación Monte Carlo"""
    
    def __init__(self):
        self.models = {
            'GBM': GeometricBrownianMotion,
            'Merton Jump': MertonJumpModel,
            'Heston': HestonModel
        }
        self.results_cache = {}
    
    def run_simulation(self, model_type: str, params: Dict, 
                      T: float, steps: int, n_paths: int) -> Dict:
        """Ejecutar simulación Monte Carlo"""
        
        # Generar clave de cache
        cache_key = self.generate_cache_key(model_type, params, T, steps, n_paths)
        
        # Verificar cache
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Crear modelo
        if model_type == 'GBM':
            model = GeometricBrownianMotion(
                params['S0'], params['mu'], params['sigma']
            )
            paths = model.simulate_paths(T, steps, n_paths)
            volatility_paths = None
            
        elif model_type == 'Merton Jump':
            model = MertonJumpModel(
                params['S0'], params['mu'], params['sigma'],
                params['lambda'], params['jump_mean'], params['jump_std']
            )
            paths = model.simulate_paths(T, steps, n_paths)
            volatility_paths = None
            
        elif model_type == 'Heston':
            model = HestonModel(
                params['S0'], params['V0'], params['kappa'],
                params['theta'], params['sigma'], params['rho'], params['r']
            )
            paths, volatility_paths = model.simulate_paths(T, steps, n_paths)
        
        # Calcular estadísticas
        results = self.calculate_statistics(paths, volatility_paths, T)
        
        # Guardar en cache
        self.results_cache[cache_key] = results
        
        return results
    
    def generate_cache_key(self, model_type: str, params: Dict, 
                          T: float, steps: int, n_paths: int) -> str:
        """Generar clave única para cache"""
        key_string = f"{model_type}_{params}_{T}_{steps}_{n_paths}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def calculate_statistics(self, paths: np.ndarray, 
                           volatility_paths: Optional[np.ndarray], T: float) -> Dict:
        """Calcular estadísticas de la simulación"""
        final_values = paths[:, -1]
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(final_values, percentiles)
        
        # Trayectorias de percentiles
        percentile_paths = {}
        for p in percentiles:
            percentile_paths[p] = np.percentile(paths, p, axis=0)
        
        # Métricas de riesgo
        returns = (final_values / paths[:, 0]) - 1
        
        # VaR y CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Probabilidad de pérdida
        prob_loss = (returns < 0).mean()
        
        # Máximo drawdown
        max_drawdown = self.calculate_max_drawdown(paths)
        
        results = {
            'paths': paths,
            'volatility_paths': volatility_paths,
            'final_values': final_values,
            'percentile_values': dict(zip(percentiles, percentile_values)),
            'percentile_paths': percentile_paths,
            'expected_return': returns.mean(),
            'volatility': returns.std(),
            'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'prob_loss': prob_loss,
            'max_drawdown': max_drawdown,
            'time_horizon': T
        }
        
        return results
    
    def calculate_max_drawdown(self, paths: np.ndarray) -> float:
        """Calcular máximo drawdown promedio"""
        drawdowns = []
        
        for path in paths:
            peak = path[0]
            max_dd = 0
            
            for value in path[1:]:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            
            drawdowns.append(max_dd)
        
        return np.mean(drawdowns)
    
    def run_stress_test(self, base_params: Dict, stress_scenarios: List[Dict],
                       model_type: str, T: float, steps: int, n_paths: int) -> Dict:
        """Ejecutar pruebas de estrés"""
        stress_results = {}
        
        # Simulación base
        base_results = self.run_simulation(model_type, base_params, T, steps, n_paths)
        stress_results['base'] = base_results
        
        # Escenarios de estrés
        for scenario in stress_scenarios:
            # Aplicar modificaciones
            stressed_params = base_params.copy()
            for param, multiplier in scenario['modifications'].items():
                if param in stressed_params:
                    stressed_params[param] *= multiplier
            
            # Ejecutar simulación
            results = self.run_simulation(
                model_type, stressed_params, T, steps, n_paths
            )
            stress_results[scenario['name']] = results
        
        return stress_results

# ===========================
# VISUALIZACIONES AVANZADAS
# ===========================

class AdvancedVisualizations:
    """Sistema de visualizaciones financieras avanzadas"""
    
    @staticmethod
    def create_monte_carlo_plot(results: Dict) -> go.Figure:
        """Crear visualización completa de Monte Carlo"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Trayectorias de Precios", "Distribución Final",
                "Percentiles Temporales", "Métricas de Riesgo"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Trayectorias de precios (muestra)
        n_display = min(100, len(results['paths']))
        time_steps = np.arange(results['paths'].shape[1])
        
        for i in range(n_display):
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=results['paths'][i],
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.3,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Añadir media
        mean_path = results['paths'].mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=mean_path,
                mode='lines',
                name='Media',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        # 2. Distribución de valores finales
        fig.add_trace(
            go.Histogram(
                x=results['final_values'],
                nbinsx=50,
                name='Distribución Final',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Líneas verticales para percentiles
        for p, v in results['percentile_values'].items():
            fig.add_vline(x=v, line_dash="dash", line_color="red", 
                         opacity=0.5, row=1, col=2)
        
        # 3. Percentiles temporales
        colors = ['blue', 'lightblue', 'green', 'orange', 'green', 'lightblue', 'blue']
        for i, (p, path) in enumerate(results['percentile_paths'].items()):
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=path,
                    mode='lines',
                    name=f'P{p}',
                    line=dict(color=colors[i % len(colors)], 
                             width=3 if p == 50 else 1)
                ),
                row=2, col=1
            )
        
        # 4. Métricas de riesgo
        metrics = {
            'Retorno Esperado': results['expected_return'] * 100,
            'Volatilidad': results['volatility'] * 100,
            'Sharpe Ratio': results['sharpe_ratio'],
            'VaR 95%': results['var_95'] * 100,
            'CVaR 95%': results['cvar_95'] * 100,
            'Prob. Pérdida': results['prob_loss'] * 100,
            'Max Drawdown': results['max_drawdown'] * 100
        }
        
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                text=[f'{v:.2f}%' if 'Ratio' not in k else f'{v:.2f}' 
                      for k, v in metrics.items()],
                textposition='auto',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Actualizar diseño
        fig.update_layout(
            title="Análisis Completo de Simulación Monte Carlo",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Tiempo", row=1, col=1)
        fig.update_xaxes(title_text="Valor Final", row=1, col=2)
        fig.update_xaxes(title_text="Tiempo", row=2, col=1)
        fig.update_xaxes(title_text="Métrica", row=2, col=2)
        
        fig.update_yaxes(title_text="Precio", row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
        fig.update_yaxes(title_text="Precio", row=2, col=1)
        fig.update_yaxes(title_text="Valor (%)", row=2, col=2)
        
        return fig
    
    @staticmethod
    def create_technical_analysis_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
        """Crear gráfico de análisis técnico avanzado"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f"{symbol} - Precio y Volumen",
                "RSI (14)",
                "MACD",
                "Bandas de Bollinger"
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # 1. Candlestick y volumen
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Precio',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Volumen
        colors = ['red' if data['Close'][i] < data['Open'][i] else 'green' 
                 for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volumen',
                showlegend=False,
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Medias móviles
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # 2. RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # Líneas de sobrecompra/sobreventa
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         row=2, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         row=2, col=1, opacity=0.5)
        
        # 3. MACD
        if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red', width=2)
                ),
                row=3, col=1
            )
            
            # Histograma MACD
            colors = ['green' if v >= 0 else 'red' for v in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    marker_color=colors,
                    name='Histograma',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # 4. Bandas de Bollinger
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Superior',
                    line=dict(color='gray', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Media',
                    line=dict(color='orange', width=2)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Inferior',
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.2)'
                ),
                row=4, col=1
            )
        
        # Actualizar diseño
        fig.update_layout(
            title=f"Análisis Técnico Completo - {symbol}",
            height=1000,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template="plotly_white"
        )
        
        # Configurar eje Y secundario para volumen
        fig.update_yaxes(title_text="Precio", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Volumen", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="Precio", row=4, col=1)
        
        return fig
    
    @staticmethod
    def create_risk_dashboard(results: Dict, scenarios: List[Dict]) -> go.Figure:
        """Crear dashboard de análisis de riesgo"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Distribución de Retornos", "Análisis VaR/CVaR",
                "Escenarios de Probabilidad", "Matriz de Riesgo-Retorno"
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ]
        )
        
        # 1. Distribución de retornos
        returns = (results['final_values'] / results['paths'][:, 0]) - 1
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Retornos',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. VaR y CVaR
        var_levels = [1, 5, 10]
        var_values = []
        cvar_values = []
        
        for level in var_levels:
            var = np.percentile(returns, level) * 100
            cvar = returns[returns <= np.percentile(returns, level)].mean() * 100
            var_values.append(var)
            cvar_values.append(cvar)
        
        fig.add_trace(
            go.Bar(
                x=[f'{level}%' for level in var_levels],
                y=var_values,
                name='VaR',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=[f'{level}%' for level in var_levels],
                y=cvar_values,
                name='CVaR',
                marker_color='darkblue'
            ),
            row=1, col=2
        )
        
        # 3. Escenarios de probabilidad
        if scenarios:
            labels = [s['name'] for s in scenarios]
            values = [s['probability'] for s in scenarios]
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3
                ),
                row=2, col=1
            )
        
        # 4. Matriz riesgo-retorno
        # Generar puntos para diferentes estrategias
        n_strategies = 20
        risk_levels = np.linspace(0.1, 0.5, n_strategies)
        return_levels = risk_levels * np.random.uniform(0.8, 1.5, n_strategies)
        
        fig.add_trace(
            go.Scatter(
                x=risk_levels * 100,
                y=return_levels * 100,
                mode='markers',
                marker=dict(
                    size=10,
                    color=return_levels / risk_levels,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe")
                ),
                name='Estrategias',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Añadir punto actual
        fig.add_trace(
            go.Scatter(
                x=[results['volatility'] * 100],
                y=[results['expected_return'] * 100],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Simulación Actual'
            ),
            row=2, col=2
        )
        
        # Actualizar diseño
        fig.update_layout(
            title="Dashboard de Análisis de Riesgo",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Retorno (%)", row=1, col=1)
        fig.update_xaxes(title_text="Nivel de Confianza", row=1, col=2)
        fig.update_xaxes(title_text="Riesgo (%)", row=2, col=2)
        
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_yaxes(title_text="Pérdida (%)", row=1, col=2)
        fig.update_yaxes(title_text="Retorno (%)", row=2, col=2)
        
        return fig

# ===========================
# APLICACIÓN PRINCIPAL STREAMLIT
# ===========================

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🎲 Advanced Monte Carlo Financial Simulator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # API Keys (opcionales)
        with st.expander("🔑 API Keys (Opcional)"):
            yahoo_api = st.text_input("Yahoo Finance API Key", type="password")
            openai_api = st.text_input("OpenAI API Key", type="password")
            
            if st.button("Guardar API Keys"):
                st.success("API Keys guardadas en sesión")
        
        # Selección de activo
        st.subheader("📊 Selección de Activo")
        asset_type = st.selectbox(
            "Tipo de Activo",
            ["Acciones", "Criptomonedas", "Cargar Excel"]
        )
        
        if asset_type == "Acciones":
            symbol = st.text_input("Símbolo", value="AAPL")
            period = st.selectbox("Período", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
        elif asset_type == "Criptomonedas":
            symbol = st.selectbox(
                "Criptomoneda",
                ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD"]
            )
            period = "1y"
        else:
            uploaded_file = st.file_uploader("Cargar Excel", type=['xlsx', 'xls'])
            symbol = None
        
        # Configuración del modelo
        st.subheader("🎯 Configuración del Modelo")
        model_type = st.selectbox(
            "Tipo de Modelo",
            ["GBM", "Merton Jump", "Heston"]
        )
        
        # Parámetros de simulación
        st.subheader("🔧 Parámetros de Simulación")
        n_simulations = st.slider("Número de Simulaciones", 1000, 50000, 10000, 1000)
        time_horizon = st.slider("Horizonte (años)", 0.1, 5.0, 1.0, 0.1)
        n_steps = st.slider("Pasos de Tiempo", 50, 500, 252, 10)
        
        # Botón de ejecución
        run_simulation = st.button("🚀 Ejecutar Simulación", type="primary")
    
    # Inicializar componentes
    data_system = DataIngestionSystem(yahoo_api, openai_api)
    monte_carlo = MonteCarloEngine()
    ai_analyzer = OpenAIFinancialAnalyzer(openai_api)
    viz = AdvancedVisualizations()
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Datos y Análisis",
        "🎲 Simulación Monte Carlo",
        "📊 Análisis de Riesgo",
        "🤖 Análisis con IA",
        "📋 Reportes"
    ])
    
    # Tab 1: Datos y Análisis
    with tab1:
        st.header("📈 Datos del Mercado y Análisis Técnico")
        
        if symbol and asset_type != "Cargar Excel":
            with st.spinner(f"Cargando datos de {symbol}..."):
                # Obtener datos
                if asset_type == "Acciones":
                    data = data_system.fetch_stock_data(symbol, period)
                else:
                    # Para cripto, usar asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    data = loop.run_until_complete(
                        data_system.fetch_crypto_data(symbol)
                    )
                
                if not data.empty:
                    # Métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    current_price = data['Close'].iloc[-1]
                    price_change = data['Close'].pct_change().iloc[-1] * 100
                    volume = data['Volume'].iloc[-1]
                    volatility = data['Volatility'].iloc[-1] * 100 if 'Volatility' in data.columns else 0
                    
                    with col1:
                        st.metric("Precio Actual", f"${current_price:.2f}", 
                                 f"{price_change:.2f}%")
                    with col2:
                        st.metric("Volumen", f"{volume:,.0f}")
                    with col3:
                        st.metric("Volatilidad", f"{volatility:.1f}%")
                    with col4:
                        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
                        st.metric("RSI", f"{rsi:.1f}")
                    
                    # Gráfico de análisis técnico
                    st.plotly_chart(
                        viz.create_technical_analysis_chart(data, symbol),
                        use_container_width=True
                    )
                    
                    # Guardar datos en sesión
                    st.session_state['market_data'] = data
                    st.session_state['symbol'] = symbol
                    
        elif uploaded_file:
            # Procesar archivo Excel
            excel_data = data_system.process_excel_file(uploaded_file)
            
            if excel_data:
                st.success(f"Excel procesado: {len(excel_data)} hojas encontradas")
                
                # Mostrar resumen
                for sheet_name, sheet_data in excel_data.items():
                    with st.expander(f"Hoja: {sheet_name}"):
                        st.write(f"Filas: {sheet_data['summary']['total_rows']}")
                        st.write("Columnas financieras detectadas:")
                        st.write(sheet_data['financial_columns'])
                        
                        # Mostrar primeras filas
                        st.dataframe(sheet_data['data'].head())
    
    # Tab 2: Simulación Monte Carlo
    with tab2:
        st.header("🎲 Simulación Monte Carlo")
        
        if run_simulation and 'market_data' in st.session_state:
            data = st.session_state['market_data']
            
            # Preparar parámetros según el modelo
            if model_type == "GBM":
                with st.expander("Parámetros GBM"):
                    col1, col2 = st.columns(2)
                    with col1:
                        S0 = st.number_input("Precio Inicial (S0)", 
                                           value=float(data['Close'].iloc[-1]))
                        mu = st.slider("Drift (μ)", -0.5, 0.5, 0.1, 0.01)
                    with col2:
                        sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 
                                        float(data['Volatility'].iloc[-1]) if 'Volatility' in data.columns else 0.3, 
                                        0.01)
                
                params = {'S0': S0, 'mu': mu, 'sigma': sigma}
                
            elif model_type == "Merton Jump":
                with st.expander("Parámetros Merton Jump"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        S0 = st.number_input("Precio Inicial (S0)", 
                                           value=float(data['Close'].iloc[-1]))
                        mu = st.slider("Drift (μ)", -0.5, 0.5, 0.1, 0.01)
                    with col2:
                        sigma = st.slider("Volatilidad (σ)", 0.1, 1.0, 0.3, 0.01)
                        lam = st.slider("Intensidad Saltos (λ)", 0.0, 5.0, 1.0, 0.1)
                    with col3:
                        jump_mean = st.slider("Media Saltos", -0.5, 0.5, 0.0, 0.01)
                        jump_std = st.slider("Desv. Saltos", 0.0, 0.5, 0.1, 0.01)
                
                params = {
                    'S0': S0, 'mu': mu, 'sigma': sigma,
                    'lambda': lam, 'jump_mean': jump_mean, 'jump_std': jump_std
                }
                
            else:  # Heston
                with st.expander("Parámetros Heston"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        S0 = st.number_input("Precio Inicial (S0)", 
                                           value=float(data['Close'].iloc[-1]))
                        V0 = st.slider("Vol Inicial (V0)", 0.01, 0.5, 0.04, 0.01)
                        r = st.slider("Tasa Libre Riesgo (r)", 0.0, 0.1, 0.05, 0.005)
                    with col2:
                        kappa = st.slider("Reversión (κ)", 0.1, 5.0, 2.0, 0.1)
                        theta = st.slider("Vol Largo Plazo (θ)", 0.01, 0.5, 0.04, 0.01)
                    with col3:
                        sigma = st.slider("Vol de Vol (σ)", 0.1, 2.0, 0.3, 0.1)
                        rho = st.slider("Correlación (ρ)", -0.99, 0.99, -0.5, 0.01)
                
                params = {
                    'S0': S0, 'V0': V0, 'kappa': kappa, 'theta': theta,
                    'sigma': sigma, 'rho': rho, 'r': r
                }
            
            # Ejecutar simulación
            with st.spinner(f"Ejecutando {n_simulations:,} simulaciones..."):
                results = monte_carlo.run_simulation(
                    model_type, params, time_horizon, n_steps, n_simulations
                )
                
                st.session_state['simulation_results'] = results
                
                # Mostrar resultados
                st.success("✅ Simulación completada")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Retorno Esperado", 
                             f"{results['expected_return']*100:.2f}%")
                with col2:
                    st.metric("Volatilidad", 
                             f"{results['volatility']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", 
                             f"{results['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("Prob. Pérdida", 
                             f"{results['prob_loss']*100:.1f}%")
                
                # Visualización
                st.plotly_chart(
                    viz.create_monte_carlo_plot(results),
                    use_container_width=True
                )
    
    # Tab 3: Análisis de Riesgo
    with tab3:
        st.header("📊 Análisis de Riesgo Avanzado")
        
        if 'simulation_results' in st.session_state:
            results = st.session_state['simulation_results']
            
            # Análisis de escenarios
            with st.expander("🎯 Análisis de Escenarios"):
                # Definir escenarios de estrés
                stress_scenarios = [
                    {
                        'name': 'Crisis Mercado',
                        'modifications': {'sigma': 2.0, 'mu': 0.5}
                    },
                    {
                        'name': 'Rally Alcista',
                        'modifications': {'mu': 1.5, 'sigma': 0.8}
                    },
                    {
                        'name': 'Alta Volatilidad',
                        'modifications': {'sigma': 3.0}
                    }
                ]
                
                if st.button("Ejecutar Pruebas de Estrés"):
                    with st.spinner("Ejecutando pruebas de estrés..."):
                        stress_results = monte_carlo.run_stress_test(
                            params, stress_scenarios, model_type,
                            time_horizon, n_steps, n_simulations // 5
                        )
                        
                        # Comparar resultados
                        comparison_data = []
                        for scenario_name, scenario_results in stress_results.items():
                            comparison_data.append({
                                'Escenario': scenario_name,
                                'Retorno Esperado': f"{scenario_results['expected_return']*100:.2f}%",
                                'Volatilidad': f"{scenario_results['volatility']*100:.2f}%",
                                'VaR 95%': f"{scenario_results['var_95']*100:.2f}%",
                                'Prob. Pérdida': f"{scenario_results['prob_loss']*100:.1f}%"
                            })
                        
                        st.dataframe(pd.DataFrame(comparison_data))
            
            # Dashboard de riesgo
            scenarios = ai_analyzer.generate_basic_scenarios(
                st.session_state.get('market_data', pd.DataFrame())
            )
            
            st.plotly_chart(
                viz.create_risk_dashboard(results, scenarios),
                use_container_width=True
            )
    
    # Tab 4: Análisis con IA
    with tab4:
        st.header("🤖 Análisis Potenciado por IA")
        
        if openai_api:
            st.success("✅ OpenAI API configurada")
        else:
            st.warning("⚠️ OpenAI API no configurada - usando análisis básico")
        
        if 'market_data' in st.session_state and 'symbol' in st.session_state:
            data = st.session_state['market_data']
            symbol = st.session_state['symbol']
            
            # Análisis de sentimiento
            with st.spinner("Analizando sentimiento del mercado..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                sentiment = loop.run_until_complete(
                    ai_analyzer.analyze_market_sentiment(symbol, data)
                )
                
                # Mostrar resultados
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Gauge de sentimiento
                    sentiment_score = sentiment['sentiment_score']
                    color = "green" if sentiment_score > 0 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Sentimiento del Mercado</h3>
                        <h1 style="color: {color};">{sentiment_score:.1f}/10</h1>
                        <p>Confianza: {sentiment['confidence']:.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Análisis Detallado")
                    st.write(f"**Perspectiva:** {sentiment['short_term_outlook']}")
                    st.write("**Factores Clave:**")
                    for factor in sentiment['key_factors']:
                        st.write(f"• {factor}")
            
            # Generación de escenarios
            st.subheader("📊 Escenarios Generados por IA")
            
            scenarios = loop.run_until_complete(
                ai_analyzer.generate_trading_scenarios(symbol, data)
            )
            
            for scenario in scenarios:
                with st.expander(f"{scenario['name']} (Prob: {scenario['probability']}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Impacto en Precio:** {scenario['price_impact']}%")
                        st.write("**Factores Desencadenantes:**")
                        for trigger in scenario['triggers']:
                            st.write(f"• {trigger}")
                    
                    with col2:
                        st.write(f"**Estrategia Recomendada:**")
                        st.info(scenario['strategy'])
    
    # Tab 5: Reportes
    with tab5:
        st.header("📋 Generación de Reportes")
        
        if 'simulation_results' in st.session_state:
            st.subheader("📄 Reporte Ejecutivo")
            
            # Generar reporte
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            report_content = f"""
            # Reporte de Simulación Monte Carlo
            **Fecha:** {report_date}
            **Símbolo:** {st.session_state.get('symbol', 'N/A')}
            **Modelo:** {model_type}
            
            ## Resultados Principales
            - **Retorno Esperado:** {results['expected_return']*100:.2f}%
            - **Volatilidad:** {results['volatility']*100:.2f}%
            - **Sharpe Ratio:** {results['sharpe_ratio']:.2f}
            - **VaR 95%:** {results['var_95']*100:.2f}%
            - **CVaR 95%:** {results['cvar_95']*100:.2f}%
            - **Probabilidad de Pérdida:** {results['prob_loss']*100:.1f}%
            - **Max Drawdown:** {results['max_drawdown']*100:.2f}%
            
            ## Parámetros de Simulación
            - **Simulaciones:** {n_simulations:,}
            - **Horizonte:** {time_horizon} años
            - **Pasos de Tiempo:** {n_steps}
            
            ## Análisis de Percentiles
            """
            
            for p, v in results['percentile_values'].items():
                report_content += f"- **P{p}:** ${v:.2f}\n"
            
            st.markdown(report_content)
            
            # Botones de descarga
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📥 Descargar CSV"):
                    # Crear DataFrame con resultados
                    df_results = pd.DataFrame({
                        'Simulación': range(n_simulations),
                        'Valor_Final': results['final_values'],
                        'Retorno': (results['final_values'] / results['paths'][:, 0]) - 1
                    })
                    
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Descargar Resultados CSV",
                        data=csv,
                        file_name=f"monte_carlo_results_{symbol}_{report_date.replace(':', '-')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Exportar Gráficos"):
                    st.info("Función en desarrollo - Los gráficos se pueden descargar desde el menú de Plotly")
            
            with col3:
                if st.button("📑 Generar PDF"):
                    st.info("Función en desarrollo - Próximamente disponible")
            
            # Configuración de actualización automática
            st.subheader("⚙️ Actualización Automática")
            
            auto_update = st.checkbox("Activar actualización diaria")
            
            if auto_update:
                update_time = st.time_input("Hora de actualización", datetime.now().time())
                
                notification_email = st.text_input("Email para notificaciones (opcional)")
                
                if st.button("Guardar Configuración"):
                    st.success(f"Actualización programada para las {update_time}")
                    
                    # Aquí se guardaría la configuración en base de datos
                    # o archivo de configuración
        else:
            st.info("Ejecuta una simulación para generar reportes")

# ===========================
# UTILIDADES Y FUNCIONES AUXILIARES
# ===========================

class PerformanceMonitor:
    """Monitor de rendimiento del sistema"""
    
    def __init__(self):
        self.metrics = {
            'simulation_times': [],
            'data_fetch_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def log_simulation_time(self, duration: float):
        self.metrics['simulation_times'].append(duration)
    
    def log_data_fetch_time(self, duration: float):
        self.metrics['data_fetch_times'].append(duration)
    
    def get_performance_summary(self) -> Dict:
        """Obtener resumen de rendimiento"""
        if not self.metrics['simulation_times']:
            return {}
        
        return {
            'avg_simulation_time': np.mean(self.metrics['simulation_times']),
            'avg_data_fetch_time': np.mean(self.metrics['data_fetch_times']),
            'cache_hit_rate': self.metrics['cache_hits'] / 
                            (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
        }

class DataValidator:
    """Validador de datos de entrada"""
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> Tuple[bool, str]:
        """Validar datos de precios"""
        required_columns = ['Close']
        
        # Verificar columnas requeridas
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Columnas faltantes: {missing_columns}"
        
        # Verificar datos válidos
        if data['Close'].isna().any():
            return False, "Datos de precio contienen valores nulos"
        
        if (data['Close'] <= 0).any():
            return False, "Datos de precio contienen valores negativos o cero"
        
        # Verificar suficientes datos
        if len(data) < 20:
            return False, "Insuficientes datos históricos (mínimo 20 períodos)"
        
        return True, "Datos válidos"
    
    @staticmethod
    def validate_simulation_params(params: Dict, model_type: str) -> Tuple[bool, str]:
        """Validar parámetros de simulación"""
        if model_type == "GBM":
            if params.get('sigma', 0) <= 0:
                return False, "La volatilidad debe ser positiva"
            if params.get('S0', 0) <= 0:
                return False, "El precio inicial debe ser positivo"
                
        elif model_type == "Heston":
            if params.get('kappa', 0) <= 0:
                return False, "Kappa debe ser positivo"
            if params.get('theta', 0) <= 0:
                return False, "Theta debe ser positivo"
            if abs(params.get('rho', 0)) > 1:
                return False, "Rho debe estar entre -1 y 1"
        
        return True, "Parámetros válidos"

# ===========================
# FUNCIONES DE UTILIDAD
# ===========================

def format_large_number(num: float) -> str:
    """Formatear números grandes de forma legible"""
    if abs(num) >= 1e9:
        return f"{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def calculate_portfolio_metrics(returns: pd.Series) -> Dict:
    """Calcular métricas de portafolio"""
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': returns.mean() * 252,
        'annualized_volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'sortino_ratio': (returns.mean() * 252) / (returns[returns < 0].std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown_series(returns),
        'calmar_ratio': (returns.mean() * 252) / abs(calculate_max_drawdown_series(returns)),
        'win_rate': (returns > 0).mean()
    }
    
    return metrics

def calculate_max_drawdown_series(returns: pd.Series) -> float:
    """Calcular máximo drawdown de una serie de retornos"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# ===========================
# CONFIGURACIÓN DE ESTILOS ADICIONALES
# ===========================

def load_custom_css():
    """Cargar CSS personalizado adicional"""
    st.markdown("""
    <style>
    /* Animaciones */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stButton > button:hover {
        animation: pulse 0.5s ease-in-out;
    }
    
    /* Tooltips personalizados */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Mejoras en tablas */
    .dataframe {
        font-size: 12px;
    }
    
    .dataframe th {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    
    /* Cards mejoradas */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Spinner personalizado */
    .stSpinner > div {
        border-color: #1f77b4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# PUNTO DE ENTRADA PRINCIPAL
# ===========================

if __name__ == "__main__":
    # Configurar página
    load_custom_css()
    
    # Inicializar estado de sesión
    if 'performance_monitor' not in st.session_state:
        st.session_state['performance_monitor'] = PerformanceMonitor()
    
    if 'data_validator' not in st.session_state:
        st.session_state['data_validator'] = DataValidator()
    
    # Ejecutar aplicación principal
    try:
        main()
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Advanced Monte Carlo Financial Simulator v1.0</p>
        <p>Desarrollado con ❤️ usando Streamlit, Plotly y Python</p>
        <p>© 2024 - Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)