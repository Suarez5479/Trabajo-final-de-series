"""
Aplicación Streamlit: Análisis de series temporales (Incautación de Estupefacientes)
- Interfaz con TABS superiores (azul + amarillo)
- Carga dataset local, transformación a series por CLASE BIEN
- ACF/PACF, ADF/KPSS, diferenciación automática, ARIMA automática/manual
- Diagnósticos completos
- Comparación entre drogas (correlaciones + VAR)
- Backtest 50% + métricas (RMSE, MAE, MAPE, MSE, Bias, AIC, BIC)
- Optimizado para velocidad
"""

import io
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
sns.set()

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR

from scipy import stats as scipy_stats
from scipy.stats import jarque_bera, anderson, normaltest
import warnings
warnings.filterwarnings("ignore")

# auto_arima opcional
try:
    from pmdarima import auto_arima
    AUTO_ARIMA = True
except Exception:
    AUTO_ARIMA = False

# --------------------
# CONFIG Y ESTILOS CSS
# --------------------
st.set_page_config(page_title="Análisis ARIMA - Incautación Estupefacientes", layout="wide")
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #002D72 0%, #003C91 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        color: white;
        text-align: center;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: 1px;
    }
    .subtitle {
        color: #E2E8F0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 6px solid #FDB813;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .metric-card h4 {
        margin: 0 0 0.5rem 0;
        color: #002D72;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #002D72;
    }
    h1, h2, h3 { color: #002D72 !important; }
    .stButton>button { 
        background-color: #FDB813; 
        color: #002D72; 
        font-weight: 700; 
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stDownloadButton>button { 
        background-color: #002D72; 
        color: #fff; 
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: #002D72;
    }
    .stTabs [aria-selected="true"] {
        background-color: #002D72;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------
# FUNCIONES UTILIDAD
# --------------------
@st.cache_data
def cargar_datos(path):
    """Carga CSV y transforma FECHA HECHO a datetime"""
    if isinstance(path, str):
        df = pd.read_csv(path, encoding='latin-1')
    else:
        df = pd.read_csv(path, encoding='latin-1')
    
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    
    if 'FECHA HECHO' not in df.columns and 'FECHA_HECHO' in df.columns:
        df.rename(columns={'FECHA_HECHO':'FECHA HECHO'}, inplace=True)
    
    df['FECHA HECHO'] = pd.to_datetime(df['FECHA HECHO'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['FECHA HECHO', 'CLASE BIEN', 'CANTIDAD'])
    df['CANTIDAD'] = pd.to_numeric(df['CANTIDAD'], errors='coerce')
    df = df.dropna(subset=['CANTIDAD'])
    df = df.sort_values('FECHA HECHO').reset_index(drop=True)
    return df

def resumen_estadistico_rapido(serie):
    """Retorna diccionario con estadísticas"""
    d = {
        "N (periodos)": int(serie.shape[0]),
        "Total (kg)": float(serie.sum()),
        "Media (kg/periodo)": float(serie.mean()),
        "Mediana": float(serie.median()),
        "Desv. estándar": float(serie.std()),
        "Varianza": float(serie.var()),
        "Min": float(serie.min()),
        "Max": float(serie.max()),
        "Asimetría": float(serie.skew()),
        "Curtosis": float(serie.kurtosis()),
    }
    return d

def tarjetas_metricas_html(stats_dict):
    """Devuelve HTML con tarjetas de métricas principales"""
    media = f"{stats_dict['Media (kg/periodo)']:,.2f}"
    total = f"{stats_dict['Total (kg)']:,.2f}"
    desv = f"{stats_dict['Desv. estándar']:,.2f}"
    maxi = f"{stats_dict['Max']:,.2f}"
    
    html = f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; margin-bottom:1.5rem;">
      <div class="metric-card">
        <h4>Media (kg/periodo)</h4>
        <div class="metric-value">{media}</div>
      </div>
      <div class="metric-card">
        <h4>Total (kg)</h4>
        <div class="metric-value">{total}</div>
      </div>
      <div class="metric-card">
        <h4>Desv. estándar</h4>
        <div class="metric-value">{desv}</div>
      </div>
      <div class="metric-card">
        <h4>Máximo (kg)</h4>
        <div class="metric-value">{maxi}</div>
      </div>
    </div>
    """
    return html

def calcular_metricas(y_true, y_pred):
    """Calcula métricas de evaluación"""
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))) * 100
    bias = np.mean(y_pred - y_true)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "Bias": bias}

def prueba_adf(series):
    """Test ADF para estacionariedad"""
    res = adfuller(series.dropna(), autolag='AIC')
    return {
        "Estadístico": res[0], 
        "p-valor": res[1], 
        "Lags": res[2],
        "N observaciones": res[3], 
        "Valores críticos": res[4],
        "Es estacionaria": res[1] < 0.05
    }

def prueba_kpss(series):
    """Test KPSS para estacionariedad"""
    try:
        res = kpss(series.dropna(), regression='c', nlags='auto')
        return {
            "Estadístico": res[0], 
            "p-valor": res[1], 
            "Lags": res[2],
            "Valores críticos": res[3],
            "Es estacionaria": res[1] > 0.05
        }
    except Exception as e:
        return {"error": str(e)}

def construir_serie(df, clase, freq='D', fillna_method='zero'):
    """Construye serie temporal por droga"""
    df_f = df[df['CLASE BIEN'] == clase].copy()
    ts = df_f.groupby('FECHA HECHO')['CANTIDAD'].sum()
    if ts.empty:
        return pd.Series([], dtype=float)
    inicio = ts.index.min()
    fin = ts.index.max()
    rango = pd.date_range(inicio, fin, freq=freq)
    serie = ts.reindex(rango)
    if fillna_method == 'ffill':
        serie = serie.fillna(method='ffill')
    elif fillna_method == 'bfill':
        serie = serie.fillna(method='bfill')
    else:
        serie = serie.fillna(0.0)
    serie.index.name = 'FECHA'
    return serie

@st.cache_data
def ajustar_modelo_sarimax(y, order=(1,0,0), seasonal_order=(0,0,0,0)):
    """Ajusta modelo SARIMAX con caché"""
    model = SARIMAX(
        y, 
        order=order, 
        seasonal_order=seasonal_order, 
        enforce_stationarity=False, 
        enforce_invertibility=False
    )
    fit = model.fit(disp=False, method='lbfgs', maxiter=50)  # Limitar iteraciones
    return fit

# --------------------
# CARGA DE DATOS
# --------------------
RUTA_POR_DEFECTO = r"C:\Users\ASUS\OneDrive - Universidad Santo Tomás\SANTO TOMAS\8-SEMESTRE\SERIES\trabajofinal\Incautación_de_Estupefacientes._20251117.csv"

# Header principal
st.markdown(
    '<div class="main-header">'
    '<div class="main-title">📊 Análisis de Incautaciones de Estupefacientes</div>'
    '<div class="subtitle">Sistema de análisis de series temporales con modelado ARIMA</div>'
    '</div>',
    unsafe_allow_html=True
)

# Sidebar para carga
with st.sidebar:
    st.title("⚙️ Configuración")
    uploaded = st.file_uploader("📁 Sube archivo CSV", type=['csv'])
    
    if uploaded is None:
        ruta = RUTA_POR_DEFECTO
        st.info(f"📂 Usando: `{os.path.basename(ruta)}`")
    else:
        ruta = uploaded

# Cargar datos
with st.spinner("Cargando datos..."):
    try:
        df = cargar_datos(ruta)
        st.sidebar.success(f"✅ {len(df):,} registros cargados")
    except FileNotFoundError:
        st.error("❌ Archivo no encontrado")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# Lista de drogas
lista_drogas = sorted(df['CLASE BIEN'].unique())
default_idx = lista_drogas.index('COCAINA') if 'COCAINA' in lista_drogas else 0

# --------------------
# TABS PRINCIPALES
# --------------------
tabs = st.tabs([
    "🏠 Inicio", 
    "🔍 Exploración", 
    "🔄 Transformación", 
    "📈 Modelado ARIMA",
    "🔬 Diagnóstico",
    "🎯 Predicción",
    "🔗 Comparación"
])

# ===================
# TAB: INICIO
# ===================
with tabs[0]:
    st.markdown("## Resumen General")
    
    # Estadísticas globales
    serie_total = df.groupby('FECHA HECHO')['CANTIDAD'].sum().sort_index()
    stats_total = resumen_estadistico_rapido(serie_total)
    st.markdown(tarjetas_metricas_html(stats_total), unsafe_allow_html=True)
    
    # Tabla resumen por droga
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📊 Resumen por Droga")
        resumen = df.groupby('CLASE BIEN')['CANTIDAD'].agg([
            ('Registros', 'count'),
            ('Total (kg)', 'sum'),
            ('Media', 'mean'),
            ('Mediana', 'median'),
            ('Desv. est.', 'std'),
            ('Max', 'max')
        ]).reset_index()
        resumen.columns = ['Droga', 'Registros', 'Total (kg)', 'Media', 'Mediana', 'Desv. est.', 'Max']
        st.dataframe(
            resumen.style.format({
                'Total (kg)': '{:,.2f}', 
                'Media': '{:,.2f}', 
                'Mediana': '{:,.2f}',
                'Desv. est.': '{:,.2f}',
                'Max': '{:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.markdown("### 🔝 Top 5 Drogas")
        top5 = resumen.nlargest(5, 'Total (kg)')[['Droga', 'Total (kg)']]
        fig_pie = px.pie(
            top5, 
            values='Total (kg)', 
            names='Droga',
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Gráfica temporal total
    st.markdown("### 📈 Serie Temporal Total")
    fig = px.line(
        x=serie_total.index, 
        y=serie_total.values,
        labels={'x': 'Fecha', 'y': 'Cantidad (kg)'},
        title='Evolución histórica de incautaciones totales'
    )
    fig.update_traces(line_color='#002D72')
    st.plotly_chart(fig, use_container_width=True)

# ===================
# TAB: EXPLORACIÓN
# ===================
with tabs[1]:
    st.markdown("## Exploración de Datos")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        clase_exp = st.selectbox("🔍 Selecciona droga", lista_drogas, index=default_idx, key='exp_clase')
    with col2:
        freq_exp = st.selectbox("📅 Frecuencia", ['D', 'W', 'M'], index=2, key='exp_freq',
                                format_func=lambda x: {'D':'Diaria', 'W':'Semanal', 'M':'Mensual'}[x])
    with col3:
        fill_exp = st.selectbox("🔧 Rellenar NA", ['zero', 'ffill', 'bfill'], key='exp_fill')
    
    serie_exp = construir_serie(df, clase_exp, freq=freq_exp, fillna_method=fill_exp)
    
    if serie_exp.empty:
        st.warning("⚠️ Serie vacía")
    else:
        # Métricas
        stats_exp = resumen_estadistico_rapido(serie_exp)
        st.markdown(tarjetas_metricas_html(stats_exp), unsafe_allow_html=True)
        
        # Gráfica principal
        st.markdown(f"### Serie: {clase_exp}")
        fig = px.line(x=serie_exp.index, y=serie_exp.values, 
                     labels={'x': 'Fecha', 'y': 'Cantidad (kg)'})
        fig.update_traces(line_color='#002D72')
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Distribución")
            fig_hist = px.histogram(serie_exp, nbins=50, 
                                   labels={'value': 'Cantidad', 'count': 'Frecuencia'})
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Boxplot")
            fig_box = px.box(y=serie_exp.values, labels={'y': 'Cantidad (kg)'})
            st.plotly_chart(fig_box, use_container_width=True)

# ===================
# TAB: TRANSFORMACIÓN
# ===================
with tabs[2]:
    st.markdown("## Transformación y Pruebas de Estacionariedad")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        clase_trans = st.selectbox("🔍 Droga", lista_drogas, index=default_idx, key='trans_clase')
    with col2:
        freq_trans = st.selectbox("📅 Frecuencia", ['D', 'W', 'M'], index=2, key='trans_freq',
                                  format_func=lambda x: {'D':'Diaria', 'W':'Semanal', 'M':'Mensual'}[x])
    
    serie_trans = construir_serie(df, clase_trans, freq=freq_trans, fillna_method='zero')
    
    if serie_trans.empty:
        st.warning("⚠️ Serie vacía")
    else:
        # Visualización original vs transformada
        st.markdown("### 🔄 Serie Original vs Transformada")
        
        # Calcular diferenciaciones
        serie_d1 = serie_trans.diff().dropna()
        if freq_trans == 'M' and len(serie_trans) > 12:
            serie_d1s = serie_trans.diff().dropna().diff(12).dropna()
        else:
            serie_d1s = None
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Serie Original',
                'Primera Diferencia (d=1)',
                'ACF Original',
                'PACF Original'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Serie original
        fig.add_trace(
            go.Scatter(x=serie_trans.index, y=serie_trans.values, 
                      name='Original', line=dict(color='#002D72')),
            row=1, col=1
        )
        
        # Primera diferencia
        fig.add_trace(
            go.Scatter(x=serie_d1.index, y=serie_d1.values,
                      name='d=1', line=dict(color='#FDB813')),
            row=1, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tests de estacionariedad en columnas
        st.markdown("### 🧪 Pruebas de Estacionariedad")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📌 Serie Original")
            adf_orig = prueba_adf(serie_trans)
            kpss_orig = prueba_kpss(serie_trans)
            
            st.metric("ADF p-valor", f"{adf_orig['p-valor']:.6f}",
                     delta="Estacionaria" if adf_orig['Es estacionaria'] else "No estacionaria",
                     delta_color="normal" if adf_orig['Es estacionaria'] else "inverse")
            st.metric("KPSS p-valor", f"{kpss_orig['p-valor']:.6f}",
                     delta="Estacionaria" if kpss_orig['Es estacionaria'] else "No estacionaria",
                     delta_color="normal" if kpss_orig['Es estacionaria'] else "inverse")
            
            with st.expander("Ver detalles ADF"):
                st.json(adf_orig)
            with st.expander("Ver detalles KPSS"):
                st.json(kpss_orig)
        
        with col2:
            st.markdown("#### 📌 Primera Diferencia (d=1)")
            adf_d1 = prueba_adf(serie_d1)
            kpss_d1 = prueba_kpss(serie_d1)
            
            st.metric("ADF p-valor", f"{adf_d1['p-valor']:.6f}",
                     delta="Estacionaria" if adf_d1['Es estacionaria'] else "No estacionaria",
                     delta_color="normal" if adf_d1['Es estacionaria'] else "inverse")
            st.metric("KPSS p-valor", f"{kpss_d1['p-valor']:.6f}",
                     delta="Estacionaria" if kpss_d1['Es estacionaria'] else "No estacionaria",
                     delta_color="normal" if kpss_d1['Es estacionaria'] else "inverse")
            
            with st.expander("Ver detalles"):
                st.json({"ADF": adf_d1, "KPSS": kpss_d1})
        
        with col3:
            if serie_d1s is not None:
                st.markdown("#### 📌 Dif. Estacional (d=1, D=1)")
                adf_d1s = prueba_adf(serie_d1s)
                kpss_d1s = prueba_kpss(serie_d1s)
                
                st.metric("ADF p-valor", f"{adf_d1s['p-valor']:.6f}",
                         delta="Estacionaria" if adf_d1s['Es estacionaria'] else "No estacionaria",
                         delta_color="normal" if adf_d1s['Es estacionaria'] else "inverse")
                st.metric("KPSS p-valor", f"{kpss_d1s['p-valor']:.6f}",
                         delta="Estacionaria" if kpss_d1s['Es estacionaria'] else "No estacionaria",
                         delta_color="normal" if kpss_d1s['Es estacionaria'] else "inverse")
            else:
                st.info("Diferenciación estacional disponible solo con frecuencia mensual y suficientes datos")
        
        # ACF y PACF
        st.markdown("### 📊 ACF y PACF")
        
        # Serie Original
        st.markdown("#### Serie Original")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ACF - Serie Original**")
            fig_acf = plt.figure(figsize=(10, 4))
            ax = fig_acf.add_subplot(111)
            plot_acf(serie_trans.dropna(), lags=min(40, len(serie_trans)//2 - 1), ax=ax)
            plt.title("ACF - Serie Original")
            plt.tight_layout()
            st.pyplot(fig_acf)
            plt.close(fig_acf)
        
        with col2:
            st.markdown("**PACF - Serie Original**")
            fig_pacf = plt.figure(figsize=(10, 4))
            ax = fig_pacf.add_subplot(111)
            plot_pacf(serie_trans.dropna(), lags=min(40, len(serie_trans)//2 - 1), method='ywm', ax=ax)
            plt.title("PACF - Serie Original")
            plt.tight_layout()
            st.pyplot(fig_pacf)
            plt.close(fig_pacf)
        
        # Serie Diferenciada (d=1)
        st.markdown("#### Primera Diferencia (d=1)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ACF - Diferenciada**")
            fig_acf_d = plt.figure(figsize=(10, 4))
            ax = fig_acf_d.add_subplot(111)
            plot_acf(serie_d1.dropna(), lags=min(40, len(serie_d1)//2 - 1), ax=ax)
            plt.title("ACF - Primera Diferencia")
            plt.tight_layout()
            st.pyplot(fig_acf_d)
            plt.close(fig_acf_d)
        
        with col2:
            st.markdown("**PACF - Diferenciada**")
            fig_pacf_d = plt.figure(figsize=(10, 4))
            ax = fig_pacf_d.add_subplot(111)
            plot_pacf(serie_d1.dropna(), lags=min(40, len(serie_d1)//2 - 1), method='ywm', ax=ax)
            plt.title("PACF - Primera Diferencia")
            plt.tight_layout()
            st.pyplot(fig_pacf_d)
            plt.close(fig_pacf_d)
        
        # Si hay diferenciación estacional
        if serie_d1s is not None and len(serie_d1s) > 20:
            st.markdown("#### Diferenciación Estacional (d=1, D=1)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**ACF - Dif. Estacional**")
                fig_acf_s = plt.figure(figsize=(10, 4))
                ax = fig_acf_s.add_subplot(111)
                plot_acf(serie_d1s.dropna(), lags=min(40, len(serie_d1s)//2 - 1), ax=ax)
                plt.title("ACF - Diferencia Estacional")
                plt.tight_layout()
                st.pyplot(fig_acf_s)
                plt.close(fig_acf_s)
            
            with col2:
                st.markdown("**PACF - Dif. Estacional**")
                fig_pacf_s = plt.figure(figsize=(10, 4))
                ax = fig_pacf_s.add_subplot(111)
                plot_pacf(serie_d1s.dropna(), lags=min(40, len(serie_d1s)//2 - 1), method='ywm', ax=ax)
                plt.title("PACF - Diferencia Estacional")
                plt.tight_layout()
                st.pyplot(fig_pacf_s)
                plt.close(fig_pacf_s)

# ===================
# TAB: MODELADO
# ===================
with tabs[3]:
    st.markdown("## Modelado ARIMA")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        clase_model = st.selectbox("🔍 Droga", lista_drogas, index=default_idx, key='model_clase')
    with col2:
        freq_model = st.selectbox("📅 Frecuencia", ['D', 'W', 'M'], index=2, key='model_freq',
                                  format_func=lambda x: {'D':'Diaria', 'W':'Semanal', 'M':'Mensual'}[x])
    
    serie_model = construir_serie(df, clase_model, freq=freq_model, fillna_method='zero')
    
    if serie_model.empty:
        st.warning("⚠️ Serie vacía")
    else:
        # Visualización
        st.markdown("### Serie a modelar")
        fig = px.line(x=serie_model.index, y=serie_model.values)
        fig.update_traces(line_color='#002D72')
        st.plotly_chart(fig, use_container_width=True)
        
        # Auto ARIMA
        if AUTO_ARIMA:
            st.markdown("### 🤖 Auto-ARIMA")
            if st.button("🚀 Ejecutar Auto-ARIMA", key='auto_arima_btn'):
                with st.spinner("Buscando mejor modelo..."):
                    try:
                        modelo_auto = auto_arima(
                            serie_model, 
                            seasonal=False, 
                            stepwise=True,
                            suppress_warnings=True, 
                            error_action='ignore',
                            max_p=3, max_q=3, max_d=2,  # Limitar búsqueda
                            n_jobs=1,
                            information_criterion='aic'
                        )
                        st.success(f"✅ Mejor modelo: ARIMA{modelo_auto.order}")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("AIC", f"{modelo_auto.aic():.2f}")
                        col2.metric("BIC", f"{modelo_auto.bic():.2f}")
                        col3.metric("Orden", str(modelo_auto.order))
                        
                        st.session_state['modelo_auto'] = modelo_auto
                        with st.expander("📄 Ver resumen completo"):
                            st.text(modelo_auto.summary())
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # ARIMA Manual
        st.markdown("### ⚙️ ARIMA Manual")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            p = st.number_input("p", 0, 5, 1, key='p_manual')
            P = st.number_input("P", 0, 2, 0, key='P_manual')
        with col2:
            d = st.number_input("d", 0, 2, 1, key='d_manual')
            D = st.number_input("D", 0, 1, 0, key='D_manual')
        with col3:
            q = st.number_input("q", 0, 5, 0, key='q_manual')
            Q = st.number_input("Q", 0, 2, 0, key='Q_manual')
        with col4:
            s = st.number_input("s", 0, 365, 0 if freq_model != 'M' else 12, key='s_manual')
        
        if st.button("🔧 Ajustar modelo manual"):
            with st.spinner("Ajustando SARIMAX..."):
                try:
                    order = (int(p), int(d), int(q))
                    seasonal_order = (int(P), int(D), int(Q), int(s)) if s > 0 else (0,0,0,0)
                    
                    fit = ajustar_modelo_sarimax(serie_model, order, seasonal_order)
                    
                    st.success("✅ Modelo ajustado")
                    col1, col2 = st.columns(2)
                    col1.metric("AIC", f"{fit.aic:.2f}")
                    col2.metric("BIC", f"{fit.bic:.2f}")
                    
                    st.session_state['modelo_fit'] = fit
                    st.session_state['orden_modelo'] = {"order": order, "seasonal_order": seasonal_order}
                    
                    with st.expander("📄 Ver resumen"):
                        st.text(fit.summary().as_text())
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    
# ===================
# TAB: DIAGNÓSTICO
# ===================
with tabs[4]:
    st.markdown("## Diagnóstico del Modelo")
    
    if 'modelo_fit' not in st.session_state:
        st.info("⚠️ Primero ajusta un modelo en la sección 'Modelado ARIMA'")
    else:
        fit = st.session_state['modelo_fit']
        residuos = fit.resid
        
        # Resumen de métricas del modelo
        col1, col2, col3 = st.columns(3)
        col1.metric("AIC", f"{fit.aic:.2f}")
        col2.metric("BIC", f"{fit.bic:.2f}")
        col3.metric("Log-Likelihood", f"{fit.llf:.2f}")
        
        # Gráficos de diagnóstico
        st.markdown("### 📊 Análisis de Residuos")
        
        # Layout 2x2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Residuos en el tiempo")
            fig_r = plt.figure(figsize=(10, 4))
            plt.plot(residuos, color='#002D72', alpha=0.7)
            plt.axhline(0, linestyle='--', color='red', alpha=0.5)
            plt.title("Residuos del modelo")
            plt.ylabel("Residuos")
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)
        
        with col2:
            st.markdown("#### 📊 Distribución de residuos")
            fig_hist = plt.figure(figsize=(10, 4))
            sns.histplot(residuos, kde=True, stat='density', color='#002D72')
            plt.axvline(0, color='red', linestyle='--', alpha=0.5)
            plt.title("Histograma + KDE")
            plt.xlabel("Residuos")
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close(fig_hist)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📉 Q-Q Plot")
            fig_qq = plt.figure(figsize=(8, 6))
            scipy_stats.probplot(residuos, dist="norm", plot=plt)
            plt.title("Q-Q Plot")
            plt.tight_layout()
            st.pyplot(fig_qq)
            plt.close(fig_qq)
        
        with col2:
            st.markdown("#### 📊 ACF de residuos")
            fig_acf = plt.figure(figsize=(10, 4))
            ax = fig_acf.add_subplot(111)
            plot_acf(residuos, lags=min(40, len(residuos)//2 - 1), ax=ax)
            plt.title("ACF - Residuos")
            plt.tight_layout()
            st.pyplot(fig_acf)
            plt.close(fig_acf)
        
        # Pruebas estadísticas
        st.markdown("### 🧪 Pruebas Estadísticas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Ljung-Box")
            try:
                lj = acorr_ljungbox(residuos, lags=[10], return_df=True)
                st.dataframe(lj.style.format({'lb_stat':'{:.4f}','lb_pvalue':'{:.4f}'}))
                if lj['lb_pvalue'].values[0] > 0.05:
                    st.success("✅ No autocorrelación")
                else:
                    st.warning("⚠️ Autocorrelación detectada")
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col2:
            st.markdown("#### Jarque-Bera")
            jb = jarque_bera(residuos)
            st.metric("Estadístico", f"{jb[0]:.4f}")
            st.metric("p-valor", f"{jb[1]:.4f}")
            if jb[1] > 0.05:
                st.success("✅ Normalidad")
            else:
                st.warning("⚠️ No normal")
        
        with col3:
            st.markdown("#### Anderson-Darling")
            andr = anderson(residuos)
            st.metric("Estadístico", f"{andr.statistic:.4f}")
            st.write("Valores críticos:")
            for i, (sig, crit) in enumerate(zip(andr.significance_level, andr.critical_values)):
                st.text(f"{sig}%: {crit:.4f}")
        
        # Interpretación automática
        st.markdown("### 💡 Interpretación")
        interpretacion = []
        
        # Ljung-Box
        try:
            if lj['lb_pvalue'].values[0] < 0.05:
                interpretacion.append("⚠️ **Autocorrelación**: Los residuos muestran autocorrelación (p < 0.05). Considera aumentar el orden del modelo.")
            else:
                interpretacion.append("✅ **Sin autocorrelación**: Los residuos no muestran autocorrelación significativa.")
        except:
            pass
        
        # Normalidad
        if jb[1] < 0.05:
            interpretacion.append("⚠️ **Normalidad**: Residuos no siguen distribución normal (p < 0.05). Esto puede afectar intervalos de confianza.")
        else:
            interpretacion.append("✅ **Normalidad**: Residuos compatibles con distribución normal.")
        
        # Media y varianza
        media_res = np.mean(residuos)
        var_res = np.var(residuos)
        interpretacion.append(f"📊 **Media residuos**: {media_res:.6f} (cercano a 0 es mejor)")
        interpretacion.append(f"📊 **Varianza residuos**: {var_res:.6f}")
        
        for item in interpretacion:
            st.markdown(item)

# ===================
# TAB: PREDICCIÓN
# ===================
with tabs[5]:
    st.markdown("## Predicción y Backtest")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        clase_pred = st.selectbox("🔍 Droga", lista_drogas, index=default_idx, key='pred_clase')
    with col2:
        freq_pred = st.selectbox("📅 Frecuencia", ['D', 'W', 'M'], index=2, key='pred_freq',
                                 format_func=lambda x: {'D':'Diaria', 'W':'Semanal', 'M':'Mensual'}[x])
    
    serie_pred = construir_serie(df, clase_pred, freq=freq_pred, fillna_method='zero')
    
    if serie_pred.empty:
        st.warning("⚠️ Serie vacía")
    else:
        # Configuración del backtest
        st.markdown("### ⚙️ Configuración del Backtest")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("% datos para test", 10, 50, 50, 5, key='test_size')
        with col2:
            usar_modelo_sesion = st.checkbox("Usar modelo guardado", value=True, 
                                            disabled='modelo_fit' not in st.session_state)
        
        if not usar_modelo_sesion or 'modelo_fit' not in st.session_state:
            st.info("⚙️ Configuración manual del modelo")
            col1, col2, col3 = st.columns(3)
            with col1:
                p_pred = st.number_input("p", 0, 5, 1, key='p_pred')
                d_pred = st.number_input("d", 0, 2, 1, key='d_pred')
                q_pred = st.number_input("q", 0, 5, 1, key='q_pred')
            with col2:
                P_pred = st.number_input("P", 0, 2, 0, key='P_pred')
                D_pred = st.number_input("D", 0, 1, 0, key='D_pred')
                Q_pred = st.number_input("Q", 0, 2, 0, key='Q_pred')
            with col3:
                s_pred = st.number_input("s", 0, 52, 0, key='s_pred', 
                                        help="0 para no estacional, 7 para semanal, 12 para mensual")
            
            order_pred = (int(p_pred), int(d_pred), int(q_pred))
            seasonal_pred = (int(P_pred), int(D_pred), int(Q_pred), int(s_pred)) if s_pred > 0 else (0,0,0,0)
        else:
            order_pred = st.session_state['orden_modelo']['order']
            seasonal_pred = st.session_state['orden_modelo']['seasonal_order']
            st.info(f"📋 Usando modelo guardado: ARIMA{order_pred} x {seasonal_pred}")
        
        if st.button("🚀 Ejecutar Backtest", type="primary"):
            with st.spinner("Ejecutando backtest..."):
                try:
                    # División train/test
                    n = len(serie_pred)
                    split = int(n * (1 - test_size/100))
                    
                    if split < 10:
                        st.error("❌ No hay suficientes datos para entrenamiento. Reduce el % de test.")
                        st.stop()
                    
                    train = serie_pred.iloc[:split]
                    test = serie_pred.iloc[split:]
                    
                    st.info(f"📊 Train: {len(train)} obs | Test: {len(test)} obs")
                    
                    # Ajustar modelo con más iteraciones para convergencia
                    model = SARIMAX(
                        train, 
                        order=order_pred, 
                        seasonal_order=seasonal_pred,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    
                    # Aumentar maxiter para mejor ajuste
                    fit = model.fit(disp=False, method='lbfgs', maxiter=100)
                    
                    # Predicción OUT-OF-SAMPLE (fuera de muestra)
                    fc = fit.get_forecast(steps=len(test))
                    yhat = fc.predicted_mean
                    conf_int = fc.conf_int()
                    
                    # Verificar que la predicción tenga variación
                    if yhat.std() == 0:
                        st.warning("⚠️ El modelo predice valores constantes. Intenta con otro orden ARIMA.")
                    
                    # Alinear índices
                    yhat.index = test.index
                    conf_int.index = test.index
                    
                    # Calcular métricas
                    metrics = calcular_metricas(test.values, yhat.values)
                    
                    st.success("✅ Backtest completado")
                    
                    # ADVERTENCIA si residuos no son normales
                    residuos_model = fit.resid
                    jb_test = jarque_bera(residuos_model)
                    if jb_test[1] < 0.05:
                        st.warning("⚠️ **ADVERTENCIA**: Los residuos del modelo NO son normales (Jarque-Bera p < 0.05). "
                                  "Esto puede afectar la confiabilidad de los intervalos de confianza. "
                                  "Considera transformar la serie (log, Box-Cox) o probar otro modelo.")
                    
                    # Mostrar información del ajuste
                    with st.expander("ℹ️ Información del modelo ajustado"):
                        st.write(f"**Convergencia**: {'✅ Sí' if fit.mle_retvals['converged'] else '❌ No'}")
                        st.write(f"**Iteraciones**: {fit.mle_retvals.get('iterations', 'N/A')}")
                        st.write(f"**Desv. estándar predicción**: {yhat.std():.2f}")
                        st.write(f"**Normalidad residuos (Jarque-Bera)**: p-valor = {jb_test[1]:.4f}")
                    
                    # Mostrar métricas
                    st.markdown("### 📊 Métricas de Evaluación")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("RMSE", f"{metrics['RMSE']:,.2f}")
                    col2.metric("MAE", f"{metrics['MAE']:,.2f}")
                    col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                    col4.metric("MSE", f"{metrics['MSE']:,.2f}")
                    col5.metric("Bias", f"{metrics['Bias']:,.2f}")
                    
                    # Métricas del modelo
                    col1, col2, col3 = st.columns(3)
                    col1.metric("AIC", f"{fit.aic:.2f}")
                    col2.metric("BIC", f"{fit.bic:.2f}")
                    col3.metric("Log-Likelihood", f"{fit.llf:.2f}")
                    
                    # Gráfica de predicción vs real
                    st.markdown("### 📈 Predicción vs Valores Reales")
                    
                    fig = go.Figure()
                    
                    # Serie de entrenamiento
                    fig.add_trace(go.Scatter(
                        x=train.index,
                        y=train.values,
                        name='Entrenamiento',
                        line=dict(color='#002D72', width=2),
                        mode='lines'
                    ))
                    
                    # Valores reales (test)
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=test.values,
                        name='Valores Reales',
                        line=dict(color='#28a745', width=2),
                        mode='lines+markers',
                        marker=dict(size=6)
                    ))
                    
                    # Predicción
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=yhat.values,
                        name='Predicción',
                        line=dict(color='#FDB813', width=2, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=6, symbol='x')
                    ))
                    
                    # Intervalo de confianza
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=conf_int.iloc[:, 1],
                        name='IC Superior',
                        line=dict(width=0),
                        mode='lines',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=test.index,
                        y=conf_int.iloc[:, 0],
                        name='Intervalo 95%',
                        fill='tonexty',
                        line=dict(width=0),
                        mode='lines',
                        fillcolor='rgba(253, 184, 19, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title=f"Backtest: {clase_pred} (Train: {len(train)} obs, Test: {len(test)} obs)",
                        xaxis_title="Fecha",
                        yaxis_title="Cantidad (kg)",
                        hovermode='x unified',
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Gráfica de errores
                    st.markdown("### 📉 Análisis de Errores")
                    errors = test.values - yhat.values
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_err = go.Figure()
                        fig_err.add_trace(go.Scatter(
                            x=test.index,
                            y=errors,
                            mode='lines+markers',
                            name='Errores',
                            line=dict(color='#dc3545')
                        ))
                        fig_err.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_err.update_layout(
                            title="Errores de Predicción",
                            xaxis_title="Fecha",
                            yaxis_title="Error (Real - Predicción)",
                            height=350
                        )
                        st.plotly_chart(fig_err, use_container_width=True)
                    
                    with col2:
                        fig_err_hist = go.Figure()
                        fig_err_hist.add_trace(go.Histogram(
                            x=errors,
                            nbinsx=30,
                            name='Distribución de errores',
                            marker_color='#dc3545'
                        ))
                        fig_err_hist.update_layout(
                            title="Distribución de Errores",
                            xaxis_title="Error",
                            yaxis_title="Frecuencia",
                            height=350
                        )
                        st.plotly_chart(fig_err_hist, use_container_width=True)
                    
                    # Guardar en sesión
                    st.session_state['backtest_results'] = {
                        'train': train,
                        'test': test,
                        'pred': yhat,
                        'conf_int': conf_int,
                        'metrics': metrics,
                        'model': fit
                    }
                    
                    # Tabla comparativa completa
                    st.markdown("### 📋 Tabla Comparativa Completa")
                    
                    # Calcular errores absolutos porcentuales
                    errors_pct = np.abs(errors / np.where(test.values == 0, 1e-6, test.values)) * 100
                    
                    df_comp_full = pd.DataFrame({
                        'Fecha': test.index,
                        'Real': test.values,
                        'Predicción': yhat.values,
                        'Error': errors,
                        'Error Abs': np.abs(errors),
                        'Error %': errors_pct,
                        'IC Inferior': conf_int.iloc[:, 0].values,
                        'IC Superior': conf_int.iloc[:, 1].values
                    })
                    
                    # Colorear filas según error
                    def color_error(val):
                        if abs(val) > 50:
                            return 'background-color: #ffcccc'
                        elif abs(val) > 30:
                            return 'background-color: #fff4cc'
                        else:
                            return 'background-color: #ccffcc'
                    
                    st.dataframe(
                        df_comp_full.style.format({
                            'Real': '{:,.2f}',
                            'Predicción': '{:,.2f}',
                            'Error': '{:,.2f}',
                            'Error Abs': '{:,.2f}',
                            'Error %': '{:.2f}%',
                            'IC Inferior': '{:,.2f}',
                            'IC Superior': '{:,.2f}'
                        }).applymap(color_error, subset=['Error %']),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Estadísticas de errores
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Error promedio", f"{np.mean(errors):.2f} kg")
                    col2.metric("Error máximo", f"{np.max(np.abs(errors)):.2f} kg")
                    col3.metric("% predicciones dentro IC", 
                               f"{np.sum((test.values >= conf_int.iloc[:, 0]) & (test.values <= conf_int.iloc[:, 1])) / len(test) * 100:.1f}%")
                    
                    # Recomendaciones
                    st.markdown("### 💡 Recomendaciones")
                    recomendaciones = []
                    
                    # Evaluar MAPE
                    if metrics['MAPE'] > 100:
                        recomendaciones.append("⚠️ **MAPE muy alto (>100%)**: La serie tiene valores muy cercanos a cero o alta volatilidad. "
                                             "Considera: (1) Usar frecuencia mayor (ej. mensual en vez de semanal), "
                                             "(2) Aplicar transformación logarítmica, (3) Usar otro modelo.")
                    elif metrics['MAPE'] > 50:
                        recomendaciones.append("⚠️ **MAPE alto (>50%)**: Ajuste moderado. "
                                             "Intenta ajustar los parámetros p, d, q o agregar componente estacional.")
                    else:
                        recomendaciones.append("✅ **MAPE aceptable**: El modelo tiene un ajuste razonable.")
                    
                    # Evaluar residuos
                    if jb_test[1] < 0.05:
                        recomendaciones.append("⚠️ **Residuos no normales**: Considera aplicar transformaciones a la serie original "
                                             "(Box-Cox, logarítmica) antes de modelar.")
                    
                    # Evaluar convergencia
                    if not fit.mle_retvals['converged']:
                        recomendaciones.append("❌ **Modelo no convergió**: El modelo puede no ser confiable. "
                                             "Intenta reducir el orden del modelo o cambiar parámetros.")
                    
                    # Evaluar predicción constante
                    if yhat.std() < test.std() * 0.1:
                        recomendaciones.append("⚠️ **Predicción muy plana**: El modelo predice valores poco variables. "
                                             "Esto puede indicar sobrediferenciación (d muy alto) o modelo muy simple. "
                                             "Intenta reducir d o aumentar p/q.")
                    
                    for rec in recomendaciones:
                        st.markdown(rec)
                    
                    # Sugerencias de parámetros
                    st.markdown("#### 🔧 Sugerencias de parámetros según frecuencia:")
                    if freq_pred == 'D':
                        st.info("**Frecuencia Diaria**: Prueba p=1-2, d=1, q=0-1, s=7 (semanal) o s=0")
                    elif freq_pred == 'W':
                        st.info("**Frecuencia Semanal**: Prueba p=1, d=1, q=1, s=0 (no estacional)")
                    else:
                        st.info("**Frecuencia Mensual**: Prueba p=1-2, d=1, q=1, s=12 (estacional anual)")
                    
                except Exception as e:
                    st.error(f"❌ Error en backtest: {str(e)}")
                    import traceback
                    with st.expander("Ver detalles del error"):
                        st.code(traceback.format_exc())

# ===================
# TAB: COMPARACIÓN
# ===================
with tabs[6]:
    st.markdown("## Comparación entre Drogas")
    
    freq_comp = st.selectbox("📅 Frecuencia", ['D', 'W', 'M'], index=2, key='comp_freq',
                             format_func=lambda x: {'D':'Diaria', 'W':'Semanal', 'M':'Mensual'}[x])
    
    # Construir panel de datos
    with st.spinner("Construyendo panel de datos..."):
        panel = df.groupby(['FECHA HECHO', 'CLASE BIEN'])['CANTIDAD'].sum().unstack(fill_value=0)
        panel = panel.resample(freq_comp).sum()
        panel = panel.loc[:, panel.sum(axis=0) > 0]  # Eliminar columnas vacías
    
    st.success(f"✅ Panel construido: {panel.shape[0]} periodos × {panel.shape[1]} drogas")
    
    # Matriz de correlaciones
    st.markdown("### 🔗 Matriz de Correlaciones")
    corr = panel.corr()
    
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlaciones entre drogas',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlaciones
    st.markdown("### 🔝 Top Correlaciones")
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append({
                'Droga 1': corr.columns[i],
                'Droga 2': corr.columns[j],
                'Correlación': corr.iloc[i, j]
            })
    
    df_corr = pd.DataFrame(corr_pairs).sort_values('Correlación', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Correlaciones más altas")
        st.dataframe(
            df_corr.head(10).style.format({'Correlación': '{:.3f}'}),
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Correlaciones más bajas")
        st.dataframe(
            df_corr.tail(10).style.format({'Correlación': '{:.3f}'}),
            use_container_width=True
        )
    
    # Series temporales comparativas
    st.markdown("### 📈 Series Temporales Comparativas")
    drogas_selec = st.multiselect(
        "Selecciona drogas para comparar",
        options=lista_drogas,
        default=lista_drogas[:min(3, len(lista_drogas))]
    )
    
    if drogas_selec:
        fig = go.Figure()
        for droga in drogas_selec:
            if droga in panel.columns:
                fig.add_trace(go.Scatter(
                    x=panel.index,
                    y=panel[droga],
                    name=droga,
                    mode='lines'
                ))
        
        fig.update_layout(
            title="Comparación temporal de incautaciones",
            xaxis_title="Fecha",
            yaxis_title="Cantidad (kg)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Modelo VAR
    st.markdown("### 🔀 Modelo VAR (Vector Autoregression)")
    
    if panel.shape[1] < 2:
        st.warning("⚠️ Se necesitan al menos 2 series para ajustar VAR")
    else:
        if st.button("🚀 Ajustar modelo VAR"):
            with st.spinner("Ajustando VAR..."):
                try:
                    model_var = VAR(panel)
                    lag_res = model_var.select_order(maxlags=8)
                    best_lag = int(lag_res.selected_orders['aic'])
                    
                    st.info(f"Mejor orden según AIC: {best_lag}")
                    
                    var_fit = model_var.fit(best_lag)
                    
                    st.success("✅ Modelo VAR ajustado")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Orden (lags)", best_lag)
                    col2.metric("AIC", f"{var_fit.aic:.2f}")
                    
                    with st.expander("📄 Ver resumen completo"):
                        st.text(var_fit.summary())
                    
                    st.session_state['var_fit'] = var_fit
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Sistema de Análisis de Series Temporales</strong></p>
        <p>Desarrollado con Streamlit | Análisis ARIMA de incautaciones de estupefacientes</p>
    </div>
    """,
    unsafe_allow_html=True
)