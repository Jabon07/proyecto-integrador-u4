
import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from scipy import stats

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================

st.set_page_config(
    page_title="Optimización Agrícola",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Optimización Inferencial del Rendimiento Agrícola")
st.subheader("Eficiencia de Riego en Cultivos de Invernadero")

st.markdown("""
Dashboard de análisis agroindustrial para estudiar la relación entre
el volumen de riego, el fertilizante aplicado y el rendimiento de
las parcelas, así como evaluar un nuevo bioestimulante.
""")

# ==========================================
# 2. GENERACIÓN DE DATOS CON DUCKDB
# ==========================================

@st.cache_data
def generar_datos():

    conexion = duckdb.connect(database=":memory:")

    # Semilla determinista solicitada en el proyecto
    conexion.execute("SELECT setseed(0.42)")

    conexion.execute("""
        CREATE TABLE telemetria_agroindustrial AS

        WITH datos_base AS (
            SELECT
                i AS parcela_id,

                ROUND(
                    100 + random() * 200,
                    2
                ) AS volumen_riego,

                ROUND(
                    50 + random() * 100,
                    2
                ) AS fertilizante_kg,

                CASE
                    WHEN random() < 0.50
                    THEN 'Tratamiento_A'
                    ELSE 'Tratamiento_B'
                END AS tipo_tratamiento,

                ROUND(
                    18 + random() * 17,
                    2
                ) AS temperatura,

                -0.8 + random() * 1.6 AS ruido

            FROM generate_series(1, 100) AS t(i)
        )

        SELECT
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            temperatura,

            ROUND(
                5.0
                + (0.035 * volumen_riego)
                + (0.020 * fertilizante_kg)
                + CASE
                    WHEN tipo_tratamiento = 'Tratamiento_B'
                    THEN 1.8
                    ELSE 0.0
                  END
                + ruido,
                2
            ) AS rendimiento_ton

        FROM datos_base
        ORDER BY parcela_id
    """)

    # Extracción SQL hacia Pandas
    df = conexion.execute("""
        SELECT *
        FROM telemetria_agroindustrial
        ORDER BY parcela_id
    """).df()

    conexion.close()

    return df


df_original = generar_datos()

# ==========================================
# 3. FILTROS INTERACTIVOS
# ==========================================

st.sidebar.header("Filtros de análisis")

tratamientos = st.sidebar.multiselect(
    "Seleccionar tratamiento:",
    ["Tratamiento_A", "Tratamiento_B"],
    default=["Tratamiento_A", "Tratamiento_B"]
)

datos = df_original[
    df_original["tipo_tratamiento"].isin(tratamientos)
].copy()

if datos.empty:
    st.warning("Selecciona al menos un tratamiento.")
    st.stop()

# ==========================================
# 4. INDICADORES GENERALES
# ==========================================

st.markdown("---")
st.header("Indicadores del cultivo")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Parcelas analizadas",
    len(datos)
)

col2.metric(
    "Rendimiento promedio",
    f"{datos['rendimiento_ton'].mean():.2f} ton/ha"
)

col3.metric(
    "Riego promedio",
    f"{datos['volumen_riego'].mean():.2f} m³/ha"
)

col4.metric(
    "Fertilizante promedio",
    f"{datos['fertilizante_kg'].mean():.2f} kg/ha"
)

# ==========================================
# 5. REGRESIÓN LINEAL MÚLTIPLE
# ==========================================

X = datos[[
    "volumen_riego",
    "fertilizante_kg"
]]

X = sm.add_constant(X)

Y = datos["rendimiento_ton"]

modelo = sm.OLS(Y, X).fit()

predicciones = modelo.predict(X)

residuos = Y - predicciones

rmse = np.sqrt(np.mean(residuos ** 2))

st.markdown("---")
st.header("Análisis de regresión y correlación")

col1, col2, col3 = st.columns(3)

col1.metric(
    "R²",
    f"{modelo.rsquared:.4f}"
)

col2.metric(
    "R² ajustado",
    f"{modelo.rsquared_adj:.4f}"
)

col3.metric(
    "RMSE",
    f"{rmse:.4f}"
)

with st.expander("Consultar resultados completos del modelo OLS"):
    st.write(modelo.summary().as_text())

    st.write("Ecuación de regresión:")

    st.latex(
        rf"\hat{{Y}} = {modelo.params.iloc[0]:.3f}"
        rf" + {modelo.params.iloc[1]:.4f}X_1"
        rf" + {modelo.params.iloc[2]:.4f}X_2"
    )

# ==========================================
# 6. GRÁFICA 1: MAPA DE CORRELACIÓN
# ==========================================

st.markdown("---")
st.subheader("1. Mapa de calor de correlación de Pearson")

variables = datos[[
    "volumen_riego",
    "fertilizante_kg",
    "temperatura",
    "rendimiento_ton"
]]

correlacion = variables.corr(method="pearson")

fig1 = px.imshow(
    correlacion,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    labels={"color": "Correlación"}
)

fig1.update_layout(
    title="Matriz de correlación entre variables agrícolas",
    height=500
)

st.plotly_chart(fig1, use_container_width=True)

st.caption(
    "La temperatura se incorpora como variable ambiental "
    "complementaria para esta visualización."
)

# ==========================================
# 7. GRÁFICA 2: REGRESIÓN OLS
# ==========================================

st.markdown("---")
st.subheader("2. Dispersión con línea de regresión OLS")

X_simple = sm.add_constant(
    datos[["volumen_riego"]]
)

modelo_simple = sm.OLS(
    datos["rendimiento_ton"],
    X_simple
).fit()

x_linea = np.linspace(
    datos["volumen_riego"].min(),
    datos["volumen_riego"].max(),
    100
)

X_pred = sm.add_constant(
    pd.DataFrame({"volumen_riego": x_linea}),
    has_constant="add"
)

pred = modelo_simple.get_prediction(X_pred).summary_frame()

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=datos["volumen_riego"],
    y=datos["rendimiento_ton"],
    mode="markers",
    name="Parcelas observadas",
    marker=dict(opacity=0.65)
))

fig2.add_trace(go.Scatter(
    x=x_linea,
    y=pred["mean"],
    mode="lines",
    name="Regresión OLS",
    line=dict(width=3)
))

fig2.add_trace(go.Scatter(
    x=np.concatenate([
        x_linea,
        x_linea[::-1]
    ]),
    y=np.concatenate([
        pred["mean_ci_upper"].to_numpy(),
        pred["mean_ci_lower"].to_numpy()[::-1]
    ]),
    fill="toself",
    fillcolor="rgba(31,119,180,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Intervalo de confianza 95%"
))

fig2.update_layout(
    title="Relación entre volumen de riego y rendimiento",
    xaxis_title="Volumen de riego (m³/ha)",
    yaxis_title="Rendimiento (ton/ha)",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 8. GRÁFICA 3: BOXPLOT
# ==========================================

st.markdown("---")
st.subheader("3. Diagrama de cajas comparativo")

fig3 = px.box(
    datos,
    x="tipo_tratamiento",
    y="rendimiento_ton",
    color="tipo_tratamiento",
    points="all",
    title="Comparación del rendimiento entre tratamientos",
    labels={
        "tipo_tratamiento": "Tipo de tratamiento",
        "rendimiento_ton": "Rendimiento (ton/ha)"
    }
)

fig3.update_layout(height=500)

st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# 9. GRÁFICA 4: HISTOGRAMA DE RESIDUOS
# ==========================================

st.markdown("---")
st.subheader("4. Histograma de residuos del modelo")

fig4 = go.Figure()

fig4.add_trace(go.Histogram(
    x=residuos,
    histnorm="probability density",
    nbinsx=15,
    name="Residuos",
    opacity=0.65
))

if len(residuos) > 2 and residuos.std() > 0:

    x_norm = np.linspace(
        residuos.min(),
        residuos.max(),
        150
    )

    y_norm = stats.norm.pdf(
        x_norm,
        residuos.mean(),
        residuos.std(ddof=1)
    )

    fig4.add_trace(go.Scatter(
        x=x_norm,
        y=y_norm,
        mode="lines",
        name="Curva normal",
        line=dict(width=3)
    ))

fig4.add_vline(
    x=0,
    line_dash="dash",
    annotation_text="Residuo cero"
)

fig4.update_layout(
    title="Distribución de errores del modelo",
    xaxis_title="Residuo",
    yaxis_title="Densidad",
    height=500
)

st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# 10. INFERENCIA ESTADÍSTICA
# ==========================================

st.markdown("---")
st.header("Análisis inferencial de tratamientos")

# Se utilizan los grupos completos para la prueba,
# independientemente del filtro visual del dashboard.

grupo_A = df_original.loc[
    df_original["tipo_tratamiento"] == "Tratamiento_A",
    "rendimiento_ton"
]

grupo_B = df_original.loc[
    df_original["tipo_tratamiento"] == "Tratamiento_B",
    "rendimiento_ton"
]

# Hipótesis:
# H0: mu_B - mu_A <= 0
# H1: mu_B - mu_A > 0

prueba = stats.ttest_ind(
    grupo_B,
    grupo_A,
    equal_var=False,
    alternative="greater"
)

diferencia = grupo_B.mean() - grupo_A.mean()

# Intervalo de confianza bilateral del 95%
n_A = len(grupo_A)
n_B = len(grupo_B)

var_A = grupo_A.var(ddof=1)
var_B = grupo_B.var(ddof=1)

error = np.sqrt(
    var_A / n_A + var_B / n_B
)

grados = (
    (var_A / n_A + var_B / n_B) ** 2
    /
    (
        (var_A / n_A) ** 2 / (n_A - 1)
        +
        (var_B / n_B) ** 2 / (n_B - 1)
    )
)

t_critico = stats.t.ppf(
    0.975,
    grados
)

ic_inferior = diferencia - t_critico * error
ic_superior = diferencia + t_critico * error

col1, col2, col3 = st.columns(3)

col1.metric(
    "Media Tratamiento A",
    f"{grupo_A.mean():.2f} ton/ha"
)

col2.metric(
    "Media Tratamiento B",
    f"{grupo_B.mean():.2f} ton/ha"
)

col3.metric(
    "Diferencia B - A",
    f"{diferencia:.2f} ton/ha"
)

st.write("### Formulación de hipótesis")

st.latex(
    r"H_0: \mu_B-\mu_A \leq 0"
)

st.latex(
    r"H_1: \mu_B-\mu_A > 0"
)

st.write("### Resultados de la prueba t-Student")

st.write(f"Estadístico t: **{prueba.statistic:.4f}**")

st.write(f"Valor p unilateral: **{prueba.pvalue:.6f}**")

st.write(
    f"Intervalo de confianza del 95%: "
    f"**[{ic_inferior:.4f}, {ic_superior:.4f}] ton/ha**"
)

if prueba.pvalue < 0.05:

    st.success(
        "Se rechaza H₀. Existe evidencia estadística "
        "para afirmar que el Tratamiento B incrementa "
        "el rendimiento promedio respecto al Tratamiento A."
    )

else:

    st.warning(
        "No se rechaza H₀. No existe evidencia estadística "
        "suficiente para afirmar que el Tratamiento B "
        "incrementa el rendimiento promedio."
    )

# ==========================================
# 11. TABLA DE TELEMETRÍA
# ==========================================

st.markdown("---")
st.header("Tabla de telemetría agroindustrial")

st.dataframe(
    datos,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="Descargar datos en CSV",
    data=datos.to_csv(index=False).encode("utf-8"),
    file_name="telemetria_agroindustrial.csv",
    mime="text/csv"
)

st.markdown("---")

st.caption(
    "Proyecto Integrador U4 | Análisis y Exploración de Datos | "
    "Optimización Inferencial del Rendimiento Agrícola"
)
