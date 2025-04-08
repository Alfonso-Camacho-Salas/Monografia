import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pycaret.classification import *
from PIL import Image

# Load model
pipeline = load_model(model_name="inversion")

# Sidebar navigation
def sidebar_navigation():
    st.sidebar.image("LOGOCBSM.png")
    st.sidebar.title("Navegación")
    return st.sidebar.radio("Ir a: ", ["Variables Financieras", "Modelo", "Sobre"])
    
# Main
def Variables_Financieras():
    st.title("Variables Financieras")
    st.write("En la siguiente pagina encontraras la definición de cada indicador financiero usado por el modelo")
    data = {
    'Indicador': ['BPA diluido', 'Recompra de acciones', 'Efecto de los cambios de divisas en el efectivo', 'Múltiplo de Valor Empresarial',
                 'Precio Valor Justo', 'Rendimiento por Ganancias', 'Días de Ventas Pendientes de cobro', 'Días de Pagos Pendientes',
                 'Crecimiento del Flujo de Caja Operativo', 'Crecimiento del Valor Contable por Acción'],
    'Definición': ['Beneficio por acción diluido', 'Compra de acciones propias', 'Efecto de los cambios de divisas en el efectivo', 'Múltiplo de valor empresarial',
                  'Precio justo de la acción', 'Rendimiento de las ganancias', 'Días promedio de cuentas por cobrar', 'Días promedio de cuentas por pagar',
                  'Crecimiento del flujo de caja operativo', 'Crecimiento del valor contable por acción'],
    'Cómo se mide': ['Beneficio neto atribuible a los accionistas comunes dividido por el número de acciones diluidas',
                    'Cantidad de acciones propias que la empresa compra',
                    'Impacto de las fluctuaciones cambiarias en la posición de efectivo',
                    'Relación entre el valor empresarial y el EBITDA',
                    'Valor intrínseco estimado de una acción',
                    'Beneficio por acción dividido por el precio de la acción',
                    'Número promedio de días que tarda una empresa en cobrar sus ventas a crédito',
                    'Número promedio de días que tarda una empresa en pagar a sus proveedores',
                    'Tasa de crecimiento anual del flujo de caja operativo',
                    'Tasa de crecimiento anual del valor contable por acción']
    }
    df = pd.DataFrame(data)

    # Crear la aplicación de Streamlit
    st.title("Tabla de Indicadores Financieros")
    

    # Mostrar la tabla con formato
    st.table(df.style.set_properties(**{'text-align': 'left'}).set_table_styles([
        {'selector': 'th',
        'props': [('font-size', '14px'),
                   ('text-align', 'left')]},
        {'selector': 'td',
        'props': [('font-size', '13px')]}
    ]))
# Home
def sobre():
    st.title("Sobre Nuestro Modelo de Machine Learning")
    st.subheader("¿Cómo funciona nuestro modelo?")
    
    # Imagen de la arquitectura del modelo (opcional)
    #image = Image.open('arquitectura_modelo.png')
   # st.image(image, caption='Arquitectura del modelo')

    # Explicación del proceso de ML
    st.write("""
    

        Nuestro modelo de Machine Learning ha sido diseñado para pronosticar el comportamiento de acciones de empresas individuales en la bolsa de valores, mediante el análisis de indicadores financieros obtenidos de los informes 10-K de las compañías que cotizan en el mercado. 

        **Proceso:**

        1. **Recopilación de datos:** 
            Comenzamos recopilando una gran cantidad de datos detro de los informes 10-K de las compañías que cotizan en el mercado. 

        2. **Preprocesamiento de datos:** 
            Los datos recolectados a menudo requieren un preprocesamiento para limpiarlos y prepararlos para el entrenamiento del modelo. Esto incluye tareas como:
            * Manejo de valores faltantes
            * Codificación de variables categóricas
            * Normalización de datos

        3. **Selección de características:**
            Identificamos las características más relevantes de los datos que contribuyen significativamente a la predicción. 

        4. **Entrenamiento del modelo:**
            Utilizamos un algoritmo de aprendizaje automático *RandomForestClassifier* para entrenar el modelo con los datos preprocesados. Durante el entrenamiento, el modelo aprende a identificar patrones en los datos y a realizar predicciones.

        5. **Evaluación del modelo:**
            Evaluamos el rendimiento del modelo utilizando métricas como la precisión, recuparacíon y F1. Esto nos permite medir la calidad de las predicciones del modelo.

        6. **Deployment:**
            Una vez que estamos satisfechos con el rendimiento del modelo, lo desplegamos en esta aplicación web para que pueda ser utilizado por los usuarios.
        """)

    # Imagen de un diagrama de flujo (opcional)
    #image = Image.open('diagrama_flujo.png')
    #st.image(image, caption='Diagrama de flujo del proceso de ML')


    st.write("""
    **Para más información, por favor contacta a:**
    Alfonso J Camacho Salas
    alfonsocamacho.salas0923@gmail.com
    """)
# Modelo
def modelo():
    st.title("Modelo - Stock Prediction")

    # Inputs for the variables (these should match the features your model expects)
    EPS_Diluted = st.number_input("BPA diluido", value=7.27)
    Issuance_buybacks_of_shares = st.number_input("Recompra de acciones", value=4000000.0)
    Effect_of_forex_changes_on_cash = st.number_input("Efecto de los cambios de divisas en el efectivo", value=26000000.0)
    enterpriseValueMultiple = st.number_input("Múltiplo de Valor Empresarial", value=11.12200212)
    priceFairValue = st.number_input("Precio Valor Justo", value=1.313306998)
    Earnings_Yield = st.number_input("Rendimiento por Ganancias", value=0.0884)
    Days_Sales_Outstanding = st.number_input("Días de Ventas Pendientes de cobro", value=29.9224)
    Days_Payables_Outstanding = st.number_input("Días de Pagos Pendientes", value=89.0539)
    Operating_Cash_Flow_growth = st.number_input("Crecimiento del Flujo de Caja Operativo", value=0.6561)
    Book_Value_per_Share_Growth = st.number_input("Crecimiento del Valor Contable por Acción", value=0.1187)

    # Collect inputs into a numpy array
    input_data = pd.DataFrame([
        {'EPS Diluted': EPS_Diluted,'Issuance buybacks of shares':Issuance_buybacks_of_shares,
         'Effect of forex changes on cash':Effect_of_forex_changes_on_cash, 'enterpriseValueMultiple':enterpriseValueMultiple,
         'priceFairValue': priceFairValue, 'Earnings Yield':Earnings_Yield, 'Days Sales Outstanding': Days_Sales_Outstanding,
         'Days Payables Outstanding':  Days_Payables_Outstanding,
          'Operating Cash Flow growth': Operating_Cash_Flow_growth,'Book Value per Share Growth': Book_Value_per_Share_Growth }
    ])

    # Predict button
    if st.button("Predict"):
        # Realizar predicción con PyCaret obteniendo probabilidades de ambas clases
        prediction = predict_model(pipeline, input_data, raw_score=True)

        # Extraer probabilidades (PyCaret nombra las columnas como Score_0 y Score_1)
        prob_1 = prediction["prediction_score_1"][0]  # Probabilidad de la clase 1

        # Aplicar manualmente el threshold de 0.35
        threshold = 0.33
        prediction_label = 1 if prob_1 >= threshold else 0

        # Mostrar resultados en Streamlit
        st.write("Recomendación para invertir:", prob_1)
        st.write("Marcador de predicción:", prediction_label)

        # Mensaje basado en la predicción
        if prediction_label == 1:
            st.success("El algoritmo detecta correlación positiva del crecimiento de la acción con los indicadores financieros, se recomienda invertir en esta empresa.")
        else:
            st.error("Con base en los datos proporcionados y el análisis del algoritmo, la empresa no demuestra unas finanzas para estimular el crecimiento de su acción en bolsa.")


# Main control for pages
def app():
    page = sidebar_navigation()

    if page == "Variables Financieras":
        Variables_Financieras()
    elif page == "Modelo":
        modelo()
    elif page == "Sobre":
        sobre()

if __name__ == "__main__":
    app()
