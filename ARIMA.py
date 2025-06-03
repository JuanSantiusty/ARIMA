import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error



#Paso 0: Cargar los datos

    #Por el momento se usara los datos de website_data.cvs
datos = pd.read_csv('website_data.csv');
datos.info();

datos.plot();

    #Estabilizar varianza

datos = np.log(datos);
datos.plot();
plt.show();

    #Separar en conjunto de entrenamiento y conjunto de pruebas para la prediccion de 30 unidades de tiempo
msk = (datos.index < len(datos)-30)
datos_entrenamiento = datos[msk].copy()
datos_prueba = datos[~msk].copy()

#Paso 1: Comprobar la estacionaridad de las series de tiempo

    #Graficos ACF PACF
acf_original = plot_acf(datos_entrenamiento)
pacf_original = plot_pacf(datos_entrenamiento)
plt.show();
    #Prueba ADF
adf_prueba = adfuller(datos_entrenamiento)
print(f'p-valor: {adf_prueba[1]}')


    #Ajustar si se requiere, por el momento solo se estaviliza la media
datos_entrenamiento_media = datos_entrenamiento.diff().dropna()
datos_entrenamiento_media.plot()
plt.show()

    #Mostrar graficos ACF y PACF de la nueva serie
acf_media= plot_acf(datos_entrenamiento_media)
pacf_media= plot_pacf(datos_entrenamiento_media)
plt.show()

#Paso 2: Establezer parametros p, i y q

    #Como realizamos 1 diferencia establezemos i en 1
i = 1
    #Analizando los graficos ACF y PACF de la serie se establecen los valores de p y q
p = 1
q = 0

#Paso 3: Crear modelo arima

modelo = ARIMA(datos_entrenamiento, order=(p,i,q))
modelo_fit = modelo.fit()
print(modelo_fit.summary())

#Modelo ARIMA Automatico


#Paso 4: Realizar predicciones

    #Evaluar modelo
residuos = modelo_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuos.plot(title='Residuos', ax=ax[0])
residuos.plot(title='Densidad', kind='kde',ax=ax[1])
plt.show()

acf_res = plot_acf(residuos)
pacf_res = plot_pacf(residuos)
plt.show()

    #Prediccion  comparando con valores reales
prediccion_prueba = modelo_fit.forecast(len(datos_prueba))
datos['Manual'] = [None]*len(datos_entrenamiento)+list(prediccion_prueba)
datos.plot()
plt.show()

#Mostrar el mape
mape = mean_absolute_percentage_error(datos_prueba, prediccion_prueba)
print(mape)