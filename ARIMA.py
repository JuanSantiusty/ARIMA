import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error



#Paso 0: Cargar los datos

# Cargar datos
datos = pd.read_csv('future-gc00-daily-prices.csv', parse_dates=['Date'])

# Eliminar comas y truncar decimales (todo después del punto)
datos['Close'] = datos['Close'].astype(str).str.replace(',', '').str.extract(r'^(\d+)', expand=False).astype(float)

while True:
    print("1- Grafica de los datos")
    print("2- Informacion de los datos")
    print("3- Continuar al siguiente paso")
    x= int(input("Ingrese una opcion:"))
    if x==1 :
        #datos.plot(y='Close',kind='hist',figsize=(10,5))
        plt.scatter(datos['Date'], datos['Close'], color='red', s=30)  # s=tamaño puntos
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.show()
    elif x==2:
        datos.info();
    elif x==3:
        break
    else:
        print("Opcion invalida")
    

datos = pd.read_csv('future-gc00-daily-prices.csv', index_col="Date")
datos['Close'] = datos['Close'].astype(str).str.replace(',', '').str.extract(r'^(\d+)', expand=False).astype(float)
"""
    #Estabilizar varianza

datos = np.log(datos);
datos.plot();
plt.show();

"""

bandera_ajuste = False
estacionaridad = ""
parametro_i = 0
while True:
    print("1- Grafica ACf y PACF")
    print("2- Prueba ADF")
    print("3- Ajustar estacionariedad")
    print("4- Continuar al siguiente paso")
    op= int(input("Ingrese una opcion:"))
    if op==1:
        acf_original = plot_acf(datos['Close'],lags=40)
        pacf_original = plot_pacf(datos['Close'],lags=40)
        plt.show()
    elif op==2:
        adf_prueba = adfuller(datos['Close'])
        print(f'p-valor: {adf_prueba[1]}')
        if adf_prueba[1]>0.05:
            bandera_ajuste=True
            estacionaridad = "No estacionario"
        else:
            estacionaridad = "Estacionario"
        print(f'Estacionariedad: {estacionaridad}')
    elif op==3:
        if bandera_ajuste:
            datos["Close"] = datos["Close"].diff().dropna()
            parametro_i = parametro_i+1
    elif op==4:
        break
    else:
        print("Opcion Invalida")
        


"""

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
"""
parametro_p = 0
parametro_q = 0
while True:
    print("1-Graficos ACF y PACF")
    print("2-Establecer parametros p y q del modelo arima")
    print("3-Crear modelo Arima")
    op=int(input("Ingrese una opcion:"))
    if op==1:
        acf_original = plot_acf(datos['Close'],lags=40)
        pacf_original = plot_pacf(datos['Close'],lags=40)
        plt.show()
    elif op==2:
        parametro_p=int(input("Ingrese parametro p:"))
        parametro_q=int(input("Ingrese parametro q:"))
    elif op==3:
        break
    else:
        print("Opcion invalida")




"""

#Paso 2: Establezer parametros p, i y q

    #Como realizamos 1 diferencia establezemos i en 1
i = 1
    #Analizando los graficos ACF y PACF de la serie se establecen los valores de p y q
p = 1
q = 0

"""
#Creacion modelo ARIMA
porcentaje = float(input("Ingrese porcentaje de los datos para los datos de entrenamiento:"))
datos_aux = datos['Close']
entrenamiento_tamano = int(len(datos_aux) * porcentaje)
datos_entrenamiento, datos_prueba = datos_aux[:entrenamiento_tamano], datos_aux[entrenamiento_tamano:]

modelo = ARIMA(datos_entrenamiento, order=(parametro_p,parametro_i,parametro_q))
modelo_fit = modelo.fit()

while True:
    print("1- Informacion modelo ARIMA")
    print("2- Grafica de residuos y densidad")
    print("3- Predecir")
    print("4- Valor Mape")
    print("5- Finalizar")
    op=int(input("Ingrese una opcion:"))
    if op==1:
        modelo_resumen=modelo_fit.summary()
        print(modelo_resumen)
    elif op==2:
        residuos = modelo_fit.resid
        fig, ax = plt.subplots(1,2)
        residuos.plot(title='Residuos', ax=ax[0])
        residuos.plot(title='Densidad', kind='kde',ax=ax[1])
        plt.show()
    elif op==3:
        forecast = modelo_fit.forecast(steps=len(datos_prueba))
        plt.figure(figsize=(15, 10))
        
        # Gráfico de dispersión para Train (azul)
        plt.scatter(datos_aux.index[:entrenamiento_tamano], datos_entrenamiento, label='Train', color='blue', alpha=0.6)
        
        # Gráfico de dispersión para Test (verde)
        plt.scatter(datos_aux.index[entrenamiento_tamano:], datos_prueba, label='Test', color='green', alpha=0.6)
        
        # Gráfico de dispersión para Forecast (rojo)
        plt.scatter(datos_aux.index[entrenamiento_tamano:], forecast, label='Forecast', color='red', alpha=0.6)
        
        plt.legend()
        plt.title('ARIMA Forecast vs Actual (Scatter Plot)')
        plt.show()
    elif op==4:
        prediccion_prueba = modelo_fit.forecast(len(datos_prueba))        
        mape = mean_absolute_percentage_error(datos_prueba, prediccion_prueba)
        print(f'Valor del MAPE: {mape}')
    elif op==5:
        break
    else:
        print("Opcion Invalida")



"""

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

"""