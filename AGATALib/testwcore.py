###Importamos las librerias necesarias:
##Importamos Pandas para trabajar con DataFrames
import pandas as pd

##Importamos la lireria itertools para tener herramientas de iteracion sobre diferentes conjuntos
import itertools

##Importamos Numpy para incluir mas funciones matematicas
import numpy as np

##Importamos la funcion rankdata de scipy para poder asignar rangos en el test de wilcoxon
from scipy.stats import rankdata

##Importamos MatPlotLib para realizar representaciones
import matplotlib.pyplot as plt
from matplotlib import gridspec #Es un paquete de matplotlib que permite realizar figuras de varias subfiguras

##Importamos la funcion display de la libreria IPython.display, que permite mostrar contenido de manera mas enriquecida
from IPython.display import display

##Importamos el modulo time para poder medir el tiempo de ejecución del codigo
import time

##Immportamos el generador de números aleatorios
import random

##---------------------------------Test de Wilcoxon-------------------------------------##
##Funcion para generar las posibles combinacinoes de pares de pulsos sin repeticion a partir de un Dataframe con una combinacion de pulsos.
#Va a devolver la lista de pares de indices o la lista de indices y el Dataframe combinado.
def paired_pulses(df, return_df = False):
    #Calculamos las posibles combinaciones de pulsos mediante los indices
    paired_index = list(itertools.combinations(df.index, 2))

    #La funcion devuelve o la lista de indices, o la lista de indices y el dataframe con las combinaciones de pares de pulsos, 
    #dependiendo de la eleccion que se haya elegido al llamar a la función.
    
    if return_df == False:
        return paired_index #Devolvemos solo los pares de indices
    
    if return_df == True:
        #Generamos un nuevo Dataframe que contenga las posibles combinaciones de pares de pulsos
        #Preparamos una lista donde vamos a almacenar cada par de pulsos concatenados
        new_rows = []

        #Iteramos sobre las posibles combinaciones de pares de pulsos (pares de indices)
        for index1, index2 in paired_index:
            #Extraemos los pulsos asociados a cada uno de los indices
            pulse1 = df.loc[index1].values
            pulse2 = df.loc[index2].values

            #Concatenamos los valores de ambos pulsos en una sola fila
            concatenated_pulses = [index1, index2] + list(np.concatenate([pulse1, pulse2]))

            #Aregamos el pulso concatenado a la lista de filas de las combinaciones de pulsos
            new_rows.append(concatenated_pulses)

        #Convertimos en un Dataframe la lista de pares de pulsos generada
        columns = ['id_1', 'id_2'] + list(range(df.shape[1] * 2))
        combined_df = pd.DataFrame(new_rows, columns=columns)

        #Devolvemos el Dataframe generado y la lista de indices
        return paired_index, combined_df
    
##Creamos una funcion que recibe dos pulsos como entrada y los alinea el 10% de la altura maxima ("down") o al 90% de la altura máxima ("up") segun se indique en el argumento de la funcion.
#Devuelve como salida los pulsos alineados.
#Tener en cuenta que los pulsos deben estar normalizados a [0,1]
def alinear(pulso1, pulso2):
    #Definimos los valores para los cuales se alcanza el 90% y el 10% del máximo. Que al estar normalizados en [0,1] son 0.9 y 0.1 respectivamente.
    T90 = 0.9 #90% de la altura del pulso
    T10 = 0.1 #10% de la altura del pulso

    #Calculamos la posicion en la que el pulso 1 alcanza el 90% en subida
    for i in pulso1.index.astype(int):
        if pulso1.iloc[i] > T90:
            i90_1 = i - 1
            break
        else: 
            continue
    
    #Calculamos la posicion en la que el pulso 1 alcanza el 10% en bajada
    for i in pulso1.index.astype(int):
        if pulso1.iloc[len(pulso2.index.astype(int)) - i] < T10:
            i10_1 = len(pulso2.index.astype(int)) - i + 1
            break
        else:
            continue
    
    #Calculamos la posicion en la que el pulso 2 alcanza el 90% en subida
    for i in pulso2.index.astype(int):
        if pulso2.iloc[i] > T90:
            i90_2 = i - 1
            break
        else: 
            continue

    #Calculamos la posicion en la que el pulso 2 alcanza el 10% en bajada
    for i in pulso2.index.astype(int):
        if pulso2.iloc[len(pulso2.index.astype(int)) - i] < T10:
            i10_2 = len(pulso2.index.astype(int)) - i + 1
            break
        else:
            continue
    
    #Calculamos el ancho de la rampa para el pulso 1 y el pulso 2 y nos quedamos con el máximo entre los dos
    long = min([i90_1 - i10_1, i90_2 - i10_2])

    #Alineamos los pulsos con la posicion al 10%, alineacion "down"
    pulso1_T10 = pulso1.iloc[i10_1 : i10_1 + long]
    pulso1_T10.index = range(long)
    pulso2_T10 = pulso2.iloc[i10_2 : i10_2 + long]
    pulso2_T10.index = range(long)

    #Alineamos los pulsos con la posicion al 90%, alineacion "up"
    pulso1_T90 = pulso1.iloc[i90_1 - long : i90_1]
    pulso1_T90.index = range(long)
    pulso2_T90 = pulso2.iloc[i90_2 - long : i90_2]
    pulso2_T90.index = range(long)

    #Devolvemos los pulsos alineados en ambas posiciones
    return pulso1_T10, pulso2_T10, pulso1_T90, pulso2_T90

##Definimos una funcion que aplica el test de wilcoxon a un par de pulsos. Devuelve como resultado el valor minimo entre R+ y R-, junto con el número de diferencias no nulas, para poder hacer el contraste
def test_wilcoxon(pulso1, pulso2):
    #Calculamos el vector de diferencias entre ambos pulsos
    dif = pulso1 - pulso2

    #Establecemos el umbral a partir del cual la diferencia se considera nula
    #Y comprobamos la condicion en los elementos de la lista: 0 si estan por debajo del embrul y si estan por encima se mantiene el mismo número
    MinT = 0.03 #Este valor umbral habrá que comprobar cual es el más apropiado
    for i, num in enumerate(dif):
        if abs(num) < MinT:
            dif[i] = 0
        else: continue

    #Eliminamos las diferencias que no son cero para hacer el test y guardamos el número de elementos nulos para corregir posteriormente R+ y R-
    nonzero_dif = dif[dif != 0]
    count_ceros = len(dif[dif == 0])

    #Asignamos rangos a las diferencias absolutas entre los pulsos.
    #La funcion rankdata de scipy.stats incluye automaticamente el tratamiento de los empates
    abs_diffs = np.abs(nonzero_dif)
    ranks = rankdata(abs_diffs)

    #Sumamos los rangos obtenidos para las diferencias positivas y negativas por separado para obtener R+ y R-
    R_pos =  np.sum(ranks[nonzero_dif > 0]) + count_ceros*(count_ceros+1)/4
    R_neg =  np.sum(ranks[nonzero_dif < 0]) + count_ceros*(count_ceros+1)/4
    #Obtenemos el valor minimo entre R+ y R-
    R = min(R_pos, R_neg)
    #Devolvemos dicho valor minimo como resultado del test y el número de diferencias no nulas
    return R, len(nonzero_dif)

##Definimos una funcion que lea un par de pulsos concatenados y les aplica el test de Wilcoxon. Como resultado devuleve 1 si los pulsos son estadisticamente iguales y 0 si no
def compare_pulses(row):
    #Definimos el valor de z que queremos emplear para la significación estadística
    z = 1.9599
    #Separamos la fila del dataframe en los dos pulsos que queremos comparar
    pulso1 = row[2:102]
    pulso2 = row[-100:]
    #Cambiamos los indices de las series de los pulsos para que vayan de 1 a 100
    pulso1.index = range(1, 101)
    pulso2.index = range(1, 101)
    #Alineamos los pulsos al 10% y al 90% de la altura máxima
    pulso1_T10, pulso2_T10, pulso1_T90, pulso2_T90 = alinear(pulso1, pulso2)
    #Aplicamos el test de Wilcoxon para el par de pulsos alineados al 10% de la altura máxima
    R_T10, n = test_wilcoxon(pulso1_T10, pulso2_T10)
    #Calculamos el valor del estadístico de contraste para el número de elementos que tengan los pulsos
    R_z = n*(n+1)/4 - z*np.sqrt(n*(n+1)*(n+2)/24)
    #Comprobamos si los pulsos son estadisticamente iguales para la alineacion T_10. Si lo son se comprueba la alineacion a T90 y si no se concluye que son estadisticamente diferentes
    if R_T10 > R_z:
        R_T90, n = test_wilcoxon(pulso1_T90, pulso2_T90)
        if R_T90 > R_z:
            igual = 1
        else:
            igual = 0
    else: 
        igual = 0

    return igual