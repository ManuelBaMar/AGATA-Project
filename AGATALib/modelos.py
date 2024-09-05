###Importamos las librerias necesarias:
##TensorFlow, Keras a partir de TensorFlow; y de Keras además importamos el comando para crear capas
import tensorflow as tf #Importamos TensorFlow
#import tensorflow_hub as hub #Importamos esta libreria que contiene modelos preentrenados para poder importarlos
from tensorflow import keras #Importamos keras desde TensorFlow
from tensorflow.keras import layers #Importamos la funcion para crear capas de redes neuronales desde keras
from tensorflow.keras.models import Model #Importamis la clase Model de tensorflow, lo que permite crear modelos de DL como clases, con arquitecturas más complejas que usar Sequential
from tensorflow.keras.callbacks import EarlyStopping #Importamos la funcion para realizar el earlystopping desde la libreria de callbacks de keras
from tensorflow.keras.models import load_model #Cargamos la funcion para poder importar modelos de machine learning
from tensorflow.keras.utils import plot_model #Cargamos la funcion para poder realizar representaciones de la arquitectura de modelos
from keras.saving import register_keras_serializable #Funcion de queras necesaria para guardar modelos que se hayan definidos como clases

##Importamos keras tuning para poder optimizar los hiperparametros del modelo
import keras_tuner as kt

##-----------------------------------Modelo basado en Liverpool----------------------------------------##

##Pasamos a definir el modelo de autoencoder que vamos a emplear. Lo creamos como una clase de keras y heredando la clase padre Model
#Buscamos que el encoder sea capaz de reproducir las señales con las que le entrenamos. Tanto el encoder como el decoder va a estar compuesto por capas densas
#Este primer modelo es una adaptacion a un solo segmento de la arquitectura propuesta en la tesis de Liverpool.
@register_keras_serializable() #Necesario añadirlo justo antes de definir el modelo como una clase para poder guardarlo correctamente una vez entrenado.
class Autoencoder_V1(Model):
    #Creamos la funcion que inicializa el modelo de autoencoder
    #Como argumento la clase toma un diccionario con los hiperparametros del modelo
    def __init__(self, hyper_params,**kwargs):
        #Inicializamos la clase padre de lal que estamos heredando propiedades para que el modelo funcione correctaente
        super(Autoencoder_V1, self).__init__(**kwargs)
        #Definimos las caracteristicas de las diferentes capas como hiperparametros ajustable del modelo
        self.hyper_params = hyper_params #Definimos los hiperparametros como un a variable de clase
        self.filters_cv = self.hyper_params["filters_cv"] #Filtros de la capa convolucional
        self.units_encoder = self.hyper_params["units_encoder"] #Neuronas de la capa densa en el encoder
        self.units_decoder = self.hyper_params["units_decoder"] #Neuronas de la capa densa en el decoder
        self.latem_dimension = self.hyper_params["latem_dimension"] #Dimension del espacio latente
        self.l1_penalty = self.hyper_params["l1_penalty"] #Factor de penalizacion lineal a las capas
        self.l2_penalty = self.hyper_params["l2_penalty"] #Factor de penalizacion cuadrático a las capas

        #Definimos la parte del encoder (codificación de la señal al espacio de dimensión reducida), como una serie de capas densas con funcion de activacion ReLU
        self.encoder = keras.Sequential([
            #Capa que define la dimension de entrada
            layers.Input(shape = [100, 1]),

            #Bloque convolucional para la deteccion de caracteristicas en la señal
            layers.Conv1D(filters = self.filters_cv, kernel_size = 3, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.BatchNormalization(), #Normalizacion
            layers.Activation("relu"),
            layers.MaxPool1D(),

            #Bloque de capas densas para la reduccion de la dimensión
            layers.Flatten(), #Aplanamos a una dimension la capa convolucional
            #1º capa densa
            layers.Dense(units = self.units_encoder, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.BatchNormalization(), #Normalizacion
            layers.Activation("relu"),
            #2º capa densa
            layers.Dense(units = self.latem_dimension, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.BatchNormalization(), #Normalizacion
            layers.Activation("relu"),
        ])

        #Definimos la parte del decoder (reconstrucción de la señal), como una serie de capas densas con función de activación ReLU
        self.decoder = keras.Sequential([
            #Capa que define la dimension de netrada del espacio latente
            layers.Input(shape = [self.latem_dimension]),

            #Bloque de capas densas para la reconstrucción de la señal
            layers.Dense(units = self.units_decoder, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.BatchNormalization(), #Normalizacion
            layers.Activation("relu"),

            #Capa de salida con función de activación sigmoid dado los datos de entrada estan acotados en [0,1]
            layers.Dense(units = 100, activation = "sigmoid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))
        ])

    #Definimos como se va acomportar el modelo cuando reciba datos de entrada
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
    #Modificamos las funciones de get_config() y from_config() para que la serializacion del modelo al guardarlo se haga correctamente
    def get_config(self):
        config = super(Autoencoder_V1, self).get_config()
        config.update({
            'hyper_params': self.hyper_params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        hyper_params = config.pop('hyper_params')
        return cls(hyper_params, **config)
    
##Una vez definido el modelo de autoencoder debemos definir el hipermodelo asociado a dicho modelo para poder ajustar los hiperparametros que hemos dejado libres en la arquitectura del modelo.
#Para ello instanciamos la clase que define el autoencoder dentro de una nueva clase kt.HyperModel que vamos a definir para poder optimizar los hiperparametros
class AE_HyperModel(kt.HyperModel):
    #En primer lugar inicializamos la calse, donde se va a definir el diccionario que contiene los hiperparametros que se quieren optimizar
    def __init__(self):
        self.hyper_params = {
            #Hiperparametros asociados a la arquitectura del modelo
            "filters_cv" : lambda hp: hp.Int("filters_cv", min_value = 2, max_value = 10, step = 1), #Número de filtros en la capa convolucional
            "units_encoder": lambda hp: hp.Int("units_encoder", min_value = 80, max_value = 200, step = 10), #Número de neuronas en la capa denda intermedia del encoder
            "units_decoder" : lambda hp: hp.Int("units_decoder",min_value = 20, max_value = 80, step = 10), #Número de neuronas en la capa densa intermedia del decoder
            "latem_dimension" : lambda hp: hp.Int("latem_dimension", min_value = 1, max_value = 10, step = 1), #Dimensión del espacio latente del decoder

            #Pesos de las penalizaciones aplicadas al modelo. Combinacion de penalizacion lineal y cuadrática
            "l1_penalty" : lambda hp: hp.Choice("l1_penalty", [1e-4, 1e-5, 1e-6, 1e-7]),
            "l2_penalty" : lambda hp: hp.Choice("l2_penalty", [1e-4, 1e-5, 1e-6, 1e-7]),

            #Hiperparametros del compilador del modelo
            "learning_rate" : lambda hp: hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]), #Ritmo de aprendizaje

            #Hiperparametros del entrenamiento
            "batch_size" : lambda hp: hp.Int("batch_size", min_value = 50, max_value = 300, step = 50),
            "min_delta" : lambda hp: hp.Choice("min_delta", [0.01, 0.005, 0.001]), #Criterio de convergencia del early_stopping
            "patience" : lambda hp: hp.Int("patience", min_value = 50, max_value = 100, step = 10), #Número minimo de iteraciones del early_stopping
        } 
    
    #Creamos el metodo build (constructor del hipermodelo), que va a definir el modelo teniendo en cuenta los diferentes hiperparametros
    def build(self, hp):
        #Creamos el diccionario de hiperparametros con los valores seleccionados en el constructor
        hyperparameters = {key: func(hp) for key, func in self.hyper_params.items()} #Para cada par (key, función que define cada hiperparametro correspondiente) se devuelve el par (key, valor elegido de los posibles del hiperparametro para probar)

        #Generamos el modelo añadiendo como argumentos los hiperparametros definidos en el constructor
        autoencoder = Autoencoder_V1(hyperparameters)

        #Compilamos el modelo teniendo en cuenta los hiperparametros
        autoencoder.compile(
            optimizer = keras.optimizers.Adam(learning_rate = hyperparameters["learning_rate"]), #Valor por defecto del learning_rate
            loss = keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.CosineSimilarity()],
        )
        return autoencoder

    #Sobreescribimos el metodo de entrenamiento para poder añadir ciertos hiperparametros
    def fit(self, hp, model, *args, **kwargs):
        kwargs['batch_size'] = self.hyper_params["batch_size"](hp)
        kwargs['callbacks'] = [
            EarlyStopping(
                min_delta=self.hyper_params["min_delta"](hp),
                patience=self.hyper_params["patience"](hp),
                restore_best_weights=True
            )
        ]
        return model.fit(*args, **kwargs)

##-----------------------------------Primer modelo convolucional---------------------------------------##

##Definimos un modelo de Autoencoder convolucional para tratar de reducir las señales.
#Este modelo trata de comprimir la señal, pero no extraé parametros que caractericen la señal.
@register_keras_serializable()
class Convolutional_AE1(Model):
    ##Creamos la funcion que inicializa el modelo de autoencoder convolucional
    #Como argumento de la clase toma un diccionario de los hiperparametro requeridos por el modelo
    def __init__(self, hyper_params,  **kwargs):
        #Inicializamos la clase padre del modelo para poder emplear sus atributos
        super(Convolutional_AE1, self).__init__(**kwargs)

        #Definimos las caracterisicas de las diferentes capas como parametrs ajustables del modelo
        self.hyper_params = hyper_params #Definimos el diccionario de hiperparametros como una variable de clase
        self.filters_c1 = self.hyper_params["filters_c1"] #Filtros del primer bloque convolucional del encoder y último del decoder
        self.filters_c2 = self.hyper_params["filters_c2"] #Filtros del segundo bloque convolucional del encoder y penúltimo del decoder
        self.kernel_size_c1 = self.hyper_params["kernel_size_c1"] #Tamaño del kernel en el primer bloque convolucional del encoder y último del decoder
        self.kernel_size_c2 = self.hyper_params["kernel_size_c2"] #Tamaño del kernel en el segundo bloque convolucional del encoder y penúltimo del decoder
        self.kernel_size_c3 = self.hyper_params["kernel_size_c3"] #Tamaño del kernel en la última capa del encoder y primera del decoder
        self.l1_penalty = self.hyper_params["l1_penalty"] #Factor de penalizacion lineal a las capas
        self.l2_penalty = self.hyper_params["l2_penalty"] #Factor de penalizacion cuadrático a las capas


        #Definimos la parte del encoder (codificación de la señal al espacio de dimensión reducida), como una serie de bloques convolucionales
        self.encoder = keras.Sequential([
            #Definimos la capa de entrada para definir el tamaño de los datos de entrada
            layers.Input(shape = [100, 1]),

            #Definimos el primer bloque convolucional, que va a extraer 8 caracteristicas de la señal y reducir la dimension de los datos a la mitad
            layers.Conv1D(filters = self.filters_c1, kernel_size = self.kernel_size_c1, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.Activation("relu"),
            layers.MaxPool1D(pool_size = 2, padding = "valid"),

            #Definimos el segundo bloque convolucional, que va a extraer 4 caracteristicas de la señal y reducir la dimension de los datos a la mitad
            layers.Conv1D(filters = self.filters_c2, kernel_size = self.kernel_size_c2, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.Activation("relu"),
            layers.MaxPool1D(pool_size = 2, padding = "valid"),

            #Definimos el tercer bloque convolucional, que va a extraer 1 caracteristica de la señal y reducir la dimension de los datos a la mitad
            layers.Conv1D(filters = 1, kernel_size = self.kernel_size_c3, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.Activation("relu"),
            layers.MaxPool1D(pool_size = 2, padding = "valid"),
        ])

        #Definimos la parte deñ decoder (reconstrucción de la señal), como una serie de bloques convolucionales
        self.decoder = keras.Sequential([
            #Defnimos la dimension de la capa de entrada en funcion de la dimension de la salida del encoder
            layers.Input(shape = self.encoder.output_shape[1:]), 

            #Definimos el primer bloque convolucional que va a extraer una caracteristica y duplicar el tamaño de la señal
            layers.Conv1DTranspose(filters = 1, kernel_size = self.kernel_size_c3, padding = "valid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.Activation("relu"),
            layers.UpSampling1D(size = 2),

            #Definimos el segundo bloque convolucional que va a extraer un total de 8 caracteristicas y duplicar el tamaño de la señal
            layers.Conv1DTranspose(filters = self.filters_c2, kernel_size = self.kernel_size_c2, padding = "valid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.Activation("relu"),
            layers.UpSampling1D(size = 2),

            #Definimos el tercer bloque convolucional que va a quedarse con un total de 4 caracteristicas y duplicar el tamaño de la señal
            layers.Conv1DTranspose(filters = self.filters_c1, kernel_size = self.kernel_size_c1, padding = "valid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)),
            layers.Activation("relu"),
            layers.UpSampling1D(size = 2),

            #Definimos la capa de salida que va a reconstruir la señal a traves de las caracteristicas extraidas en las capas convolucionales
            layers.Flatten(), #Aplanamos la señal para poder meterla en una capa densa
            layers.Dense(units = 100, activation = "sigmoid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty)), #Funcion de activacion sigmoid
        ])

    #Definimos el metoda call del modelo para estrablecer como debe ser la topología de la red al aplicarse sobre un conjunto de datos de entrada
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
    #Modificamos las funciones de get_config() y from_config() para que la serializacion del modelo al guardarlo se haga correctamente
    def get_config(self):
        config = super(Convolutional_AE1, self).get_config()
        config.update({
            'hyper_params': self.hyper_params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        hyper_params = config.pop('hyper_params')
        return cls(hyper_params, **config)

##Una vez definido el modelo de autoencoder debemos definir el hipermodelo asociado a dicho modelo para poder ajustar los hiperparametros que hemos dejado libres en la arquitectura del modelo.
#Para ello instanciamos la clase que define el autoencoder dentro de una nueva clase kt.HyperModel que vamos a definir para poder optimizar los hiperparametros
class CAE_1_HyperModel(kt.HyperModel):
    #En primer lugar inicializamos la calse, donde se va a definir el diccionario que contiene los hiperparametros que se quieren optimizar
    def __init__(self):
        self.hyper_params = {
            #Hiperparametros asociados a la arquitectura del modelo
            "filters_c1" : lambda hp: hp.Int("filters_c1", min_value = 1, max_value = 10, step = 1), #Filtros del primer bloque convolucional del encoder y último del decoder
            "filters_c2": lambda hp: hp.Int("filters_c2", min_value = 1, max_value = 10, step = 1), #Filtros del segundo bloque convolucional del encoder y penúltimo del decoder
            "kernel_size_c1" : lambda hp: hp.Choice("kernel_size_c1", [3, 5, 7]), #Tamaño del kernel en el primer bloque convolucional del encoder y último del decoder
            "kernel_size_c2" : lambda hp: hp.Choice("kernel_size_c2", [3, 5, 7]), #Tamaño del kernel en el segundo bloque convolucional del encoder y penúltimo del decoder
            "kernel_size_c3" : lambda hp: hp.Choice("kernel_size_c3", [3, 5, 7]), #Tamaño del kernel en la última capa del encoder y primera del decoder

            #Pesos de las penalizaciones aplicadas al modelo. Combinacion de penalizacion lineal y cuadrática
            "l1_penalty" : lambda hp: hp.Choice("l1_penalty", [1e-4, 1e-5, 1e-6, 1e-7]),
            "l2_penalty" : lambda hp: hp.Choice("l2_penalty", [1e-4, 1e-5, 1e-6, 1e-7]),
            
            #Hiperparametros del compilador del modelo
            "learning_rate" : lambda hp: hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]), #Ritmo de aprendizaje

            #Hiperparametros del entrenamiento
            "batch_size" : lambda hp: hp.Int("batch_size", min_value = 50, max_value = 300, step = 50),
            "min_delta" : lambda hp: hp.Choice("min_delta", [0.01, 0.005, 0.001]), #Criterio de convergencia del early_stopping
            "patience" : lambda hp: hp.Int("patience", min_value = 50, max_value = 100, step = 10), #Número minimo de iteraciones del early_stopping
        } 
    
    #Creamos el metodo build (constructor del hipermodelo), que va a definir el modelo teniendo en cuenta los diferentes hiperparametros
    def build(self, hp):
        #Creamos el diccionario de hiperparametros con los valores seleccionados en el constructor
        hyperparameters = {key: func(hp) for key, func in self.hyper_params.items()} #Para cada par (key, función que define cada hiperparametro correspondiente) se devuelve el par (key, valor elegido de los posibles del hiperparametro para probar)

        #Generamos el modelo añadiendo como argumentos los hiperparametros definidos en el constructor
        autoencoder = Convolutional_AE1(hyperparameters)

        #Compilamos el modelo teniendo en cuenta los hiperparametros
        autoencoder.compile(
            optimizer = keras.optimizers.Adam(learning_rate = hyperparameters["learning_rate"]), #Hiperparametro asociado al ritmo de entrenamiento
            loss = keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.CosineSimilarity()],
        )
        return autoencoder

    #Sobreescribimos el metodo de entrenamiento para poder añadir ciertos hiperparametros
    def fit(self, hp, model, *args, **kwargs):
        kwargs['batch_size'] = self.hyper_params["batch_size"](hp)
        kwargs['callbacks'] = [
            EarlyStopping(
                min_delta=self.hyper_params["min_delta"](hp),
                patience=self.hyper_params["patience"](hp),
                restore_best_weights=True
            )
        ]
        return model.fit(*args, **kwargs)
    

##-----------------------------------------Segundo modelo convolucional-----------------------------------------------##

##Definimos un modelo de Autoencoder convolucional para tratar de reducir las señales.
#Este modelo trata de comprimir la señal, pero no extraé parametros que caractericen la señal.
@register_keras_serializable()
class Convolutional_AE2(Model):
    ##Creamos la funcion que inicializa el modelo de autoencoder convolucional
    #Como argumento de la clase toma un diccionario de los hiperparametro requeridos por el modelo
    def __init__(self, hyper_params,  **kwargs):
        #Inicializamos la clase padre del modelo para poder emplear sus atributos
        super(Convolutional_AE2, self).__init__(**kwargs)

        #Definimos las caracterisicas de las diferentes capas como parametrs ajustables del modelo
        self.hyper_params = hyper_params #Definimos el diccionario de hiperparametros como una variable de clase
        self.filters_c1 = self.hyper_params["filters_c1"] #Filtros del primer bloque convolucional del encoder y último del decoder
        self.filters_c2 = self.hyper_params["filters_c2"] #Filtros del segundo bloque convolucional del encoder y penúltimo del decoder. Va a corresponder tambien a la dimension del espacio latente
        self.kernel_size_c1 = self.hyper_params["kernel_size_c1"] #Tamaño del kernel en el primer bloque convolucional del encoder y último del decoder
        self.kernel_size_c2 = self.hyper_params["kernel_size_c2"] #Tamaño del kernel en el segundo bloque convolucional del encoder y penúltimo del decoder
        self.l1_penalty = self.hyper_params["l1_penalty"] #Factor de penalizacion lineal a las capas
        self.l2_penalty = self.hyper_params["l2_penalty"] #Factor de penalizacion cuadrático a las capas


        #Definimos la parte del encoder (codificación de la señal al espacio de dimensión reducida), como una serie de bloques convolucionales
        self.encoder, kernel_size_c3 = self.__build_encoder__()
        #Definimos la parte del decoder (reconstrucción de la señal), como una serie de bloques convolucionales
        self.decoder = self.__build_decoder__(kernel_size_c3)

    #Definimos la funcion que va a construir el encoder mediante Functional API para poder modificar de manera dinámica las dimensiones de las capas
    def __build_encoder__(self):
        #Empezamos definiendo la capa de entrada del encoder
        inputs = layers.Input(shape = [100,1])

        #Definimos mediante Functional API los diferentes bloques del encoder hasta llegar a la penúltuma capa, teniendo en cuenta los hiperparametros. En Functional API se define la capa y sobre que datos actua
        #Primer bloque convolucional:
        x = layers.Conv1D(filters = self.filters_c1, kernel_size = self.kernel_size_c1, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(inputs)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool1D(pool_size = 2, padding = "valid")(x)

        #Segundo bloque convolucional:
        x = layers.Conv1D(filters = self.filters_c2, kernel_size = self.kernel_size_c2, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool1D(pool_size = 2, padding = "valid")(x)

        #Extraemos la dimension de la señal a la salida del segundo bloque convolucional 
        output_size = x.shape[1] #Obtenemos el tamaño de la salida de la capa anterior

        #Última capa convolucional
        x = layers.Conv1D(filters = self.filters_c2, kernel_size = output_size, padding = "valid", strides = 1, kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(x)
        x = layers.Activation("relu")(x)

        #Creamos el modelo de encoder en base a la entrada inputs y la secuencia de salidas x
        encoder = Model(inputs, x, name = "encoder")

        return encoder, output_size
    
    #Definimos la funcion que va a construir el encoder mediante Functional API para poder modificar de manera dinámica las dimensiones de las capas
    def __build_decoder__(self, kernel_size_c3):
        #Empezamos definiendo la capa de entrada del decoder
        inputs = layers.Input(shape = self.encoder.output_shape[1:])

        #Definimos mediante Functional API los diferentes bloques del encoder hasta llegar a la penúltuma capa, teniendo en cuenta los hiperparametros. En Functional API se define la capa y sobre que datos actua
        #Bloque convolucional de entrada:
        x = layers.Conv1DTranspose(filters = self.filters_c2, kernel_size = kernel_size_c3, padding = "valid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(inputs)
        x = layers.Activation("relu")(x)

        #Definimos el segundo bloque convolucional que va a duplicar el tamaño de la señal
        x = layers.Conv1DTranspose(filters = self.filters_c2, kernel_size = self.kernel_size_c2, padding = "valid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(x)
        x = layers.Activation("relu")(x)
        x = layers.UpSampling1D(size = 2)(x)

        #Definimos el tercer bloque convolucional que va a duplicar el tamaño de la señal
        x = layers.Conv1DTranspose(filters = self.filters_c1, kernel_size = self.kernel_size_c1, padding = "valid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(x)
        x = layers.Activation("relu")(x)
        x = layers.UpSampling1D(size = 2)(x)

        #Definimos la capa de salida que va a reconstruir la señal a traves de las caracteristicas extraidas en las capas convolucionales
        x = layers.Flatten()(x) #Aplanamos la señal para poder meterla en una capa densa
        x = layers.Dense(units = 100, activation = "sigmoid", kernel_regularizer = keras.regularizers.L1L2(l1 = self.l1_penalty, l2 = self.l2_penalty))(x) #Funcion de activacion sigmoid

        #Creamos el modelo de decoder en base a la entrada inputs y la secuencia de salidas x
        decoder = Model(inputs, x, name ="decoder")
        
        return decoder


    #Definimos el metoda call del modelo para estrablecer como debe ser la topología de la red al aplicarse sobre un conjunto de datos de entrada
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded
    
    #Modificamos las funciones de get_config() y from_config() para que la serializacion del modelo al guardarlo se haga correctamente
    def get_config(self):
        config = super(Convolutional_AE2, self).get_config()
        config.update({
            'hyper_params': self.hyper_params,
        })
        return config

    @classmethod
    def from_config(cls, config):
        hyper_params = config.pop('hyper_params')
        return cls(hyper_params, **config)
    
##Una vez definido el modelo de autoencoder debemos definir el hipermodelo asociado a dicho modelo para poder ajustar los hiperparametros que hemos dejado libres en la arquitectura del modelo.
#Para ello instanciamos la clase que define el autoencoder dentro de una nueva clase kt.HyperModel que vamos a definir para poder optimizar los hiperparametros
class CAE_2_HyperModel(kt.HyperModel):
    #En primer lugar inicializamos la calse, donde se va a definir el diccionario que contiene los hiperparametros que se quieren optimizar
    def __init__(self):
        self.hyper_params = {
            #Hiperparametros asociados a la arquitectura del modelo
            "filters_c1" : lambda hp: hp.Int("filters_c1", min_value = 1, max_value = 10, step = 1), #Filtros del primer bloque convolucional del encoder y último del decoder
            "filters_c2": lambda hp: hp.Int("filters_c2", min_value = 1, max_value = 10, step = 1), #Filtros del segundo bloque convolucional del encoder y penúltimo del decoder
            "kernel_size_c1" : lambda hp: hp.Choice("kernel_size_c1", [3, 5, 7]), #Tamaño del kernel en el primer bloque convolucional del encoder y último del decoder
            "kernel_size_c2" : lambda hp: hp.Choice("kernel_size_c2", [3, 5, 7]), #Tamaño del kernel en el segundo bloque convolucional del encoder y penúltimo del decoder

            #Pesos de las penalizaciones aplicadas al modelo. Combinacion de penalizacion lineal y cuadrática
            "l1_penalty" : lambda hp: hp.Choice("l1_penalty", [1e-4, 1e-5, 1e-6, 1e-7]),
            "l2_penalty" : lambda hp: hp.Choice("l2_penalty", [1e-4, 1e-5, 1e-6, 1e-7]),
            
            #Hiperparametros del compilador del modelo
            "learning_rate" : lambda hp: hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]), #Ritmo de aprendizaje

            #Hiperparametros del entrenamiento
            "batch_size" : lambda hp: hp.Int("batch_size", min_value = 50, max_value = 300, step = 50),
            "min_delta" : lambda hp: hp.Choice("min_delta", [0.01, 0.005, 0.001]), #Criterio de convergencia del early_stopping
            "patience" : lambda hp: hp.Int("patience", min_value = 50, max_value = 100, step = 10), #Número minimo de iteraciones del early_stopping
        } 
    
    #Creamos el metodo build (constructor del hipermodelo), que va a definir el modelo teniendo en cuenta los diferentes hiperparametros
    def build(self, hp):
        #Creamos el diccionario de hiperparametros con los valores seleccionados en el constructor
        hyperparameters = {key: func(hp) for key, func in self.hyper_params.items()} #Para cada par (key, función que define cada hiperparametro correspondiente) se devuelve el par (key, valor elegido de los posibles del hiperparametro para probar)

        #Generamos el modelo añadiendo como argumentos los hiperparametros definidos en el constructor
        autoencoder = Convolutional_AE2(hyperparameters)

        #Compilamos el modelo teniendo en cuenta los hiperparametros
        autoencoder.compile(
            optimizer = keras.optimizers.Adam(learning_rate = hyperparameters["learning_rate"]), #Hiperparametro asociado al ritmo de entrenamiento
            loss = keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.CosineSimilarity()],
        )
        return autoencoder

    #Sobreescribimos el metodo de entrenamiento para poder añadir ciertos hiperparametros
    def fit(self, hp, model, *args, **kwargs):
        kwargs['batch_size'] = self.hyper_params["batch_size"](hp)
        kwargs['callbacks'] = [
            EarlyStopping(
                min_delta=self.hyper_params["min_delta"](hp),
                patience=self.hyper_params["patience"](hp),
                restore_best_weights=True
            )
        ]
        return model.fit(*args, **kwargs)