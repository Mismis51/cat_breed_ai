#Pour des informations plus détaillées, aller voir sur https://www.tensorflow.org/

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

class make_dataset:
    '''
    Définition des paramètres des datasets. Pour créer un dataset :
    make_dataset(directory, batch_size, img_height, img_width).train() => dataset d'entraînement
    make_dataset(directory, batch_size, img_height, img_width).val() => dataset avec les images de validation
    '''
    def __init__(self, directory, batch_size, img_height, img_width):
        self.directory = directory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

    def train(self):
        '''
        Création du dataset d'entraînement. Renvoie un tuple image_en_array, nom_de_classe
        '''
        train = tf.keras.preprocessing.image_dataset_from_directory(
         self.directory,
         labels = "inferred",
         label_mode="int", 
         validation_split=0.2,
         subset="training",
         seed=123,
         image_size=(self.img_height, self.img_width),
         batch_size=self.batch_size)
        train = train.shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE) # Va mélanger les images dans le dataset
        return(train)
    
    def val(self):
        '''
        Création du dataset de validation. Renvoie un tuple image_en_array, nom_de_classe
        '''
        val = tf.keras.preprocessing.image_dataset_from_directory(
         self.directory,
         labels = "inferred",
         label_mode="int",
         validation_split=0.2,
         subset="validation",
         seed=123,
         image_size=(self.img_height, self.img_width),
         batch_size=self.batch_size)
        val = val.prefetch(buffer_size=tf.data.AUTOTUNE)
        return(val)



class make_model:
    '''
    Création du modèle. On peut faire son propre modèle, ou commencer à partir du modèle déjà entraîné de Google :
    make_model(imh_height, img_width, train_ds, num_classes).load_v2(learning_rate) => Créer et compiler un modèle à partir de MobileNetV2
    make_model(imh_height, img_width, train_ds, num_classes).create_model(learning_rate) => Créer et compiler un modèle à partir de ses données
    '''
    def __init__(self, img_height, img_width, train_ds, num_classes):

        #Prépare la couche d'augmentation de données. Cela change certains paramètres de l'image, comme l'orientation.
        #Cela nous donne effectivement de nouvelles images
        self.data_augmentation = Sequential([
         layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
         layers.experimental.preprocessing.RandomRotation(0.1),
         layers.experimental.preprocessing.RandomZoom(0.1),])
        self.img_height = img_height
        self.img_width = img_width
        self.train_ds = train_ds
        self.num_classes = num_classes
    
    def load_v2(self, learning_rate):
        '''
        Création de modèle avec MobileNetV2
        '''

        #Charge le modèle MobileNetV2, et bloque son apprentissage (il y a bien trop de couches dedans)
        base_model = tf.keras.applications.MobileNetV2(input_shape=(self.img_height, self.img_width) + (3,), include_top=False, weights='imagenet')
        base_model.trainable = False

        #Rajoute certaines couches supplémentaires qui nous appartiennent, qui elles peuvent être entraînées
        inputs = tf.keras.Input(shape=(self.img_height, self.img_width, 3))
        x = self.data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)
        model = tf.keras.Model(inputs, outputs)

        #Compile de modèle avec l'optimisateur Adam. Cela utilise des concepts compliqués, mais ça réduit l'overfitting
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        return(model, base_model)
    
    def create_model(self, learning_rate):
        '''
        Création de modèle avec ses données à soi
        '''

        #Utilise le module Sequeltial de keras pour créer des couches
        model = Sequential([
         self.data_augmentation,
         layers.experimental.preprocessing.Rescaling(1./255), #Change les valeurs rgb à des valeurs [0, 1] pour faciliter l'apprentissage
         layers.Conv2D(16, 3, padding='same', activation='relu'),
         layers.MaxPooling2D(),
         layers.Conv2D(32, 3, padding='same', activation='relu'),
         layers.MaxPooling2D(),
         layers.Conv2D(64, 3, padding='same', activation='relu'),
         layers.MaxPooling2D(),
         layers.Dropout(0.2),
         layers.Flatten(),
         layers.Dense(128, activation='relu'),
         layers.Dense(self.num_classes)
        ])

        #Compile de modèle avec l'optimisateur Adam. Cela utilise des concepts compliqués, mais ça réduit l'overfitting
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        return(model, model)



class compile_for_fine_tuning:
    '''
    Recompiler un modèle pour l'entraîner à un taux d'apprentissage moins élevé pour évitter du overfitting :
    compile_for_fine_tuning(model, base_model, learning_rate).unlock(layer) => débloque certaines couches pour les entraîner et gagner un peu plus de précision
    '''
    def __init__(self, model, base_model, learning_rate):
        self.model = model
        self.base_model = base_model
        self.learning_rate = learning_rate/10
    
    def unlock(self, layer):
        '''
        Débloque certaines couches et recompile le modèle
        '''
        #Débloque les couches supérieures, qui sont moins importantes que les couches inférieures, pour pouvoir les entraîner.
        self.base_model.trainable = True
        for layer in self.base_model.layers[:layer]:
            layer.trainable =  False

        #On baisse le taux d'apprentissage pour éviter l'overfitting, et on recompile tout ça, avec un optimisateur différent
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    optimizer = tf.keras.optimizers.RMSprop(lr=self.learning_rate),
                    metrics=['accuracy'])
        return(self.model)
            