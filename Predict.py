import tensorflow as tf
import numpy as np
from glob import glob
import sys

cat_dataset = "./Cat_Dataset"
save_directory = "./Caches"
test = sys.argv[1]
img_height = 160
img_width = 160

#Définition du nom de classe des chats
class_names = sorted([i.replace(cat_dataset, "") for i in glob(cat_dataset+"/*")])
print(class_names)
model = tf.keras.models.load_model(save_directory)

#Pour chaque fichier dans ce dossier Image_Test, Daisy va essayer de prédire la race du chat.
for path in sorted(glob(test+"/*")):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(img_height, img_width, 3))

    #Transforme l'image en array, où chaque pixel est une liste, avec 3 valeurs RBG
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    #Prédit la ressemblance de l'image avec chaqu'une des classes qu'elle connaît
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    print(f"L'Image {path} appartient sûrement aux {class_names[np.argmax(score)]} avec {round(np.max(score)*100, 2)}% de sûreté")
