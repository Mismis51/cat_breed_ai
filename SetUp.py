from glob import glob
import tensorflow as tf
import os

path = "./"

#Ceci va créer un fichier pour y upload le modèle intraîné, si ce fichier n'existe pas déjà
if "Caches" not in glob(path + "/*"):
    os.mkdir(path + "/Caches")

#Quelquefois, certaines images dans le dataset sont corrompues, et tensorflow n'interronp pas le programme avec un message d'erreur retraçable.
#La seule solution, quoiqu'elle n'est pas belle, est de regarder si il y a un fichier corrompu dans un dossier, et si oui, de décommenter les deux dernières lignes et de retracer le fichier.
for directories in glob(path + "/*"):
    print(directories)
    for file_path in glob(directories+"/*"):
        filecontents = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(filecontents, channels = 3)
#        print(image)
#        print(file_path)
