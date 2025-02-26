import os
import cv2
import numpy as np

from PIL import Image #pip install pillow opencv-contrib-python opencv-python

#Inizializza nomi e percorsi come liste vuote
names = []
path = []

# Ricavo il nome di tutti gli utenti salvati
for users in os.listdir("dataset"):
    names.append(users)

# Leggo la directory di tutte le foto
for name in names:
    for image in os.listdir("dataset/{}".format(name)):
        path_string = os.path.join("dataset/{}".format(name), image)
        path.append(path_string)


faces = []
ids = []

#Per ogni immagine creo un array di tipo numpy e lo aggiungo all'elenco dei volti
for img_path in path:
    image = Image.open(img_path).convert("L")

    imgNp = np.array(image, "uint8")

    id = int(img_path.split("/")[2].split("_")[0])

    faces.append(imgNp)
    ids.append(id)

#Converto gli id to array di tipo numpy a li aggiungo alla lista degli id
ids = np.array(ids)

print("[INFO] Creati array numpy per facce e nomi")
print("[INFO] Inizializzazione del classificatore")

#Chiamo il tool di riconoscimento
trainer = cv2.face.LBPHFaceRecognizer_create()
#Gli passo gli array numpy per immagini e id
trainer.train(faces, ids)
#Scrivo il modello generato nel file training.yml
trainer.write("training.yml")

print("[INFO] Addestramento completato")
