import cv2
import os
from pathlib import Path

def generate_dataset ():
    #Inizializzo il classificatore
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    #Avvio la videocamera
    vc = cv2.VideoCapture(0)

    #Leggo userID e userName
    userId = input("Inserisci il numero ID della nuova persona da riconoscere: ")
    userName = input("Inserisci il nome della nuova persona da riconoscere: ")

    # Inizializzo un count per contare quante "foto" ho scattato
    count = 1
    
    # TODO: Implementare un controllo per evitare che vengano inseriti nomi già esistenti
    # def checkUser(userId, userName):
    #     # Controllo se esiste già una cartella con lo stesso nome
    #     if os.path.exdists("dataset/{}".format(userName)):
    #         print("[ERROR] Il nome utente esiste già.")
    #         return False
    #     else:
    #         return True

    # Funzione per salvare le immagini
    def saveImage(image, userName, userId, imgId):
        # Creo una cartella
        Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
        # Salvo le immagini
        cv2.imwrite("dataset/{}/{}_{}.jpg".format(userName, userId, imgId), image)
        print("[INFO] L'immagine {} è stata salvata nella cartella: {}".format(
            imgId, userName))


    print("[INFO] La cattura video si sta avviando, attendi...")

    while True:
        # Cattura il frame
        _, img = vc.read()

        # Copia l'immagine originale
        originalImg = img.copy()

        # Rendo l'immagine grigia per ottimizzare il riconoscimetno
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Leggo le coordinate della faccia nella foto per salvare solo il volto
        faces = faceCascade.detectMultiScale(gray_img,
                                            scaleFactor=1.2,
                                            minNeighbors=5,
                                            minSize=(50, 50))

        # Disegno un rettangolo attorno al volto
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            coords = [x, y, w, h]
            # Controllo se almeno una coordinata esiste (se esiste una esistono tutte)
            if x:
                # Se ho meno di 1000 "foto" salvo una nuova foto del mio volto nel dataset 
                if count <= 1000:
                    roi_img = originalImg[coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]]
                    saveImage(roi_img, userName, userId, count)
                    count += 1
                else:
                    break

        # Mostra il video in tempo reale
        cv2.imshow("Recognize", img)

        if count == 1000:
            break

        #Attende che l'utente prema un tasto
        key = cv2.waitKey(1) & 0xFF

        # Se l'utente preme q sulla tastiera interrompe il processo
        if key == ord('q'):
            break

    print("[INFO] Il dataset è stato creato per {}".format(userName))

    # Spegne la videocamera
    vc.release()
    # Chiude la finestra
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        generate_dataset()
    except KeyboardInterrupt:
        print("[INFO] Il processo è stato interrotto dall'utente.")
    except Exception as e:
        print("[ERROR] Si è verificato un errore: {}".format(e))