import cv2
import os

# Carica il file XML per il riconoscimento facciale
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Carica il modello addestrato
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

# Ottieni i nomi dagli ID nel dataset
names = [user for user in os.listdir("dataset")]

# Avvia la webcam (0 = webcam integrata, 1 = webcam esterna)
video_capture = cv2.VideoCapture(0)

print("[INFO] Webcam avviata. Premi 'q' per uscire.")

while True:
    # Cattura un frame dal video
    ret, frame = video_capture.read()
    if not ret:
        print("[ERRORE] Impossibile catturare il video.")
        break

    # Converti il frame in scala di grigi
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Rileva i volti
    faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    # Riconoscimento facciale con precisione
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Predice l'ID e ottiene la confidenza
        id, confidence = recognizer.predict(gray_frame[y:y+h, x:x+w])
        
        # Normalizza la precisione (100% = perfetto, 0% = incerto)
        precision = max(0, min(100, 100 - confidence))  # Invertiamo il valore

        if id and id - 1 < len(names) and precision>50:  # Verifica che l'ID sia valido e che ci sia una corrispondenza almeno del 50%
            name = f"{names[id - 1]} ({precision:.2f}%)"
        else:
            name = "Unknown"

        # Mostra il nome e la precisione sul video
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostra il video in tempo reale
    cv2.imshow("Recognize", frame)

    # Premere 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Rilascia la webcam e chiudi le finestre
video_capture.release()
cv2.destroyAllWindows()
