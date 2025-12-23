import cv2
import numpy as np
import base64

def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def pupil_position(eye_gray):
    eye_gray = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, thresh = cv2.threshold(eye_gray, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(largest)
    return x + w // 2

def detect_gaze(image):
    debug_img = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {
            "mirando": False,
            "razon": "No se detectó rostro",
            "imagen": image_to_base64(debug_img)
        }

    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = debug_img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

    if len(eyes) < 2:
        return {
            "mirando": False,
            "razon": "No se detectaron ambos ojos",
            "imagen": image_to_base64(debug_img)
        }

    ratios = []

    for (ex, ey, ew, eh) in eyes[:2]:
        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_color = roi_color[ey:ey+eh, ex:ex+ew]

        pupil_x = pupil_position(eye_gray)
        if pupil_x is None:
            continue

        ratio = pupil_x / float(ew)
        ratios.append(ratio)

        # 🎯 Centro del ojo
        center_x = ew // 2
        center_y = eh // 2

        # 🎯 Dibujos
        cv2.circle(eye_color, (pupil_x, center_y), 4, (0, 255, 0), -1)
        cv2.circle(eye_color, (center_x, center_y), 4, (255, 0, 0), -1)
        cv2.line(
            eye_color,
            (0, center_y),
            (ew, center_y),
            (255, 255, 0),
            1
        )

    if not ratios:
        return {
            "mirando": False,
            "razon": "No se detectó pupila",
            "imagen": image_to_base64(debug_img)
        }

    promedio = sum(ratios) / len(ratios)
    mirando = 0.40 <= promedio <= 0.60

    deviation = abs(promedio - 0.5)
    confidence = max(0.0, (0.1 - deviation) / 0.1) * 100
    

    return {
    "veredicto": "MIRA A CÁMARA" if round(float(confidence), 2)>=75 else "NO MIRA A CÁMARA",
    "mirando": bool(mirando),
    "ratio": round(float(promedio), 3),
    "confidence": round(float(confidence), 2),
    "imagen": image_to_base64(debug_img)
          }

