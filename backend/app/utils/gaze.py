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
    x, y, w, h = cv2.boundingRect(largest)
    return x + w // 2


def detect_gaze(image):
    debug_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        confidence = 0.0
        promedio = 0.0
        return {
            "veredicto": "MIRA A CÁMARA" if confidence >= 70 else "NO MIRA A CÁMARA",
            "mirando": False,
            "ratio": promedio,
            "confidence": confidence,
            "imagen": image_to_base64(debug_img)
        }

    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = debug_img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

    ratios = []
    eye_centers = []

    for (ex, ey, ew, eh) in eyes[:2]:
        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_color = roi_color[ey:ey+eh, ex:ex+ew]

        center_x = ew // 2
        center_y = eh // 2
        eye_centers.append(ex + center_x)

        cv2.circle(eye_color, (center_x, center_y), 4, (255, 0, 0), -1)

        pupil_x = pupil_position(eye_gray)
        if pupil_x is not None:
            ratio = pupil_x / float(ew)
            ratios.append(ratio)
            cv2.circle(eye_color, (pupil_x, center_y), 4, (0, 255, 0), -1)

    # ======================
    # DECISIÓN FINAL
    # ======================

    # 🔹 Método 1: Pupila real
    if ratios:
        promedio = sum(ratios) / len(ratios)
        deviation = abs(promedio - 0.5)
        confidence = max(0.0, (0.1 - deviation) / 0.1) * 100

    # 🔹 Método 2: Simetría ocular
    elif len(eye_centers) == 2:
        avg_eye_x = sum(eye_centers) / 2
        face_center = w / 2
        desviacion = abs(avg_eye_x - face_center) / w

        promedio = 0.5
        confidence = 78.0 if desviacion < 0.08 else 60.0

    # 🔹 Método 3: Muy poca info
    else:
        promedio = 0.0
        confidence = 45.0

    mirando = confidence >= 70

    return {
        "veredicto": "MIRA A CÁMARA" if confidence >= 70 else "NO MIRA A CÁMARA",
        "mirando": bool(mirando),
        "ratio": round(float(promedio), 3),
        "confidence": round(float(confidence), 2),
        "imagen": image_to_base64(debug_img)
    }
