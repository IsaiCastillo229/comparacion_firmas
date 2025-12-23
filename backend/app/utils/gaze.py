import cv2
import numpy as np
import base64

# =============================
# Utils
# =============================

def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


# =============================
# Haar Cascades
# =============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# =============================
# Pupila
# =============================

def pupil_position(eye_gray):
    eye_gray = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, thresh = cv2.threshold(
        eye_gray, 30, 255, cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x + w // 2


# =============================
# Gaze Detection
# =============================

def detect_gaze(image):
    debug_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ---------------------------
    # 1️⃣ ROSTRO
    # ---------------------------
    if len(faces) == 0:
        return _response(
            False, 0.0, None, debug_img
        )

    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = debug_img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(
        roi_gray, 1.1, 10
    )

    ratios = []
    eye_centers = []

    # ---------------------------
    # 2️⃣ INTENTO PUPILA
    # ---------------------------
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
        eye_color = roi_color[ey:ey+eh, ex:ex+ew]

        pupil_x = pupil_position(eye_gray)
        center_x = ew // 2
        center_y = eh // 2

        eye_centers.append(ex + center_x)

        if pupil_x is not None:
            ratio = pupil_x / float(ew)
            ratios.append(ratio)

            # debug
            cv2.circle(eye_color, (pupil_x, center_y), 4, (0,255,0), -1)

        cv2.circle(eye_color, (center_x, center_y), 3, (255,0,0), -1)

    # ---------------------------
    # 3️⃣ CASO PUPILA OK
    # ---------------------------
    if ratios:
        promedio = sum(ratios) / len(ratios)
        deviation = abs(promedio - 0.5)

        confidence = max(0.0, (0.15 - deviation) / 0.15) * 100
        mirando = 0.35 <= promedio <= 0.65

        return _response(
            mirando, confidence, promedio, debug_img
        )

    # ---------------------------
    # 4️⃣ FALLBACK: SIMETRÍA DE OJOS
    # ---------------------------
    if len(eye_centers) == 2:
        dist_eyes = abs(eye_centers[0] - eye_centers[1])
        face_center = w / 2

        symmetry = abs(
            (eye_centers[0] + eye_centers[1]) / 2 - face_center
        )

        if symmetry < w * 0.08:
            return _response(
                True, 65.0, None, debug_img
            )

    # ---------------------------
    # 5️⃣ FALLBACK FINAL: ROSTRO FRONTAL
    # ---------------------------
    aspect_ratio = w / float(h)

    if 0.75 <= aspect_ratio <= 1.3:
        return _response(
            True, 50.0, None, debug_img
        )

    # ---------------------------
    # 6️⃣ NEGATIVO DEFINITIVO
    # ---------------------------
    return _response(
        False, 25.0, None, debug_img
    )


# =============================
# Respuesta uniforme
# =============================

def _response(mirando, confidence, ratio, image):
    return {
        "veredicto": "MIRA A CÁMARA" if confidence >= 75 else "NO MIRA A CÁMARA",
        "mirando": bool(mirando),
        "ratio": None if ratio is None else round(float(ratio), 3),
        "confidence": round(float(confidence), 2),
        "imagen": image_to_base64(image)
    }
