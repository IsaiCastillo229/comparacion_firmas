import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from fastapi import APIRouter, UploadFile, File, HTTPException
import base64

router = APIRouter()

@router.post("/compare")
async def compara_firmas(ine: UploadFile = File(...), documento: UploadFile = File(...)):
    # Validar tipo de archivo
    if ine.content_type.split('/')[0] != 'image' or documento.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="Uno o ambos archivos no son imágenes válidas.")

    # Leer bytes
    ine_content = await ine.read()
    documento_content = await documento.read()

    # Convertir a np.array
    ine_np = np.frombuffer(ine_content, np.uint8)
    documento_np = np.frombuffer(documento_content, np.uint8)

    # Decodificar imagen
    ine_img = cv2.imdecode(ine_np, cv2.IMREAD_COLOR)
    documento_img = cv2.imdecode(documento_np, cv2.IMREAD_COLOR)

    if ine_img is None or documento_img is None:
        raise HTTPException(status_code=400, detail="Error al decodificar las imágenes.")

    # Preprocesamiento avanzado
    ine_pre = preprocesar_imagen(ine_img)
    doc_pre = preprocesar_imagen(documento_img)

    # Comparación ORB
    orb_sim, kp1, kp2, good_matches = comparar_orb(ine_pre, doc_pre)

    # Comparación SSIM
    ssim_score, _ = comparar_ssim(ine_pre, doc_pre)
    # Combinación ponderada ORB + SSIM
    similitud_pct = round((0.6 * orb_sim + 0.4 * ssim_score) * 100, 2)
    dictamen = interpretar_resultado(similitud_pct)

    # Generar imagen de resultado
    if not good_matches:
        imagen_resultado = generar_imagen_placeholder("No se encontraron coincidencias", ine_pre.shape[1], ine_pre.shape[0])
    else:
        imagen_resultado = generar_img_diferencias(ine_pre, doc_pre, kp1, kp2, good_matches)

    imagen_base64 = imagen_a_base64(imagen_resultado)

    return {
        "similitud": similitud_pct,
        "dictamen": dictamen,
        "mensaje": "Comparación realizada correctamente",
        "imagen": imagen_base64
    }

# ------------------ FUNCIONES ------------------

def preprocesar_imagen(imagen: np.ndarray) -> np.ndarray:
    """Redimensiona, convierte a gris, suaviza y binariza la imagen."""
    altura, anchura = 150, 300
    imagen = cv2.resize(imagen, (anchura, altura))

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)

    # Binarización adaptativa para manejar firmas claras/oscuras
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    # Operaciones morfológicas para limpiar ruido
    kernel = np.ones((2, 2), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    return binaria

def comparar_orb(img1: np.ndarray, img2: np.ndarray):
    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0, kp1, kp2, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 50]

    similitud = len(good_matches) / len(matches) if matches else 0
    return similitud, kp1, kp2, good_matches

def comparar_ssim(img1: np.ndarray, img2: np.ndarray):
    score, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return score, diff

def generar_img_diferencias(img1: np.ndarray, img2: np.ndarray, kp1, kp2, good_matches):
    return cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def generar_imagen_placeholder(texto: str, ancho: int, alto: int) -> np.ndarray:
    img = np.ones((alto, ancho*2, 3), np.uint8) * 255
    (text_width, text_height), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x = (ancho*2 - text_width) // 2
    y = (alto + text_height) // 2
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return img

def imagen_a_base64(imagen: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", imagen)
    return base64.b64encode(buffer).decode("utf-8")

def interpretar_resultado(similitud: float) -> str:
    if similitud < 30:
        return "No coincide"
    elif similitud < 50:
        return "Dudoso"
    elif similitud < 70:
        return "Coincidencia parcial"
    else:
        return "Alta coincidencia"
