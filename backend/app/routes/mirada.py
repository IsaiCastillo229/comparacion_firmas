from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from app.utils.gaze import detect_gaze

router = APIRouter()

@router.post("/mirada")
async def detectar_mirada(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    return detect_gaze(frame)
