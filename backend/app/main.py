from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.compare import router as compare_router
from app.routes.mirada import router as gaze_router


app = FastAPI()

#  CORS (solo para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod se limita
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(compare_router, prefix="/api")
app.include_router(gaze_router, prefix="/api")

