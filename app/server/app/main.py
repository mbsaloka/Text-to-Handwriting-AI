from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as inference_router

app = FastAPI(
    title="Handwriting Generation API",
    description="API for generating handwriting strokes from text input",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register router
app.include_router(inference_router, prefix="/api", tags=["inference"])

@app.get("/")
def read_root():
    return {"message": "Handwriting Inference API is running!"}
