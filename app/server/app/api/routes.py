import torch
import numpy as np
import pickle
from fastapi import APIRouter
from pathlib import Path
from pydantic import BaseModel
from app.inference.model import HandwritingModel
from app.inference.generate import generate_sequence
from app.inference.utils import visualize_handwriting

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "handwriting-100-epoch.pt"
STYLE_STROKE_PATH = BASE_DIR / "model" / "style_stroke.pkl"
STYLE_TEXT_PATH = BASE_DIR / "model" / "style_text.pkl"

# Load the model
print("Loading model...")
model = HandwritingModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
model = model.to(device)
model.eval()

# Initialize hidden and window vector
hidden, window_vec, kappa = model.init_hidden(1, device)

# Load style data
with open(STYLE_STROKE_PATH, "rb") as f:
    style_stroke = pickle.load(f)

with open(STYLE_TEXT_PATH, "rb") as f:
    style_text = pickle.load(f)

print("Model loaded successfully.")

class InferenceRequest(BaseModel):
    text: str
    bias: float = 0.0
    style: int = 0

@router.post("/generate")
def generate_handwriting(req: InferenceRequest):
    print(f"Received request: {req.text} with bias: {req.bias} and style: {req.style}")
    try:
        prime = False
        prime_seq = None
        real_text = None

        if req.style > 0 and req.style < len(style_text):
            prime = True
            style = style_stroke[req.style]
            prime_seq = torch.from_numpy(style).unsqueeze(0).to(device)
            real_text = style_text[req.style]

        gen_seq = generate_sequence(
            model, req.text, bias=req.bias,
            prime=prime, prime_seq=prime_seq, real_text=real_text,
            hidden=hidden, window_vec=window_vec, kappa=kappa, device=device,
        )

        gen_seq = np.array(gen_seq).tolist()

        return {"strokes": gen_seq}
    except Exception as e:
        print("Error generating:", e)
        return []
