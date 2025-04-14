import torch
import gradio as gr
import numpy as np
import pickle
from pathlib import Path
from app.inference.generate import generate_sequence
from app.inference.model import HandwritingModel
from app.inference.utils import visualize_handwriting

print("Starting service...")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "handwriting-100-epoch.pt"
STYLE_STROKE_PATH = BASE_DIR / "model" / "style_stroke.pkl"
STYLE_TEXT_PATH = BASE_DIR / "model" / "style_text.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = HandwritingModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# Init hidden and window vector
hidden, window_vec, kappa = model.init_hidden(1, device)

with open(STYLE_STROKE_PATH, "rb") as f:
    style_stroke = pickle.load(f)
with open(STYLE_TEXT_PATH, "rb") as f:
    style_text = pickle.load(f)


def generate_handwriting(text: str, bias: float = 1.0, style_index: int = 0):
    print(f"Received request: '{text}' | Bias: {bias} | Style index: {style_index}")
    try:
        prime = False
        prime_seq = None
        real_text = None

        if style_index > 0:
            style_data_index = style_index - 1  # Karena Style 1 = style_stroke[0]
            if style_data_index < len(style_stroke):
                prime = True
                prime_seq = torch.from_numpy(style_stroke[style_data_index]).unsqueeze(0).to(device)
                real_text = style_text[style_data_index]

        gen_seq = generate_sequence(
            model, text, bias=bias,
            prime=prime, prime_seq=prime_seq, real_text=real_text,
            hidden=hidden, window_vec=window_vec, kappa=kappa, device=device,
        )

        gen_seq = np.array(gen_seq).tolist()
        img = visualize_handwriting(gen_seq)
        json_data = gen_seq

        return img, json_data

    except Exception as e:
        print("Error generating:", e)
        empty_image = np.zeros((256, 256, 3), dtype=np.uint8)
        empty_json = []  # Data JSON kosong
        return empty_image, empty_json

dropdown_choices = [("Random", 0)] + [(f"Style {i+1}", i+1) for i in range(len(style_text))]

iface = gr.Interface(
    fn=generate_handwriting,
    inputs=[
        gr.Textbox(label="Input Text"),
        gr.Slider(label="Bias", minimum=0.5, maximum=10.0, value=1.0, step=0.25),
        gr.Dropdown(
            choices=dropdown_choices,
            value=0,
            label="Priming Style"
        )
    ],
    outputs=[
        gr.Image(type="pil", label="Generated Handwriting"),
        gr.JSON(label="Stroke Data (JSON)")
    ],
    flagging_mode="never"
)

iface.launch(show_api=True)
