# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List
# import torch
# from torchvision import transforms
# from PIL import Image
# import io

# app = FastAPI()

# # CORS to allow frontend to call backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with your frontend origin in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model
# model = torch.load("model.pth", map_location=torch.device("cpu"))
# model.eval()

# # Define labels
# labels = ["Red", "Blue", "Green", "Striped", "Floral"]  # Update as per your model

# # Preprocessing pipeline
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Adjust size to match model input
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])  # Example normalization; update if needed
# ])


# @app.post("/predict/")
# async def predict(files: List[UploadFile] = File(...)):
#     predictions = []

#     for file in files:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#         img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

#         with torch.no_grad():
#             outputs = model(img_tensor)
#             _, predicted = torch.max(outputs, 1)
#             attribute = labels[predicted.item()]

#         predictions.append({
#             "filename": file.filename,
#             "attribute": attribute
#         })

#     return {"predictions": predictions}


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import io
from fastapi.responses import FileResponse
import os
from model import predict_folder
from fastapi.staticfiles import StaticFiles
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import json

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the images directory to serve files
app.mount("/images", StaticFiles(directory="images"), name="images")

IMAGES_DIR = "images"
PREDICTIONS_PATH = "predictions.json"

@app.post("/predict")
async def predict_images(files: List[UploadFile] = File(...)):
    # Ensure images directory exists
    os.makedirs(IMAGES_DIR, exist_ok=True)
    # Save uploaded images
    for file in files:
        contents = await file.read()
        with open(os.path.join(IMAGES_DIR, file.filename), "wb") as f:
            f.write(contents)
    # Run batch prediction
    results = predict_folder(IMAGES_DIR)
    # Save to predictions.json
    import json
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return {"status": "success"}

@app.get("/predictions")
def get_predictions():
    return FileResponse(PREDICTIONS_PATH, media_type="application/json")

def generate_predictions_pdf(predictions_path, images_dir, pdf_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    import os
    import json

    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Margins
    left_margin = 70
    right_margin = 70
    top_margin = 60
    bottom_margin = 60
    box_width = width - left_margin - right_margin

    # Add logo (top right)
    logo_path = os.path.join(os.path.dirname(__file__), "../Frontend/meesho-version-re2/public/TechTrio-logo-2.png")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, width - 120, height - 80, width=80, height=40, mask='auto')

    # Title (centered)
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.HexColor("#800080"))
    c.drawCentredString(width / 2, height - 50, "Fashion AI Tagger - Predictions")

    y = height - top_margin - 50  # Start below the title
    attr_line_height = 24
    img_width = 110
    img_height = 110
    img_left = left_margin + 10
    attr_left = img_left + img_width + 30
    value_left = attr_left + 150
    attr_top_offset = 18

    for idx, (filename, data) in enumerate(predictions.items(), 1):
        num_attrs = len(data["attributes"])
        content_height = max(img_height, num_attrs * attr_line_height + 10)
        box_height = content_height + 48  # Add padding inside box

        # If not enough space for the next box, start a new page
        if y - box_height < bottom_margin:
            c.showPage()
            # Redraw logo and title on new page
            if os.path.exists(logo_path):
                c.drawImage(logo_path, width - 120, height - 80, width=80, height=40, mask='auto')
            c.setFont("Helvetica-Bold", 22)
            c.setFillColor(colors.HexColor("#800080"))
            c.drawCentredString(width / 2, height - 50, "Fashion AI Tagger - Predictions")
            y = height - top_margin - 50

        # Draw box with new margins
        c.setStrokeColor(colors.HexColor("#800080"))
        c.setLineWidth(1.2)
        c.roundRect(left_margin, y - box_height, box_width, box_height, 10, stroke=1, fill=0)

        # Center image vertically in the box
        img_path = os.path.join(images_dir, filename)
        img_y = y - (box_height / 2) 
        if os.path.exists(img_path):
            c.drawImage(img_path, img_left, img_y - img_height / 2, width=img_width, height=img_height, mask='auto')

        # Filename (bold, purple, top-aligned with image)
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(colors.HexColor("#800080"))
        c.drawString(attr_left, y - 20, f"{idx}. {filename}")

        # Attributes (single column, aligned, with spacing)
        attr_y = y - 50
        for key, value in data["attributes"].items():
            attr_y -= attr_line_height
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(colors.black)
            c.drawString(attr_left, attr_y, f"{key}:")
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.HexColor("#444444"))
            c.drawString(value_left, attr_y, str(value))
            # Minimalistic faint line under each attribute
            c.setStrokeColor(colors.HexColor("#e0e0e0"))
            c.setLineWidth(0.3)
            c.line(attr_left, attr_y - 3, left_margin + box_width - 20, attr_y - 3)

        y -= (box_height + 30)  # More space between boxes

    c.save()

@app.get("/download-pdf")
def download_pdf():
    pdf_path = "predictions_report.pdf"
    generate_predictions_pdf("predictions.json", "images", pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename="predictions_report.pdf")
