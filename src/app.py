import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torchvision.transforms as transforms
from model import CytoMorphClassifier

app = FastAPI(
    title="CytoMorph-XAI Diagnostic Server",
    description="Production-grade API for identifying White Blood Cells from digital microscopic imagery.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

MEDICAL_METADATA = {
    "Basophil": {
        "ar": "خلية قعدة - تلعب دوراً أساسياً في الاستجابات الحساسية والالتهابات.",
        "en": "Basophil - Key player in allergic responses and inflammatory reactions."
    },
    "Eosinophil": {
        "ar": "خلية حمضة - تكافح العدوى الطفيلية وتنشط في حالات الحساسية والربو.",
        "en": "Eosinophil - Combats parasitic infections and triggers allergic asthma reactions."
    },
    "Lymphocyte": {
        "ar": "خلية ليمفاوية - ركيزة المناعة التكيفية، تشمل خلايا T و B لإنتاج الأجسام المضادة.",
        "en": "Lymphocyte - Core of adaptive immunity, includes T-cells and B-cells for antibody production."
    },
    "Monocyte": {
        "ar": "خلية وحيدة - أكبر خلايا الدم البيضاء حجماً، تتحول إلى خلايا بلعمية لالتهام الميكروبات.",
        "en": "Monocyte - Largest WBC type, differentiates into macrophages to engulf pathogens."
    },
    "Neutrophil": {
        "ar": "خلية متعادلة - خط الدفاع الأول والأكثر وفرة ضد العدوى البكتيرية الحادة.",
        "en": "Neutrophil - The most abundant WBC, acting as the primary defense against acute bacterial infections."
    }
}

device = torch.device("cpu")
model = CytoMorphClassifier(num_classes=5)
try:
    model.load_state_dict(torch.load("best_cytomorph_model.pth", map_location=device))
    model.eval()
except Exception as e:
    pass

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
async def predict_cell(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probabilities, 0)
            
        pred_class = CLASSES[class_idx.item()]
        
        return {
            "status": "success",
            "clinical_prediction": pred_class,
            "confidence_score": round(confidence.item(), 4),
            "bilingual_summary": {
                "arabic": MEDICAL_METADATA[pred_class]["ar"],
                "english": MEDICAL_METADATA[pred_class]["en"]
            },
            "all_probabilities": {CLASSES[i]: round(probabilities[i].item(), 4) for i in range(len(CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic failure: {str(e)}")
