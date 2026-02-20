import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- CONFIGURATION ---
st.set_page_config(page_title="Détecteur CIFAKE - M2 IABD", page_icon="🛡️")

# --- CHARGEMENT DU MODÈLE (MOBILE NET V2) ---
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, 'best_transfer_pytorch.pth')

    # On utilise MobileNetV2 car les clés de tes poids correspondent à ce modèle
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    
    if not os.path.exists(weights_path):
        st.error(f"❌ Fichier .pth introuvable dans : {current_dir}")
        return None

    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Erreur de chargement : {e}")
        return None

model = load_model()

# --- INTERFACE ---
st.title("🕵️ Détecteur d'Images IA (CIFAKE)")
st.write("Projet Master 2 IABD - **Fouda Etundi**")

uploaded_file = st.file_uploader("Charger une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image à analyser", use_container_width=True)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    if st.button("Lancer l'Analyse"):
        # 1. PRÉDICTION
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        classes = ["FAKE (IA)", "REAL (Humain)"]
        
        # 2. AFFICHAGE RÉSULTAT
        st.markdown("---")
        if pred.item() == 0:
            st.error(f"### Verdict : **{classes[0]}**")
        else:
            st.success(f"### Verdict : **{classes[1]}**")
        
        st.metric("Confiance", f"{conf.item()*100:.2f}%")

        # 3. GRAD-CAM (EXPLICATION)
        st.subheader("🧐 Pourquoi cette décision ? (Grad-CAM)")
        try:
            # Pour MobileNetV2, la dernière couche de features est la cible
            target_layers = [model.features[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            img_float = np.array(image.resize((224, 224))).astype(np.float32) / 255
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            
            st.image(cam_image, caption="Zones d'attention du modèle", use_container_width=True)
        except Exception as e:
            st.warning(f"Erreur Grad-CAM : {e}")

st.markdown("---")
st.caption("M2 IABD - 2026")
# --- SECTION PERFORMANCES ---
with st.expander("📊 Voir les performances de l'entraînement"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Précision (Accuracy)")
        # Ici, tu peux mettre une image de ta courbe si tu l'as enregistrée
        # st.image("accuracy_curve.png") 
        st.write("- Entraînement : 99.8%")
        st.write("- Validation : 94.2%")
    with col2:
        st.subheader("Perte (Loss)")
        # st.image("loss_curve.png")
        st.write("- Finale : 0.023")