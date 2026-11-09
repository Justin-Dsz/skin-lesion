import json
import numpy as np, torch, timm, cv2, streamlit as st
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="Skin Lesion Risk Checker", page_icon="🩺")

classes = json.load(open("classes.json"))
cfg = json.load(open("config.json"))
TEMP, THRESH = cfg["temperature"], cfg.get("flag_threshold", 0.15)
RISKY = ["mel", "bcc", "akiec"]

INFO = {
    "mel":   ("Melanoma", "🔴 High", "A serious skin cancer. See a dermatologist as soon as possible."),
    "bcc":   ("Basal cell carcinoma", "🔴 High", "A common skin cancer. See a dermatologist soon."),
    "akiec": ("Actinic keratosis", "🟠 Medium", "Can be pre-cancerous. A medical review is recommended."),
    "nv":    ("Melanocytic nevus (common mole)", "🟢 Low", "Usually harmless. Watch for changes in size, shape or colour."),
    "bkl":   ("Benign keratosis-like lesion", "🟢 Low", "Usually harmless. See a doctor if it changes."),
    "df":    ("Dermatofibroma", "🟢 Low", "A harmless skin nodule."),
    "vasc":  ("Vascular lesion", "🟢 Low", "Usually harmless (for example an angioma)."),
}

@st.cache_resource
def load_model():
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
    m.load_state_dict(torch.load("model.pth", map_location="cpu"))
    return m.eval()

model = load_model()

# must match training: resize to 256x256, centre-crop 224, normalise
prep = T.Compose([T.CenterCrop(224), T.ToTensor(),
                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

st.title("🩺 Skin Lesion Risk Checker")
st.warning("Academic prototype. This is NOT a medical diagnosis or a substitute for a doctor.")
st.caption("Detects only 7 pigmented lesion types (listed in the sidebar). It cannot recognise rashes or infections.")

with st.sidebar:
    st.header("About")
    st.write("Trained on HAM10000 (dermatoscopic images). Conditions it knows:")
    for c in classes:
        st.write(f"• {INFO[c][0]}")
    st.write("**Limitations:** trained on dermatoscope images, so phone photos may be less reliable. "
             "Mostly lighter skin tones in training data. Confirmation of cancer needs a biopsy.")

file = st.file_uploader("Upload a clear, well-lit, close-up photo of the lesion", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Your image", width=300)

    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
        st.warning("The image looks blurry. Results may be unreliable. Try a sharper photo.")

    img256 = img.resize((256, 256))
    x = prep(img256).unsqueeze(0)
    with torch.no_grad():
        p = F.softmax(model(x) / TEMP, 1)[0].numpy()

    order = p.argsort()[::-1]
    top = classes[order[0]]
    concern = sum(p[i] for i, c in enumerate(classes) if c in RISKY)

    st.subheader("Result")
    if concern >= THRESH:
        st.error(f"⚠️ Possible signs of skin cancer or pre-cancer (combined score {concern*100:.0f}%). "
                 "Please have this checked by a dermatologist.")
    elif p[order[0]] < 0.5:
        st.info("Unable to assess confidently. Please consult a doctor if you are worried.")
    else:
        name, tier, advice = INFO[top]
        st.success(f"Most likely: {name}. Risk level: {tier}. {advice}")

    st.subheader("Top 3 predictions")
    for i in order[:3]:
        st.write(f"**{INFO[classes[i]][0]}**: {p[i]*100:.1f}%")
        st.progress(float(p[i]))

    cam = GradCAM(model=model, target_layers=[model.conv_head])
    g = cam(input_tensor=x, targets=None)[0]
    rgb = np.array(img256.crop((16, 16, 240, 240))).astype(np.float32) / 255
    st.subheader("What the model looked at")
    st.image(show_cam_on_image(rgb, g, use_rgb=True), width=300,
             caption="Warmer colours = regions that influenced the prediction")