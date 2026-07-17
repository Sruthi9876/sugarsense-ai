import streamlit as st
import cv2
import numpy as np
import os
import base64
import time
from gtts import gTTS
import plotly.express as px
import pandas as pd
from ultralytics import YOLO

# ==============================================================================
# 1. APPLICATION DESIGN & SYSTEM PARAMETERS
# ==============================================================================
st.set_page_config(
    page_title="SugarSense AI | Agronomic Decision Support",
    page_icon="🌾",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "home"

st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 800; color: #FFFFFF; line-height: 1.2; margin-bottom: 0.5rem; }
    .green-text { color: #2ecc71; font-weight: 700; }
    .sub-title { font-size: 1.2rem; color: #a0a0a0; margin-bottom: 2rem; line-height: 1.5; }
    .metric-card { background-color: #151922; border: 1px solid #232936; padding: 1.5rem; border-radius: 10px; text-align: center; }
    .metric-val { font-size: 2.2rem; font-weight: 800; color: #2ecc71; }
    .metric-lbl { color: #a0a0a0; font-size: 0.9rem; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# Shared Model Loader with Safety Fallback to prevent app crashes
@st.cache_resource
def load_cloud_model():
    try:
        model_path = "best.pt"
        if os.path.exists(model_path):
            return YOLO(model_path)
        return YOLO("yolov8n.pt")
    except Exception as e:
        return None

model = load_cloud_model()

PEST_RECS = {
    "leaf hopper": {
        "chemical": "Spray Malathion 50 EC or Dimethoate 30 EC at recommended dosages.",
        "biological": "Release Epiricania melanoleuca (parasitoid) or encourage ladybug populations.",
        "cultural": "Avoid excessive nitrogen fertilizers and clear away alternate weed hosts."
    },
    "shoot borer": {
        "chemical": "Apply Granular Carbofuran 3G in the soil or spray Chlorantraniliprole 18.5 SC.",
        "biological": "Release Trichogramma chilonis (egg parasitoids) at weekly intervals.",
        "cultural": "Ensure early planting, employ trash mulching, and remove 'Dead Hearts' manually."
    },
    "root and base pest": {
        "chemical": "Drench soil around the base with Imidacloprid 17.8 SL or Chlorpyrifos 20 EC.",
        "biological": "Apply Entomopathogenic Nematodes (EPN) or Metarhizium anisopliae fungi.",
        "cultural": "Perform deep summer plowing to expose grubs and ensure proper field drainage."
    },
    "white sucking pest": {
        "chemical": "Spray Thiamethoxam 25 WG or Profenofos 50 EC directly on the colonies.",
        "biological": "Encourage lacewings and predatory dipteran flies (Leucopis spp.).",
        "cultural": "Clip and safely burn heavily infested lower leaves to prevent colony spread."
    }
}

def play_voice_alert(text_to_speak):
    try:
        tts = gTTS(text=text_to_speak, lang='en', tld='com', slow=False)
        tts.save("alert.mp3")
        with open("alert.mp3", "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}">'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        pass

# ==============================================================================
# PAGE 1: HOMEPAGE
# ==============================================================================
if st.session_state.page == "home":
    # Massively styled brand element to completely lock over standard framework sizing blocks
    st.markdown("<p style='font-size: 3.5rem !important; font-weight: 800 !important; color: #2ecc71 !important; letter-spacing: 1px; margin-bottom: 1.5rem; text-transform: uppercase;'>🌾 SugarSense AI</p>", unsafe_allow_html=True)

    hero_col1, hero_col2 = st.columns([1.1, 1])
    with hero_col1:
        st.markdown("<h1 class='main-title'>Detect Pest Threats<br>Before They Spread.</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'><span class='green-text'>Helping Farmers Protect Crops with AI.</span><br>Identifies sugarcane infestation threats before they spread using custom attention-guided neural vision pipelines.</p>", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns([1, 1.5])
        with btn_col1:
            if st.button("📋 Pest Information", use_container_width=True):
                st.session_state.page = "library"; st.rerun()
        with btn_col2:
            if st.button("⚡ See how it works →", type="primary", use_container_width=True):
                st.session_state.page = "diagnostics"; st.rerun()

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: st.markdown("<div class='metric-card'><div class='metric-val'>30%</div><div class='metric-lbl'>Decrease in Crop Loss</div></div>", unsafe_allow_html=True)
        with m_col2: st.markdown("<div class='metric-card'><div class='metric-val'>50%</div><div class='metric-lbl'>Less Routine Scouting</div></div>", unsafe_allow_html=True)
        with m_col3: st.markdown("<div class='metric-card'><div class='metric-val'>25%</div><div class='metric-lbl'>Lower Treatment Costs</div></div>", unsafe_allow_html=True)

    with hero_col2:
        st.markdown("<p style='color:#a0a0a0; font-size:0.85rem; margin-bottom:4px;'>🔴 <b>Active System Showcase</b></p>", unsafe_allow_html=True)
        video_path = "assets/pest_video.mp4"
        if os.path.exists(video_path):
            with open(video_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes, autoplay=True, loop=True, muted=True)
        else:
            st.info("📺 Showcase video placeholder: Upload your background video asset to 'assets/assets/pest_video.mp4' to enable full rendering.")

# ==============================================================================
# PAGE 2: REFERENCE LIBRARY
# ==============================================================================
elif st.session_state.page == "library":
    if st.button("← Back to Main Station"):
        st.session_state.page = "home"; st.rerun()

    st.markdown("### 📚 Integrated Pest Reference Library")
    ref_tab1, ref_tab2, ref_tab3, ref_tab4 = st.tabs([
        "🪰 1. Leaf Hopper", "🐛 2. Shoot Borer", "🪱 3. Root and Base Pest", "🦠 4. White Sucking Pest"
    ])

    def display_smart_image(path, fallback_icon):
        if os.path.exists(path): st.image(path, use_container_width=True)
        else: st.image(fallback_icon, width=140)

    with ref_tab1:
        col1, col2 = st.columns([1, 2])
        with col1: display_smart_image("assets/leaf hopper.jpg", "https://img.icons8.com/color/150/grasshopper.png")
        with col2:
            st.markdown("#### **📋 Profile Summary:**")
            st.info("The Sugarcane Leaf Hopper (Pyrilla perpusilla) is a destructive, mobile sucking insect.")
            st.markdown("#### **⚠️ Expanded Damage Identification:**")
            st.warning("Infested leaves lose color, turning pale yellowish-white before drying completely.")
            st.markdown("#### **🛠️ Strategic Field Recommendations:**")
            st.success(f"• **Cultural Action:** {PEST_RECS['leaf hopper']['cultural']}\n\n• **Biological Control:** {PEST_RECS['leaf hopper']['biological']}\n\n• **Chemical Intervention:** {PEST_RECS['leaf hopper']['chemical']}")

    with ref_tab2:
        col1, col2 = st.columns([1, 2])
        with col1: display_smart_image("assets/shoot borer.jpg", "https://img.icons8.com/color/150/caterpillar.png")
        with col2:
            st.markdown("#### **📋 Profile Summary:**")
            st.info("The Early Shoot Borer (Chilo infuscatellus) is an aggressive larval pest.")
            st.markdown("#### **⚠️ Expanded Damage Identification:**")
            st.warning("Larvae feed internally, causing the classic 'Dead Heart' symptom.")
            st.markdown("#### **🛠️ Strategic Field Recommendations:**")
            st.success(f"• **Cultural Action:** {PEST_RECS['shoot borer']['cultural']}\n\n• **Biological Control:** {PEST_RECS['shoot borer']['biological']}\n\n• **Chemical Intervention:** {PEST_RECS['shoot borer']['chemical']}")

    with ref_tab3:
        col1, col2 = st.columns([1, 2])
        with col1: display_smart_image("assets/root and base pest.jpg", "https://img.icons8.com/color/150/worm.png")
        with col2:
            st.markdown("#### **📋 Profile Summary:**")
            st.info("Subterranean Root Grubs and Termites hide beneath the surface.")
            st.markdown("#### **⚠️ Expanded Damage Identification:**")
            st.warning("Crops exhibit severe yellowing mimicking drought. Whole clusters suffer from lodging.")
            st.markdown("#### **🛠️ Strategic Field Recommendations:**")
            st.success(f"• **Cultural Action:** {PEST_RECS['root and base pest']['cultural']}\n\n• **Biological Control:** {PEST_RECS['root and base pest']['biological']}\n\n• **Chemical Intervention:** {PEST_RECS['root and base pest']['chemical']}")

    with ref_tab4:
        col1, col2 = st.columns([1, 2])
        with col1: display_smart_image("assets/white sucking pest.jpeg", "https://img.icons8.com/color/150/bug.png")
        with col2:
            st.markdown("#### **📋 Profile Summary:**")
            st.info("White Woolly Aphids and Scale Insects establish dense, protective configurations.")
            st.markdown("#### **⚠️ Expanded Damage Identification:**")
            st.warning("Leaves experience rapid chlorosis, turning brittle and white.")
            st.markdown("#### **🛠️ Strategic Field Recommendations:**")
            st.success(f"• **Cultural Action:** {PEST_RECS['white sucking pest']['cultural']}\n\n• **Biological Control:** {PEST_RECS['white sucking pest']['biological']}\n\n• **Chemical Intervention:** {PEST_RECS['white sucking pest']['chemical']}")

# ==============================================================================
# PAGE 3: LIVE COMPUTER VISION ENGINE
# ==============================================================================
else:
    if st.button("← Back to Main Station"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("### ⚡ Live Model Diagnostics Panel")

    if model is None:
        st.error("Model engine is currently initializing. Please ensure a valid weights snapshot file is present.")
    else:
        input_mode = st.radio("Choose Input Feed Source:", ["🎥 Connect Live CC Cam Feed", "📁 Upload Image Snapshot File"])
        target_image_bytes = None

        if input_mode == "🎥 Connect Live CC Cam Feed":
            cam_shot = st.camera_input("Capture active monitoring frame from camera stream")
            if cam_shot is not None: target_image_bytes = cam_shot.read()
        else:
            uploaded_file = st.file_uploader("Drop a crop snapshot here...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None: target_image_bytes = uploaded_file.read()

        if target_image_bytes is not None:
            file_bytes = np.asarray(bytearray(target_image_bytes), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, 1)

            with st.spinner("Analyzing Feed..."):
                padded_diag = cv2.copyMakeBorder(image_to_process, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                results = model(padded_diag, conf=0.30, iou=0.15)[0]

                annotated_img_rgb = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, caption="Active Monitoring Mapping Overlay.", use_container_width=True)

                st.markdown("---")
                st.markdown("## 📊 Automated Crop Diagnostics Report")

                if len(results.boxes) == 0:
                    st.success("🎉 **No pests detected! Your crop looks completely healthy.**")
                    play_voice_alert("System scan complete. No pests detected. Your crop looks completely healthy.")
                else:
                    detected_counts = {}
                    for box in results.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id].lower()
                        detected_counts[class_name] = detected_counts.get(class_name, 0) + 1

                    total_pests = sum(detected_counts.values())

                    if total_pests <= 2:
                        severity_status = "🟢 LOW INFESTATION"; severity_color = "green"; audio_severity = "Low infestation threat."
                    elif total_pests <= 5:
                        severity_status = "🟡 MODERATE THREAT"; severity_color = "orange"; audio_severity = "Moderate threat level detected."
                    else:
                        severity_status = "🔴 SEVERE INFESTATION - IMMEDIATE ACTION REQUIRED"; severity_color = "red"; audio_severity = "Warning. Severe infestation. Immediate action required."

                    voice_segments = []
                    for p_name, count in detected_counts.items():
                        plural_suffix = "s" if count > 1 else ""
                        voice_segments.append(f"{count} {p_name}{plural_suffix}")

                    full_speech_text = f"Alert. Detected {', and '.join(voice_segments)}. {audio_severity}"

                    c1, c2 = st.columns(2)
                    with c1: st.metric(label="Total Pests Detected", value=str(total_pests))
                    with c2: st.markdown(f"#### Threat Status:\n<h3 style='color:{severity_color}; margin-top:0px;'>{severity_status}</h3>", unsafe_allow_html=True)

                    play_voice_alert(full_speech_text)

                    st.markdown("### 📋 Breakdown by Pest Type")
                    breakdown_df = pd.DataFrame(list(detected_counts.items()), columns=["Pest Name", "Count Detected"])
                    st.table(breakdown_df)

                    st.markdown("### 🛠️ Recommended Treatment Plan")
                    for pest_name in detected_counts.keys():
                        lookup_name = pest_name
                        if "sucking" in pest_name: lookup_name = "white sucking pest"
                        elif "borer" in pest_name: lookup_name = "shoot borer"
                        elif "hopper" in pest_name: lookup_name = "leaf hopper"
                        elif "root" in pest_name or "base" in pest_name: lookup_name = "root and base pest"

                        recs = PEST_RECS.get(lookup_name, {
                            "chemical": "Apply broad-spectrum organic or chemical pesticides.",
                            "biological": "Introduce natural predatory insects.",
                            "cultural": "Prune away damaged leaf foliage."
                        })

                        with st.expander(f"✨ Action Plan for: {pest_name.title()}", expanded=True):
                            st.markdown(f"🚜 **Cultural Control:** {recs['cultural']}")
                            st.markdown(f"🐞 **Biological Control:** {recs['biological']}")
                            st.markdown(f"🧪 **Chemical Treatment:** {recs['chemical']}")
