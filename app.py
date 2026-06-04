import os
import base64
from io import BytesIO
from html import escape

from src.config import MPLCONFIGDIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import streamlit as st
from PIL import Image

from src.config import MODEL_PATH
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.model_loader import load_class_names, load_tomato_model
from src.predictor import predict_leaf_disease
from src.preprocessing import preprocess_image


st.set_page_config(
    page_title="토마토 잎 질병 감지 AI 대시보드",
    page_icon="🍅",
    layout="wide",
)


MIN_CONFIDENCE = 0.70
MIN_GREEN_RATIO = 0.12
MIN_CENTER_GREEN_RATIO = 0.08
MAX_SKIN_RATIO = 0.16
MAX_CENTER_SKIN_RATIO = 0.20


DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "label": "Bacterial Spot",
        "ko": "세균성 점무늬병",
        "description": "세균에 의해 잎과 열매에 작은 갈색 반점이 생기는 병입니다. 습도가 높고 물방울이 자주 튀는 환경에서 빠르게 퍼질 수 있습니다.",
        "actions": ["감염된 잎을 제거하고 폐기합니다.", "잎에 물이 직접 닿지 않도록 관수 방식을 조정합니다.", "도구와 손을 소독해 전염을 줄입니다."],
    },
    "Tomato___Early_blight": {
        "label": "Early Blight",
        "ko": "겹둥근무늬병",
        "description": "Alternaria 계열 곰팡이에 의해 발생하며 잎에 동심원 형태의 갈색 병반이 나타납니다. 오래된 잎에서 먼저 관찰되는 경우가 많습니다.",
        "actions": ["병든 잎을 제거하고 주변 잔재물을 정리합니다.", "통풍을 개선해 잎 표면의 습기를 줄입니다.", "예방 살균제를 정기적으로 살포합니다."],
    },
    "Tomato___Late_blight": {
        "label": "Late Blight",
        "ko": "잎마름병",
        "description": "Phytophthora infestans에 의해 발생하는 대표적인 토마토 병해입니다. 잎과 줄기에 갈색 또는 흑갈색 병반이 생기며 서늘하고 습한 환경에서 빠르게 확산됩니다.",
        "actions": ["감염된 잎과 식물체를 즉시 제거합니다.", "통풍과 배수를 개선해 습도를 낮춥니다.", "확산 전 예방 살균제를 사용합니다."],
    },
    "Tomato___Leaf_Mold": {
        "label": "Leaf Mold",
        "ko": "잎곰팡이병",
        "description": "잎 뒷면에 곰팡이층이 생기고 앞면에는 노란 반점이 나타나는 병입니다. 시설 재배처럼 습도가 높은 환경에서 자주 발생합니다.",
        "actions": ["온실 내부 환기를 강화합니다.", "식물 간 간격을 확보해 공기 흐름을 개선합니다.", "병든 잎을 조기에 제거합니다."],
    },
    "Tomato___Septoria_leaf_spot": {
        "label": "Septoria Leaf Spot",
        "ko": "Septoria 잎반점병",
        "description": "작고 둥근 회갈색 반점이 잎에 많이 생기는 병입니다. 아래쪽 잎부터 시작해 식물 전체로 번질 수 있습니다.",
        "actions": ["토양이 잎에 튀지 않도록 멀칭합니다.", "아래쪽 감염 잎을 제거합니다.", "재배 잔재물을 정리해 병원균 밀도를 낮춥니다."],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "label": "Spider Mites",
        "ko": "점박이응애 피해",
        "description": "응애가 잎의 즙액을 빨아먹어 잎이 누렇게 변하고 작은 반점이 생깁니다. 건조하고 더운 환경에서 피해가 커집니다.",
        "actions": ["잎 뒷면을 확인하고 초기 피해 잎을 제거합니다.", "과도한 건조를 피하고 적정 습도를 유지합니다.", "필요 시 응애 방제제를 사용합니다."],
    },
    "Tomato___Target_Spot": {
        "label": "Target Spot",
        "ko": "겹무늬병",
        "description": "둥근 표적 모양의 병반이 잎에 나타나는 곰팡이성 병입니다. 병반이 커지면 잎이 마르고 낙엽으로 이어질 수 있습니다.",
        "actions": ["병든 잎을 제거해 전염원을 줄입니다.", "식물체 주변 통풍을 확보합니다.", "비가림과 예방 살균 관리가 도움이 됩니다."],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "label": "Yellow Leaf Curl Virus",
        "ko": "황화잎말림바이러스",
        "description": "잎이 노랗게 변하고 말리며 생장이 위축되는 바이러스병입니다. 주로 담배가루이에 의해 전염됩니다.",
        "actions": ["감염 의심 식물은 분리하거나 제거합니다.", "담배가루이 방제를 철저히 합니다.", "방충망과 끈끈이 트랩을 활용합니다."],
    },
    "Tomato___Tomato_mosaic_virus": {
        "label": "Mosaic Virus",
        "ko": "토마토 모자이크 바이러스",
        "description": "잎에 모자이크 무늬와 기형이 나타나는 바이러스병입니다. 접촉, 작업 도구, 감염 식물체를 통해 전파될 수 있습니다.",
        "actions": ["작업 도구를 자주 소독합니다.", "감염 의심 식물과 접촉을 줄입니다.", "저항성 품종 사용을 고려합니다."],
    },
    "Tomato___healthy": {
        "label": "Healthy",
        "ko": "정상",
        "description": "업로드된 잎은 현재 모델 기준으로 뚜렷한 질병 징후가 낮게 예측되었습니다. 단, 실제 재배 환경에서는 주기적인 관찰이 필요합니다.",
        "actions": ["잎 상태를 주기적으로 관찰합니다.", "통풍과 배수를 안정적으로 유지합니다.", "병든 잎이나 해충을 초기에 확인합니다."],
    },
}


CSS = """
<style>
:root {
    --green: #19733a;
    --line: #dce3df;
    --muted: #657268;
    --text: #18202b;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: #f7faf8 !important;
    color: var(--text) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
footer {
    display: none !important;
}

[data-testid="stMain"] {
    background: radial-gradient(circle at top left, rgba(25, 115, 58, 0.06), transparent 34%), #f7faf8 !important;
}

.block-container {
    max-width: 1920px;
    padding: 0.6rem 1.55rem 0.8rem !important;
}

[data-testid="stSidebar"] {
    display: none !important;
}

.topbar {
    align-items: center;
    border-bottom: 1px solid #cfd8d2;
    display: flex;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding: 0.1rem 0 0.8rem;
}

.brand {
    align-items: center;
    display: flex;
    gap: 0.55rem;
    min-width: 0;
}

.brand-mark {
    background: #e7f5ec;
    border-radius: 8px;
    color: var(--green);
    display: grid;
    flex: 0 0 48px;
    font-size: 1.55rem;
    height: 48px;
    place-items: center;
    width: 48px;
}

.brand-title {
    color: #101826;
    font-size: clamp(2rem, 2.15vw, 2.45rem);
    font-weight: 850;
    letter-spacing: 0;
    line-height: 1.08;
    overflow-wrap: anywhere;
}

.top-meta {
    align-items: center;
    color: #1f2937;
    display: flex;
    flex: 0 0 auto;
    font-size: 1.08rem;
    font-weight: 750;
    gap: 0.35rem;
}

.dashboard-grid {
    display: grid;
    gap: 1.05rem;
    grid-template-columns: 1.02fr 1.06fr 1.04fr;
    grid-template-areas:
        "input result desc"
        "care cam cam";
}

.area-input { grid-area: input; }
.area-result { grid-area: result; }
.area-desc { grid-area: desc; }
.area-care { grid-area: care; }
.area-cam { grid-area: cam; }

.dashboard-card {
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid #d9e1dc;
    border-radius: 8px;
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.1);
    color: var(--text);
    height: 100%;
    overflow: hidden;
}

.card-title {
    background: #ffffff;
    color: var(--green);
    font-size: 1.38rem;
    font-weight: 850;
    letter-spacing: 0;
    padding: 0.9rem 1.05rem;
}

.card-body {
    padding: 1rem;
}

.sidebar-title {
    color: var(--green);
    font-size: 1.35rem;
    font-weight: 800;
    margin: 0.65rem 0 0.75rem;
}

.model-box, .status-box {
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid #d8e2db;
    border-radius: 8px;
    padding: 1.25rem;
}

.model-box {
    font-size: 1.05rem;
    line-height: 1.65;
    min-height: 11rem;
}

.model-name {
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.55rem;
}

.status-box {
    background: linear-gradient(135deg, #f3fbf5, #fbfdfb);
    border-color: #cfe5d5;
    font-size: 1rem;
    line-height: 1.55;
    min-height: 9rem;
}

.image-frame {
    background: #eef4ef;
    border-radius: 8px;
    height: clamp(20rem, 30vh, 23.5rem);
    overflow: hidden;
}

.image-frame img {
    display: block;
    height: 100%;
    object-fit: cover;
    width: 100%;
}

.placeholder-panel {
    align-items: center;
    background: #edf7f0;
    border: 1px solid #cfe5d5;
    border-radius: 8px;
    color: #19733a;
    display: flex;
    font-size: 1rem;
    font-weight: 750;
    justify-content: center;
    min-height: clamp(20rem, 30vh, 23.5rem);
    padding: 0.65rem;
    text-align: center;
}

.result-box {
    background: linear-gradient(135deg, #f3fbf5, #fbfdfb);
    border: 1px solid #8dc49d;
    border-radius: 8px;
    margin-bottom: 0.52rem;
    padding: 1rem;
    text-align: center;
}

.result-label {
    color: #29372f;
    font-size: 0.95rem;
    font-weight: 700;
}

.result-class {
    color: var(--green);
    font-size: 2.35rem;
    font-weight: 850;
    line-height: 1.15;
    margin: 0.18rem 0 0.34rem;
}

.confidence {
    align-items: baseline;
    border-top: 1px solid #a8d0b4;
    display: flex;
    gap: 0.45rem;
    justify-content: center;
    padding-top: 0.38rem;
}

.confidence strong {
    color: var(--green);
    font-size: 1.8rem;
}

.prob-heading {
    font-size: 1.18rem;
    font-weight: 850;
    margin-bottom: 0.28rem;
}

.prob-row {
    align-items: center;
    display: grid;
    font-size: 1.08rem;
    gap: 0.45rem;
    grid-template-columns: minmax(8.1rem, 1fr) minmax(7.3rem, 1.2fr) 4rem;
    margin: 0.5rem 0;
}

.prob-name {
    align-items: center;
    display: flex;
    gap: 0.33rem;
    min-width: 0;
}

.prob-dot {
    border-radius: 99px;
    flex: 0 0 auto;
    height: 0.58rem;
    width: 0.58rem;
}

.prob-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.prob-track {
    background: #e6e9ed;
    border-radius: 999px;
    height: 0.64rem;
    overflow: hidden;
}

.prob-fill {
    border-radius: 999px;
    height: 100%;
}

.prob-value {
    color: #4b5563;
    font-variant-numeric: tabular-nums;
    text-align: right;
}

.disease-desc {
    background: linear-gradient(135deg, #fff8f6, #fffdfc);
    border: 1px solid #f0d6d1;
    border-radius: 8px;
    color: #28323c;
    font-size: 1.2rem;
    line-height: 1.62;
    min-height: clamp(20rem, 30vh, 23.5rem);
    padding: 1.45rem;
}

.action-list {
    background: linear-gradient(135deg, #f7fbf7, #fff);
    border: 1px solid #d5e8da;
    border-radius: 8px;
    padding: 1.15rem;
}

.action-item {
    align-items: flex-start;
    display: flex;
    font-size: 1.12rem;
    gap: 0.5rem;
    line-height: 1.45;
    margin: 0.72rem 0;
}

.check {
    color: var(--green);
    font-weight: 900;
}

.gradcam-grid {
    align-items: center;
    display: grid;
    gap: 1.15rem;
    grid-template-columns: 1fr 3rem 1fr;
}

.cam-label {
    background: rgba(24, 32, 43, 0.72);
    border-radius: 6px;
    color: #fff;
    font-size: 0.92rem;
    font-weight: 800;
    left: 0.55rem;
    padding: 0.22rem 0.42rem;
    position: absolute;
    top: 0.5rem;
}

.cam-box {
    border-radius: 8px;
    height: clamp(15rem, 22vh, 18rem);
    overflow: hidden;
    position: relative;
}

.cam-box img {
    display: block;
    height: 100%;
    object-fit: cover;
    width: 100%;
}

.arrow {
    color: #6b7280;
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
}

.caption {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.35rem;
}

.stButton > button, [data-testid="stFileUploaderDropzone"] button {
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border-color: #cfd8d2 !important;
    border-radius: 8px;
    min-height: 5.8rem;
}

[data-testid="stFileUploaderDropzone"] section {
    padding: 0.9rem !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    border: 1px solid #cfd8d2 !important;
    color: #111827 !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    min-height: 3.2rem !important;
    min-width: 8.2rem !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #111827 !important;
}

[data-testid="stCameraInput"] button {
    background: #19733a !important;
    border-color: #19733a !important;
    color: #fff !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    min-height: 3.8rem !important;
    width: 100% !important;
}

[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border-radius: 8px !important;
    min-height: 20rem !important;
    object-fit: cover !important;
    width: 100% !important;
}

.stAlert {
    border-radius: 8px !important;
}

@media (max-width: 900px) {
    .topbar {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.4rem;
    }
    .brand-title {
        font-size: 1.25rem;
    }
    .dashboard-grid {
        grid-template-columns: 1fr;
        grid-template-areas:
            "input"
            "result"
            "desc"
            "care"
            "cam";
    }
    .prob-row {
        grid-template-columns: 1fr;
        gap: 0.3rem;
    }
    .prob-value {
        text-align: left;
    }
    .gradcam-grid {
        grid-template-columns: 1fr;
    }
    .arrow {
        transform: rotate(90deg);
    }
}
</style>
"""


@st.cache_resource
def get_model():
    return load_tomato_model()


@st.cache_data
def get_class_names():
    return load_class_names()


def display_name(class_name: str) -> str:
    return DISEASE_INFO.get(class_name, {}).get("label", class_name.replace("Tomato___", "").replace("_", " "))


def format_class_list(class_names: list[str]) -> str:
    labels = [display_name(name) for name in class_names[:4]]
    return ", ".join(labels)


def probability_rows(class_names: list[str], probabilities: np.ndarray, predicted_index: int) -> str:
    colors = ["#2f9855", "#f3b21a", "#de3d38", "#2e74dc", "#8b5cf6", "#14b8a6", "#f97316", "#6366f1", "#64748b", "#16a34a"]
    rows = []
    order = np.argsort(probabilities)[::-1]

    for rank, index in enumerate(order):
        color = colors[index % len(colors)]
        if index == predicted_index:
            color = "#de3d38"
        probability = float(probabilities[index])
        width = max(probability * 100, 1.0 if probability > 0 else 0)
        rows.append(
            '<div class="prob-row">'
            '<div class="prob-name">'
            f'<span class="prob-dot" style="background:{color};"></span>'
            f'<span class="prob-label">{escape(display_name(class_names[index]))}</span>'
            '</div>'
            '<div class="prob-track">'
            f'<div class="prob-fill" style="width:{width:.2f}%; background:{color};"></div>'
            '</div>'
            f'<div class="prob-value">{probability * 100:.1f}%</div>'
            '</div>'
        )
        if rank >= 9:
            break

    return "".join(rows)


def image_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def card_html(title: str, body: str, min_height: int | None = None, area_class: str = "") -> str:
    min_height_style = f' style="min-height:{min_height}px;"' if min_height else ""
    classes = f"dashboard-card {area_class}".strip()
    return (
        f'<div class="{classes}"{min_height_style}>'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-body">{body}</div>'
        '</div>'
    )


def image_card_body(image: Image.Image) -> str:
    return f'<div class="image-frame"><img src="{image_data_uri(image)}" alt="입력 이미지"></div>'


def estimate_leaf_presence(image: Image.Image) -> dict:
    resized = image.convert("RGB").resize((256, 256))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    r, g, b = array[..., 0], array[..., 1], array[..., 2]

    max_channel = np.max(array, axis=-1)
    min_channel = np.min(array, axis=-1)
    saturation = (max_channel - min_channel) / (max_channel + 1e-6)

    green_mask = (
        (g > r * 1.08)
        & (g > b * 1.03)
        & (saturation > 0.16)
        & (max_channel > 0.12)
    )
    green_ratio = float(np.mean(green_mask))
    center = green_mask[48:208, 48:208]
    center_green_ratio = float(np.mean(center))

    skin_mask = (
        (r > 0.32)
        & (g > 0.20)
        & (b > 0.12)
        & (r > g * 1.04)
        & (g > b * 1.03)
        & ((np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])) > 0.08)
    )
    skin_ratio = float(np.mean(skin_mask))
    center_skin_ratio = float(np.mean(skin_mask[48:208, 48:208]))
    has_large_skin_region = skin_ratio >= MAX_SKIN_RATIO or center_skin_ratio >= MAX_CENTER_SKIN_RATIO

    return {
        "is_likely_leaf": (
            green_ratio >= MIN_GREEN_RATIO
            and center_green_ratio >= MIN_CENTER_GREEN_RATIO
            and not has_large_skin_region
        ),
        "green_ratio": green_ratio,
        "center_green_ratio": center_green_ratio,
        "skin_ratio": skin_ratio,
        "center_skin_ratio": center_skin_ratio,
        "has_large_skin_region": has_large_skin_region,
    }


def gradcam_body(original: Image.Image, heatmap_image: Image.Image) -> str:
    return (
        '<div class="gradcam-grid">'
        '<div class="cam-box">'
        '<span class="cam-label">원본 이미지</span>'
        f'<img src="{image_data_uri(original)}" alt="원본 이미지">'
        '</div>'
        '<div class="arrow">→</div>'
        '<div class="cam-box">'
        '<span class="cam-label">Grad-CAM 히트맵</span>'
        f'<img src="{image_data_uri(heatmap_image)}" alt="Grad-CAM 히트맵">'
        '</div>'
        '</div>'
    )


def result_body(class_label: str, confidence_label: str, probabilities_html: str = "") -> str:
    probability_section = (
        '<div class="prob-heading">클래스별 확률</div>'
        f'{probabilities_html}'
        '<div class="caption">※ 확률은 Softmax 출력값 기준입니다.</div>'
        if probabilities_html
        else '<div class="caption">이미지가 입력되면 클래스별 확률이 표시됩니다.</div>'
    )
    return (
        '<div class="result-box">'
        '<div class="result-label">예측 클래스</div>'
        f'<div class="result-class">{escape(class_label)}</div>'
        f'<div class="confidence"><span>신뢰도</span><strong>{escape(confidence_label)}</strong></div>'
        '</div>'
        f'{probability_section}'
    )


def dashboard_grid(cards: list[str]) -> str:
    return '<div class="dashboard-grid">' + ''.join(cards) + '</div>'


st.markdown(CSS, unsafe_allow_html=True)

model_missing = not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-mark">🍅</div>
            <div class="brand-title">토마토 잎 질병 감지 AI 대시보드</div>
        </div>
        <div class="top-meta">🌿 Python / Streamlit&nbsp;&nbsp;☰</div>
    </div>
    """,
    unsafe_allow_html=True,
)

sidebar_col, main_col = st.columns([0.22, 0.78], gap="large")

with sidebar_col:
    st.markdown('<div class="sidebar-title">ⓘ 모델 정보</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="model-box">
            <div class="model-name">DenseNet121</div>
            <div><strong>분류 클래스:</strong></div>
            <div style="margin-top:0.45rem; line-height:1.7;">토마토 잎 질병 10종</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">▣ 입력 방식</div>', unsafe_allow_html=True)
    camera_file = st.camera_input("사진 촬영", label_visibility="collapsed")
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    st.divider()
    if model_missing:
        st.markdown(
            """
            <div class="status-box">
                <div style="font-weight:800; color:#b45309;">모델 파일 없음</div>
                <div style="margin-top:0.55rem; line-height:1.7;">학습 완료 후 모델 저장 셀을 실행하세요.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-box">
                <div style="font-weight:800; color:#19733a;">● 모델 로드 완료</div>
                <div style="margin-top:0.55rem; line-height:1.7;">모델이 성공적으로 로드되었습니다.<br>예측을 수행할 준비가 완료되었습니다.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="caption">© 2024 Tomato AI Lab<br>버전 1.0.0</div>', unsafe_allow_html=True)

if model_missing:
    with main_col:
        st.warning("학습된 모델 파일이 없습니다. 노트북 학습이 끝난 뒤 저장 셀을 실행하면 대시보드가 예측을 시작합니다.")
        st.code("model.save(MODEL_PATH)", language="python")
    st.stop()

model = get_model()
class_names = get_class_names()
input_file = camera_file or uploaded_file

if input_file is None:
    cards = [
        card_html(
            "▣ 1. 입력 이미지",
            '<div class="placeholder-panel">왼쪽 입력 패널에서 토마토 잎 이미지를 촬영하거나 업로드하세요.</div>',
            area_class="area-input",
        ),
        card_html("▥ 2. 분석 결과", result_body("대기 중", "--%"), area_class="area-result"),
        card_html(
            "▣ 3. 질병 설명",
            '<div class="disease-desc">분석 결과가 표시되면 예측된 질병의 설명과 관리 방법을 함께 확인할 수 있습니다.</div>',
            area_class="area-desc",
        ),
        card_html(
            "◉ 4. 관리 방법",
            '<div class="action-list"><div class="action-item"><span class="check">●</span><span>이미지를 업로드하면 관리 방법이 표시됩니다.</span></div><div class="action-item"><span class="check">●</span><span>예측 결과에 따라 질병별 조치 방법을 확인할 수 있습니다.</span></div></div>',
            area_class="area-care",
        ),
        card_html(
            "◎ 5. Grad-CAM 시각화",
            '<div class="gradcam-grid"><div class="placeholder-panel">원본 이미지</div><div class="arrow">→</div><div class="placeholder-panel">Grad-CAM 히트맵</div></div>',
            area_class="area-cam",
        ),
    ]
    with main_col:
        st.markdown(dashboard_grid(cards), unsafe_allow_html=True)
    st.stop()

image = Image.open(input_file)
result = predict_leaf_disease(model, image, class_names)
predicted_index = int(np.argmax(result["probabilities"]))
predicted_class = result["class_name"]
info = DISEASE_INFO.get(predicted_class, {})
confidence = result["confidence"]
leaf_check = estimate_leaf_presence(image)

is_valid_leaf_prediction = leaf_check["is_likely_leaf"] and confidence >= MIN_CONFIDENCE

if is_valid_leaf_prediction:
    result_label = display_name(predicted_class)
    confidence_label = f"{confidence * 100:.1f}%"
    description_body = (
        '<div class="disease-desc">'
        f'<strong>{escape(info.get("ko", display_name(predicted_class)))}({escape(display_name(predicted_class))})</strong>은/는 '
        f'{escape(info.get("description", "예측된 클래스에 대한 설명 정보가 없습니다."))}'
        '</div>'
    )
    actions = info.get("actions", ["잎 상태를 주기적으로 관찰합니다.", "통풍과 배수를 관리합니다.", "필요 시 전문가의 진단을 받습니다."])

    try:
        heatmap = make_gradcam_heatmap(model, preprocess_image(image))
        gradcam_image = overlay_heatmap(image, heatmap)
        gradcam_html = gradcam_body(image, gradcam_image)
    except Exception as exc:
        gradcam_html = f'<div class="placeholder-panel">Grad-CAM을 생성하지 못했습니다: {escape(str(exc))}</div>'
else:
    result_label = "분석 불가"
    if not leaf_check["is_likely_leaf"]:
        confidence_label = "비토마토잎 입력"
        if leaf_check["has_large_skin_region"]:
            invalid_reason = (
                "입력 이미지에서 사람 얼굴/피부로 보이는 영역이 크게 감지되었습니다. "
                f"피부색 영역 비율 {leaf_check['skin_ratio'] * 100:.1f}%, "
                f"중앙 피부색 영역 비율 {leaf_check['center_skin_ratio'] * 100:.1f}%입니다."
            )
        else:
            invalid_reason = (
                "입력 이미지에서 토마토 잎으로 볼 만한 녹색 잎 영역이 충분히 감지되지 않았습니다. "
                f"녹색 영역 비율 {leaf_check['green_ratio'] * 100:.1f}%, "
                f"중앙 녹색 영역 비율 {leaf_check['center_green_ratio'] * 100:.1f}%입니다."
            )
    else:
        confidence_label = f"최고 확률 {confidence * 100:.1f}%"
        invalid_reason = (
            f"최고 예측 확률이 {MIN_CONFIDENCE * 100:.0f}% 미만이라 질병 진단으로 사용하기 어렵습니다."
        )
    description_body = (
        '<div class="disease-desc">'
        '<strong>토마토 잎 이미지로 판단하기 어렵습니다.</strong><br><br>'
        f'{escape(invalid_reason)}<br><br>'
        '이 모델은 토마토 잎 질병 데이터만 학습했기 때문에 사람 얼굴, 사물, 배경 사진을 넣으면 '
        '질병 클래스 중 하나를 억지로 선택할 수 있습니다. 토마토 잎이 선명하게 보이는 이미지를 다시 입력하세요.'
        '</div>'
    )
    actions = [
        "토마토 잎이 화면 대부분을 차지하도록 다시 촬영합니다.",
        "사람 얼굴, 손, 배경 물체가 크게 포함되지 않도록 합니다.",
        "잎의 병반이 잘 보이는 밝고 선명한 이미지를 업로드합니다.",
    ]
    gradcam_html = (
        '<div class="gradcam-grid">'
        f'<div class="cam-box"><span class="cam-label">입력 이미지</span><img src="{image_data_uri(image)}" alt="입력 이미지"></div>'
        '<div class="arrow">×</div>'
        '<div class="placeholder-panel">토마토 잎으로 판단하기 어려워 Grad-CAM을 표시하지 않습니다.</div>'
        '</div>'
    )

action_html = "".join(f'<div class="action-item"><span class="check">●</span><span>{escape(action)}</span></div>' for action in actions)

cards = [
    card_html("▣ 1. 입력 이미지", image_card_body(image), area_class="area-input"),
    card_html(
        "▥ 2. 분석 결과",
        result_body(
            result_label,
            confidence_label,
            probability_rows(class_names, result["probabilities"], predicted_index),
        ),
        area_class="area-result",
    ),
    card_html("▣ 3. 질병 설명", description_body, area_class="area-desc"),
    card_html("◉ 4. 관리 방법", f'<div class="action-list">{action_html}</div>', area_class="area-care"),
    card_html("◎ 5. Grad-CAM 시각화", gradcam_html, area_class="area-cam"),
]
with main_col:
    st.markdown(dashboard_grid(cards), unsafe_allow_html=True)
