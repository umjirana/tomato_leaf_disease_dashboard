from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MPLCONFIGDIR = BASE_DIR / ".cache" / "matplotlib"
MODEL_PATH = BASE_DIR / "models" / "densenet121_tomato.keras"
CLASS_NAMES_PATH = BASE_DIR / "data" / "class_names.txt"
IMAGE_SIZE = (256, 256)
