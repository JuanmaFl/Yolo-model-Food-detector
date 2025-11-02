# src/01_prepare_data.py
import os
import json
from pathlib import Path
import shutil
from PIL import Image
import base64
import zlib
import traceback
import numpy as np
from io import BytesIO

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FoodSeg103"
YOLO_OUTPUT_PATH = PROJECT_ROOT / "data" / "yolo_dataset"

# --- VARIABLES GLOBALES ---
CLASS_NAME_TO_INDEX = {}

def load_class_mapping():
    """Carga las clases del meta.json y crea el mapeo Nombre de Clase → índice YOLO."""
    meta_json_path = RAW_DATA_PATH / "meta.json"
    if not meta_json_path.exists():
        raise FileNotFoundError(f"❌ No se encontró meta.json en {meta_json_path}")

    with open(meta_json_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    if "classes" not in meta_data or not isinstance(meta_data["classes"], list):
        raise ValueError("❌ Formato inesperado en meta.json.")

    class_names = [item["title"] for item in meta_data["classes"]]
    global CLASS_NAME_TO_INDEX
    CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(class_names)}

    print(f"✅ Mapeo de clases cargado. Total de clases: {len(class_names)}.")
    return class_names

def convert_bbox_to_yolo_format(bbox_pixels, img_width, img_height, class_index):
    """Convierte [x_min, y_min, w, h] → formato YOLO normalizado."""
    x_min, y_min, w, h = bbox_pixels
    if w <= 0 or h <= 0:
        return None

    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Validar que los valores estén en rango [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w_norm <= 1 and 0 < h_norm <= 1):
        return None

    return f"{class_index} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

def bitmap_to_bbox(bitmap_data, origin):
    """
    Decodifica un bitmap PNG de Supervisely y extrae el bounding box.
    
    Args:
        bitmap_data: String base64 con el PNG comprimido
        origin: [x, y] offset del bitmap en la imagen original
    
    Returns:
        [x_min, y_min, width, height] o None si falla
    """
    try:
        # 1. Decodificar Base64
        encoded_bytes = base64.b64decode(bitmap_data)
        
        # 2. Descomprimir con Zlib
        try:
            png_bytes = zlib.decompress(encoded_bytes)
        except zlib.error:
            png_bytes = encoded_bytes
        
        # 3. Verificar que sea PNG
        if not png_bytes.startswith(b'\x89PNG'):
            return None
        
        # 4. Leer el PNG como imagen
        mask_img = Image.open(BytesIO(png_bytes))
        mask_array = np.array(mask_img)
        
        # 5. Convertir a máscara binaria (1 = objeto, 0 = fondo)
        # El PNG es una imagen indexada con paleta, donde 255 = blanco = objeto
        if len(mask_array.shape) == 2:
            # Imagen en escala de grises
            binary_mask = (mask_array > 0).astype(np.uint8)
        else:
            # Imagen RGB, tomar cualquier canal no-negro
            binary_mask = (mask_array[:, :, 0] > 0).astype(np.uint8)
        
        # 6. Encontrar coordenadas de píxeles activos
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # 7. Calcular dimensiones del bbox
        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)
        
        # 8. Aplicar offset del origin
        x_min_global = int(x_min + origin[0])
        y_min_global = int(y_min + origin[1])
        
        return [x_min_global, y_min_global, width, height]
    
    except Exception as e:
        return None

def process_split_data(split_name, class_names):
    """Procesa train/test y genera imágenes + etiquetas YOLO."""
    print(f"\n{'='*60}")
    print(f"📂 Procesando: {split_name.upper()}")
    print(f"{'='*60}")

    ann_dir = RAW_DATA_PATH / split_name / "ann"
    img_dir = RAW_DATA_PATH / split_name / "img"

    if not ann_dir.exists() or not img_dir.exists():
        print(f"❌ No se encontraron las carpetas necesarias")
        return

    # Mapear yolo split
    yolo_split = "val" if split_name == "test" else split_name
    out_img_dir = YOLO_OUTPUT_PATH / "images" / yolo_split
    out_lbl_dir = YOLO_OUTPUT_PATH / "labels" / yolo_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Crear un set de imágenes disponibles
    available_images = {}
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        for img_file in img_dir.glob(f"*{ext}"):
            available_images[img_file.stem] = img_file

    print(f"📊 Imágenes disponibles: {len(available_images)}")

    annotation_files = list(ann_dir.glob("*.json"))
    print(f"📋 Anotaciones encontradas: {len(annotation_files)}")

    processed = 0
    skipped_no_img = 0
    skipped_no_objects = 0
    errors = 0

    for ann_file in annotation_files:
        try:
            # Extraer el nombre base
            ann_stem = ann_file.stem
            if ann_stem.endswith(('.jpg', '.jpeg', '.png')):
                img_base_name = ann_stem.rsplit('.', 1)[0]
            else:
                img_base_name = ann_stem

            # Buscar imagen correspondiente
            if img_base_name not in available_images:
                skipped_no_img += 1
                continue

            img_file = available_images[img_base_name]

            # Leer dimensiones de la imagen
            with Image.open(img_file) as img:
                img_w, img_h = img.size

            # Leer anotaciones
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Procesar objetos
            labels = []
            for obj in data.get("objects", []):
                class_title = obj.get("classTitle")
                bitmap = obj.get("bitmap")

                if not (class_title and bitmap):
                    continue

                if class_title not in CLASS_NAME_TO_INDEX:
                    continue

                bbox = bitmap_to_bbox(bitmap["data"], bitmap["origin"])
                if not bbox:
                    continue

                class_idx = CLASS_NAME_TO_INDEX[class_title]
                yolo_line = convert_bbox_to_yolo_format(bbox, img_w, img_h, class_idx)
                if yolo_line:
                    labels.append(yolo_line)

            # Solo guardar si hay etiquetas válidas
            if labels:
                # Copiar imagen
                target_img = out_img_dir / img_file.name
                if not target_img.exists():
                    shutil.copy(img_file, target_img)

                # Guardar etiquetas
                lbl_file = out_lbl_dir / (img_base_name + ".txt")
                with open(lbl_file, 'w') as f:
                    f.write("\n".join(labels))
                
                processed += 1
                
                if processed % 100 == 0:
                    print(f"   Procesadas: {processed} imágenes...")
            else:
                skipped_no_objects += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   ⚠️ Error en {ann_file.name}: {e}")

    print(f"\n{'='*60}")
    print(f"✅ {split_name.upper()} completado:")
    print(f"   ✔️  Procesadas correctamente: {processed}")
    print(f"   ⚠️  Sin imagen: {skipped_no_img}")
    print(f"   ⚠️  Sin objetos válidos: {skipped_no_objects}")
    print(f"   ❌ Errores: {errors}")
    print(f"{'='*60}")

def create_yaml_file(class_names):
    """Crea foodseg_yolo.yaml para YOLOv8."""
    yaml_content = f"""# FoodSeg103 en formato YOLOv8
path: {YOLO_OUTPUT_PATH.as_posix()}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = YOLO_OUTPUT_PATH / "foodseg_yolo.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content.strip())
    print(f"\n✅ Archivo YAML creado: {yaml_path}")

def main():
    print("=" * 60)
    print("🚀 CONVERSIÓN FOODSEG103 → FORMATO YOLO")
    print("=" * 60)
    
    try:
        class_names = load_class_mapping()
        process_split_data("train", class_names)
        process_split_data("test", class_names)
        create_yaml_file(class_names)
        
        print("\n" + "=" * 60)
        print("🎉 ¡CONVERSIÓN COMPLETADA!")
        print(f"📁 Dataset YOLO guardado en: {YOLO_OUTPUT_PATH}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()