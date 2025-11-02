# explore_dataset.py
import os
import json
from pathlib import Path

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FoodSeg103"

def explore_dataset():
    """Explora la estructura del dataset FoodSeg103"""
    
    print("=" * 60)
    print("🔍 EXPLORANDO DATASET FoodSeg103")
    print("=" * 60)
    
    # 1. Verificar que existe la carpeta principal
    if not RAW_DATA_PATH.exists():
        print(f"❌ No se encontró la carpeta: {RAW_DATA_PATH}")
        return
    
    print(f"\n📁 Ruta base: {RAW_DATA_PATH}")
    print(f"✅ Carpeta encontrada\n")
    
    # 2. Listar subcarpetas
    print("📂 Estructura de carpetas:")
    for item in RAW_DATA_PATH.iterdir():
        if item.is_dir():
            print(f"  └─ {item.name}/")
    
    # 3. Explorar meta.json
    meta_path = RAW_DATA_PATH / "meta.json"
    if meta_path.exists():
        print(f"\n✅ meta.json encontrado")
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"   Estructura: {list(meta.keys())}")
        if "classes" in meta:
            print(f"   Número de clases: {len(meta['classes'])}")
            print(f"   Primeras 3 clases: {[c['title'] for c in meta['classes'][:3]]}")
    
    # 4. Explorar carpetas train y test
    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"📊 Explorando: {split.upper()}")
        print(f"{'='*60}")
        
        split_path = RAW_DATA_PATH / split
        if not split_path.exists():
            print(f"❌ No existe la carpeta {split}")
            continue
        
        # Explorar subcarpetas
        ann_dir = split_path / "ann"
        img_dir = split_path / "img"
        
        if ann_dir.exists():
            ann_files = list(ann_dir.glob("*"))
            print(f"\n📋 Carpeta de anotaciones: {ann_dir.name}/")
            print(f"   Total de archivos: {len(ann_files)}")
            if ann_files:
                sample = ann_files[0]
                print(f"   Ejemplo de archivo: {sample.name}")
                print(f"   Extensión: {sample.suffix}")
                
                # Leer un ejemplo de anotación
                if sample.suffix == ".json":
                    with open(sample, 'r', encoding='utf-8') as f:
                        ann_data = json.load(f)
                    print(f"   Claves en JSON: {list(ann_data.keys())}")
                    if "objects" in ann_data:
                        print(f"   Número de objetos: {len(ann_data['objects'])}")
                        if ann_data["objects"]:
                            obj = ann_data["objects"][0]
                            print(f"   Claves del objeto: {list(obj.keys())}")
        
        if img_dir.exists():
            img_files = list(img_dir.glob("*"))
            print(f"\n🖼️  Carpeta de imágenes: {img_dir.name}/")
            print(f"   Total de archivos: {len(img_files)}")
            if img_files:
                # Agrupar por extensión
                extensions = {}
                for img in img_files[:100]:  # Revisar primeros 100
                    ext = img.suffix.lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
                
                print(f"   Extensiones encontradas: {extensions}")
                print(f"   Ejemplo de imagen: {img_files[0].name}")
        
        # 5. Verificar correspondencia entre anotaciones e imágenes
        if ann_dir.exists() and img_dir.exists():
            print(f"\n🔗 Verificando correspondencia:")
            ann_files = list(ann_dir.glob("*.json"))
            
            if ann_files:
                # Tomar 5 ejemplos
                for ann_file in ann_files[:5]:
                    # Probar diferentes patrones de nombres
                    base_name = ann_file.stem
                    
                    # Patrón 1: archivo.json → archivo.jpg
                    if base_name.endswith('.jpg'):
                        img_name = base_name  # Ya tiene la extensión
                    elif base_name.endswith('.png'):
                        img_name = base_name
                    else:
                        img_name = base_name  # Sin extensión
                    
                    # Buscar imagen correspondiente
                    found = False
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        if base_name.endswith(('.jpg', '.jpeg', '.png')):
                            # El stem ya incluye extensión, quitarla
                            clean_name = base_name.rsplit('.', 1)[0]
                            img_path = img_dir / f"{clean_name}{ext}"
                        else:
                            img_path = img_dir / f"{base_name}{ext}"
                        
                        if img_path.exists():
                            print(f"   ✅ {ann_file.name} → {img_path.name}")
                            found = True
                            break
                    
                    if not found:
                        print(f"   ❌ {ann_file.name} → No se encontró imagen correspondiente")
                        # Mostrar qué se buscó
                        print(f"      Buscado como: {base_name}.jpg, {base_name}.png")

if __name__ == "__main__":
    explore_dataset()