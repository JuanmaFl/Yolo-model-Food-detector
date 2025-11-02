# debug_bitmap.py
import json
import base64
import zlib
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "FoodSeg103"

def analyze_bitmap_format():
    """Analiza el formato de los bitmaps en las anotaciones"""
    
    print("=" * 70)
    print("🔍 ANALIZANDO FORMATO DE BITMAPS")
    print("=" * 70)
    
    # Tomar una anotación de ejemplo
    ann_dir = RAW_DATA_PATH / "test" / "ann"
    img_dir = RAW_DATA_PATH / "test" / "img"
    
    ann_files = list(ann_dir.glob("*.json"))[:5]  # Primeras 5
    
    for i, ann_file in enumerate(ann_files, 1):
        print(f"\n{'='*70}")
        print(f"📄 Ejemplo {i}: {ann_file.name}")
        print(f"{'='*70}")
        
        # Leer anotación
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Obtener imagen correspondiente
        ann_stem = ann_file.stem
        if ann_stem.endswith(('.jpg', '.jpeg', '.png')):
            img_base_name = ann_stem.rsplit('.', 1)[0]
        else:
            img_base_name = ann_stem
        
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = img_dir / f"{img_base_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if not img_file:
            print("❌ Imagen no encontrada")
            continue
        
        # Leer dimensiones
        with Image.open(img_file) as img:
            img_w, img_h = img.size
        
        print(f"🖼️  Imagen: {img_file.name}")
        print(f"   Dimensiones: {img_w}x{img_h}")
        print(f"   Objetos en anotación: {len(data.get('objects', []))}")
        
        # Analizar primer objeto
        if data.get("objects"):
            obj = data["objects"][0]
            print(f"\n📦 Primer objeto:")
            print(f"   Clase: {obj.get('classTitle')}")
            print(f"   Tipo geometría: {obj.get('geometryType')}")
            
            bitmap = obj.get("bitmap")
            if bitmap:
                print(f"\n🔢 Estructura del bitmap:")
                print(f"   Claves: {list(bitmap.keys())}")
                print(f"   Origin: {bitmap.get('origin')}")
                
                # Analizar el data
                bitmap_data = bitmap.get("data")
                if bitmap_data:
                    print(f"   Longitud del data (base64): {len(bitmap_data)} caracteres")
                    print(f"   Primeros 50 caracteres: {bitmap_data[:50]}...")
                    
                    # Intentar decodificar
                    try:
                        rle_bytes = base64.b64decode(bitmap_data)
                        print(f"   ✅ Base64 decodificado: {len(rle_bytes)} bytes")
                        print(f"   Primeros 20 bytes (hex): {rle_bytes[:20].hex()}")
                        
                        # Verificar si es PNG
                        if rle_bytes.startswith(b'\x89PNG'):
                            print(f"   ⚠️  Es una imagen PNG incrustada")
                            continue
                        
                        # Intentar descomprimir con Zlib
                        try:
                            decompressed = zlib.decompress(rle_bytes)
                            print(f"   ✅ Zlib descomprimido: {len(decompressed)} bytes")
                            print(f"   Primeros 50 bytes: {decompressed[:50]}")
                            rle_bytes = decompressed
                        except zlib.error as e:
                            print(f"   ℹ️  No es Zlib comprimido: {e}")
                        
                        # Intentar decodificar con pycocotools
                        try:
                            from pycocotools import mask as mask_util
                            
                            # Probar diferentes enfoques
                            print(f"\n🧪 Probando decodificación RLE:")
                            
                            # Enfoque 1: Como está
                            try:
                                rle_obj = mask_util.frPyObjects([rle_bytes], img_h, img_w)
                                bbox = mask_util.toBbox(rle_obj)[0]
                                print(f"   ✅ Método 1 funcionó - BBox: {bbox}")
                            except Exception as e1:
                                print(f"   ❌ Método 1 falló: {type(e1).__name__}: {e1}")
                            
                            # Enfoque 2: Como lista de enteros
                            try:
                                rle_list = list(rle_bytes)
                                rle_obj = mask_util.frPyObjects([rle_list], img_h, img_w)
                                bbox = mask_util.toBbox(rle_obj)[0]
                                print(f"   ✅ Método 2 funcionó - BBox: {bbox}")
                            except Exception as e2:
                                print(f"   ❌ Método 2 falló: {type(e2).__name__}: {e2}")
                            
                            # Enfoque 3: Como string
                            try:
                                rle_str = rle_bytes.decode('latin1')
                                rle_obj = mask_util.frPyObjects([rle_str], img_h, img_w)
                                bbox = mask_util.toBbox(rle_obj)[0]
                                print(f"   ✅ Método 3 funcionó - BBox: {bbox}")
                            except Exception as e3:
                                print(f"   ❌ Método 3 falló: {type(e3).__name__}: {e3}")
                            
                            # Enfoque 4: Interpretar como RLE directo (pares count/value)
                            try:
                                # Convertir bytes a lista de números
                                if len(rle_bytes) % 2 == 0:
                                    # Intentar como pares de bytes
                                    import struct
                                    nums = []
                                    for j in range(0, len(rle_bytes), 2):
                                        nums.append(struct.unpack('<H', rle_bytes[j:j+2])[0])
                                    print(f"   ℹ️  RLE como pares: {len(nums)} valores")
                                    print(f"   Primeros 10: {nums[:10]}")
                            except Exception as e4:
                                print(f"   ❌ Método 4 falló: {e4}")
                                
                        except ImportError:
                            print(f"   ⚠️  pycocotools no instalado")
                        
                    except Exception as e:
                        print(f"   ❌ Error al decodificar base64: {e}")
            else:
                print(f"   ⚠️  No tiene campo 'bitmap'")

if __name__ == "__main__":
    analyze_bitmap_format()