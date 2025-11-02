# IA_Cocina_RL/src/detector_module.py
"""
Módulo de detección de ingredientes usando YOLOv8 entrenado con FoodSeg103
"""
from ultralytics import YOLO
from pathlib import Path
import os

# --- CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / 'models' / 'food_detector' / 'best.pt'
# ------------------------------

# Variable global para mantener el modelo cargado
_model = None

def load_model():
    """
    Carga el modelo YOLO entrenado una sola vez (singleton pattern)
    
    Returns:
        YOLO: Modelo cargado o None si hay error
    """
    global _model
    
    if _model is not None:
        return _model
    
    try:
        if not MODEL_PATH.exists():
            print(f"❌ ERROR: No se encontró el modelo en {MODEL_PATH}")
            print(f"   Asegúrate de haber entrenado el modelo primero con:")
            print(f"   python src/02_train_detector.py")
            return None
        
        print(f"📦 Cargando modelo desde: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
        print(f"✅ Modelo cargado exitosamente")
        print(f"   • Número de clases: {len(_model.names)}")
        print(f"   • Modelo: YOLOv8n")
        
        return _model
    
    except Exception as e:
        print(f"❌ ERROR al cargar el modelo: {e}")
        return None


def detect_ingredients(image_path, conf_threshold=0.15, max_ingredients=15):
    """
    Detecta ingredientes en una imagen usando el modelo YOLO entrenado
    
    Args:
        image_path: Ruta a la imagen
        conf_threshold: Umbral de confianza mínimo (0.0 - 1.0)
                       Bajado a 0.15 porque el modelo tiene baja precisión
        max_ingredients: Número máximo de ingredientes a retornar
    
    Returns:
        list: Lista de ingredientes detectados (nombres únicos)
    """
    # Cargar modelo si no está cargado
    model = load_model()
    
    if model is None:
        return ["Error: Modelo no cargado. Verifica que exista models/food_detector/best.pt"]
    
    try:
        # Verificar que la imagen exista
        if not os.path.exists(image_path):
            return [f"Error: No se encontró la imagen en {image_path}"]
        
        print(f"\n🔍 Detectando ingredientes en: {image_path}")
        print(f"   • Umbral de confianza: {conf_threshold}")
        
        # Realizar predicción
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False,
            device=0 if _model.device.type == 'cuda' else 'cpu'
        )
        
        result = results[0]
        num_detections = len(result.boxes)
        
        print(f"   • Detecciones encontradas: {num_detections}")
        
        if num_detections == 0:
            # Si no detecta nada, sugerir ingredientes genéricos
            print("   ⚠️  No se detectaron ingredientes con confianza suficiente")
            return [
                "tomate", "cebolla", "ajo", "pollo", "arroz",
                "pasta", "huevos", "queso", "pimiento", "zanahoria"
            ]
        
        # Extraer ingredientes únicos
        ingredients_detected = []
        confidences = []
        
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            class_name = model.names[int(cls)]
            confidence = float(conf)
            
            # Evitar duplicados
            if class_name not in ingredients_detected:
                ingredients_detected.append(class_name)
                confidences.append(confidence)
                print(f"      • {class_name}: {confidence:.2%}")
        
        # Limitar número de ingredientes
        if len(ingredients_detected) > max_ingredients:
            # Ordenar por confianza y tomar los top N
            sorted_ingredients = sorted(
                zip(ingredients_detected, confidences),
                key=lambda x: x[1],
                reverse=True
            )
            ingredients_detected = [ing for ing, _ in sorted_ingredients[:max_ingredients]]
            print(f"   ⚠️  Limitando a los {max_ingredients} ingredientes más confiables")
        
        # Si aún así hay muy pocos, complementar con básicos
        if len(ingredients_detected) < 3:
            print("   ℹ️  Complementando con ingredientes básicos")
            basic_ingredients = ["aceite de oliva", "sal", "pimienta", "ajo", "cebolla"]
            for basic in basic_ingredients:
                if basic not in ingredients_detected and len(ingredients_detected) < 5:
                    ingredients_detected.append(basic)
        
        print(f"✅ Ingredientes finales: {', '.join(ingredients_detected)}")
        return ingredients_detected
    
    except Exception as e:
        print(f"❌ ERROR durante la detección: {e}")
        import traceback
        traceback.print_exc()
        return [f"Error en la detección: {str(e)}"]


def get_model_info():
    """
    Retorna información sobre el modelo cargado
    
    Returns:
        dict: Diccionario con información del modelo
    """
    model = load_model()
    
    if model is None:
        return {
            'loaded': False,
            'error': 'Modelo no encontrado'
        }
    
    return {
        'loaded': True,
        'model_path': str(MODEL_PATH),
        'num_classes': len(model.names),
        'classes': list(model.names.values()),
        'device': model.device.type
    }


# Función de compatibilidad con código anterior
def detect_food_items(image_path):
    """
    Alias de detect_ingredients para compatibilidad
    """
    return detect_ingredients(image_path)


if __name__ == "__main__":
    # Prueba del módulo
    print("=" * 60)
    print("🧪 PRUEBA DEL MÓDULO DE DETECCIÓN")
    print("=" * 60)
    
    # Mostrar info del modelo
    info = get_model_info()
    if info['loaded']:
        print(f"\n✅ Modelo cargado correctamente")
        print(f"   • Ruta: {info['model_path']}")
        print(f"   • Clases: {info['num_classes']}")
        print(f"   • Device: {info['device']}")
        print(f"\n📋 Primeras 20 clases:")
        for i, cls in enumerate(info['classes'][:20], 1):
            print(f"   {i}. {cls}")
    else:
        print(f"\n❌ {info['error']}")
    
    # Probar detección en imagen de prueba
    test_image = ROOT_DIR / 'data' / 'yolo_dataset' / 'images' / 'val'
    if test_image.exists():
        test_images = list(test_image.glob('*.jpg'))[:3]
        if test_images:
            print(f"\n🔍 Probando detección en {len(test_images)} imágenes...")
            for img in test_images:
                ingredients = detect_ingredients(str(img))
                print(f"\n📸 {img.name}:")
                print(f"   {', '.join(ingredients)}")