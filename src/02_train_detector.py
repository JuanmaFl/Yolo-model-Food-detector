# IA_Cocina_RL/src/02_train_detector.py
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_YOLO = ROOT_DIR / 'data' / 'yolo_dataset'
DATA_YAML = DATA_YOLO / 'foodseg_yolo.yaml'
RUNS_DIR = ROOT_DIR / 'runs'
MODELS_DIR = ROOT_DIR / 'models' / 'food_detector'
# ------------------------------

def check_environment():
    """Verifica el entorno antes de entrenar"""
    print("=" * 60)
    print("🔍 VERIFICANDO ENTORNO - Predator Helios 300")
    print("=" * 60)
    
    # Verificar Python
    python_version = sys.version.split()[0]
    print(f"✅ Python: {python_version}")
    
    # Verificar PyTorch y CUDA
    print(f"✅ PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU disponible: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
        print(f"   CUDA: {torch.version.cuda}")
        
        # Estimación de tiempo para Predator Helios 300
        print(f"\n⏱️  ESTIMACIÓN DE TIEMPO (200 épocas):")
        if "RTX" in gpu_name or "GTX 16" in gpu_name:
            print(f"   • GPU {gpu_name}: ~8-10 horas")
            print(f"   • Con 1,033 imágenes de entrenamiento")
            print(f"   • Con 2,135 imágenes de validación")
            print(f"   • Perfecto para entrenamiento nocturno 🌙")
        else:
            print(f"   • Estimado: ~10-14 horas")
        
        device = 0  # Usar GPU
    else:
        print("⚠️  No hay GPU disponible. Entrenando en CPU (será MUY lento)")
        print("⏱️  ESTIMACIÓN: ~96+ horas para 200 épocas en CPU")
        device = 'cpu'
    
    # Verificar archivo YAML
    if not DATA_YAML.exists():
        print(f"❌ ERROR: No se encontró {DATA_YAML}")
        print(f"   Asegúrate de haber ejecutado 01_prepare_data.py primero")
        return None
    print(f"✅ Dataset YAML encontrado: {DATA_YAML}")
    
    # Verificar imágenes
    train_imgs = DATA_YOLO / 'images' / 'train'
    val_imgs = DATA_YOLO / 'images' / 'val'
    
    if train_imgs.exists():
        train_count = len(list(train_imgs.glob('*.jpg')))
        print(f"✅ Imágenes de entrenamiento: {train_count}")
    else:
        print("❌ ERROR: No se encontró carpeta de imágenes de entrenamiento")
        return None
    
    if val_imgs.exists():
        val_count = len(list(val_imgs.glob('*.jpg')))
        print(f"✅ Imágenes de validación: {val_count}")
    else:
        print("❌ ERROR: No se encontró carpeta de imágenes de validación")
        return None
    
    # Crear carpeta de modelos si no existe
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Carpeta de modelos: {MODELS_DIR}")
    
    print("=" * 60)
    return device


def get_recommended_batch_size(device):
    """Recomienda batch size según el hardware disponible"""
    if device == 'cpu':
        return 4
    
    # Obtener VRAM disponible
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if gpu_memory < 4:
        return 4
    elif gpu_memory < 8:
        return 8
    elif gpu_memory < 12:
        return 16
    else:
        return 32


def train_yolo_model(
    model_size='n',      # 'n', 's', 'm', 'l', 'x'
    epochs=200,          # 200 épocas para entrenamiento nocturno
    imgsz=640,
    batch_size=None,     # Auto si es None
    resume=False,        # Continuar entrenamiento previo
    pretrained=True
):
    """
    Inicializa y entrena el modelo YOLOv8 usando el dataset FoodSeg103.
    
    Args:
        model_size: Tamaño del modelo ('n'=nano, 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge)
        epochs: Número de épocas
        imgsz: Tamaño de imagen (debe ser múltiplo de 32)
        batch_size: Tamaño del batch (None = auto)
        resume: Si True, continúa desde el último checkpoint
        pretrained: Si True, usa pesos pre-entrenados
    """
    print("\n" + "=" * 60)
    print("🚀 INICIANDO ENTRENAMIENTO YOLOv8 - MODO NOCTURNO 🌙")
    print("=" * 60)
    
    # 1. Verificar entorno
    device = check_environment()
    if device is None:
        print("\n❌ Entorno no válido. Abortando entrenamiento.")
        return
    
    # 2. Determinar batch size si no se especificó
    if batch_size is None:
        batch_size = get_recommended_batch_size(device)
        print(f"\n📊 Batch size automático: {batch_size}")
    
    # 3. Cargar modelo
    print(f"\n📦 Cargando modelo YOLOv8{model_size}...")
    model_name = f'yolov8{model_size}.pt'
    
    try:
        if resume:
            # Buscar último checkpoint - ✅ CORREGIDO
            last_checkpoint = RUNS_DIR / 'food_detector_train' / 'weights' / 'last.pt'
            if last_checkpoint.exists():
                print(f"🔄 Reanudando desde: {last_checkpoint}")
                model = YOLO(str(last_checkpoint))
            else:
                print("⚠️  No se encontró checkpoint previo. Iniciando desde cero.")
                model = YOLO(model_name)
        else:
            model = YOLO(model_name)
        
        print(f"✅ Modelo cargado: {model_name}")
        
    except Exception as e:
        print(f"❌ ERROR al cargar el modelo: {e}")
        print("Asegúrate de que Ultralytics esté instalado: pip install ultralytics")
        return
    
    # 4. Configuración de entrenamiento
    print(f"\n⚙️  Configuración de entrenamiento:")
    print(f"   • Modelo: YOLOv8{model_size}")
    print(f"   • Épocas: {epochs} 🌙 (modo nocturno)")
    print(f"   • Tamaño de imagen: {imgsz}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Device: {'GPU' if device == 0 else 'CPU'}")
    print(f"   • Dataset: {DATA_YAML}")
    print(f"   • Early stopping: patience=50 (detiene si no mejora)")
    
    # 5. Entrenar
    try:
        print("\n" + "=" * 60)
        print("🏋️  INICIANDO ENTRENAMIENTO...")
        print("💡 TIP: Puedes cerrar esta ventana y el entrenamiento")
        print("    continuará en segundo plano.")
        print("=" * 60 + "\n")
        
        results = model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            
            # Directorios de salida
            project=str(RUNS_DIR),
            name='food_detector_train',
            exist_ok=True,  # Sobrescribir si existe
            
            # Optimizaciones para entrenamiento largo
            patience=50,     # Early stopping si no mejora en 50 épocas
            save=True,       # Guardar checkpoints
            save_period=10,  # Guardar cada 10 épocas (importante para entrenamientos largos)
            
            # Aumentación de datos (mejora generalización)
            mosaic=1.0,      # Probabilidad de mosaic augmentation
            mixup=0.15,      # Aumentado para mejor generalización
            copy_paste=0.1,  # Copy-paste augmentation
            degrees=10.0,    # Rotación random
            translate=0.1,   # Traslación random
            scale=0.5,       # Escala random
            
            # Métricas y logging
            plots=True,      # Generar plots de entrenamiento
            verbose=True,    # Mostrar detalles
            
            # Optimizador (configuración para entrenamiento largo)
            optimizer='AdamW',
            lr0=0.001,       # Learning rate inicial
            lrf=0.01,        # Learning rate final (decae gradualmente)
            momentum=0.937,  # Momentum
            weight_decay=0.0005,  # Weight decay
            warmup_epochs=3.0,    # Warmup epochs
            
            # Workers (multiprocesamiento)
            workers=0 if os.name == 'nt' else 8,  # 0 en Windows para evitar errores
            
            # Cache para acelerar (usa más RAM pero es más rápido)
            cache=False,  # Cambiar a True si tienes >16GB RAM
        )
        
        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO FINALIZADO CON ÉXITO")
        print("=" * 60)
        
        # Copiar mejor modelo a carpeta models/food_detector
        best_model = RUNS_DIR / 'food_detector_train' / 'weights' / 'best.pt'
        if best_model.exists():
            import shutil
            dest_model = MODELS_DIR / 'best.pt'
            shutil.copy(best_model, dest_model)
            print(f"\n📦 Mejor modelo copiado a: {dest_model}")
        
        # Mostrar resultados
        print(f"\n📊 Resultados guardados en: {RUNS_DIR / 'food_detector_train'}")
        print(f"📁 Mejor modelo: {best_model}")
        print(f"📁 Último modelo: {RUNS_DIR / 'food_detector_train' / 'weights' / 'last.pt'}")
        
        # Mostrar métricas finales
        if hasattr(results, 'results_dict'):
            print("\n📈 Métricas finales:")
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                print(f"   • mAP@50: {metrics['metrics/mAP50(B)']:.3f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   • mAP@50-95: {metrics['metrics/mAP50-95(B)']:.3f}")
        
        # Guardar log de finalización
        log_file = RUNS_DIR / 'food_detector_train' / 'training_completed.txt'
        with open(log_file, 'w') as f:
            f.write(f"Entrenamiento completado con éxito\n")
            f.write(f"Épocas: {epochs}\n")
            f.write(f"Modelo: YOLOv8{model_size}\n")
            if hasattr(results, 'results_dict'):
                f.write(f"mAP@50: {metrics.get('metrics/mAP50(B)', 'N/A')}\n")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⚠️  ENTRENAMIENTO INTERRUMPIDO POR EL USUARIO")
        print("Los checkpoints guardados están disponibles en:")
        print(f"{RUNS_DIR / 'food_detector_train' / 'weights'}")
        print("\n💡 Para continuar el entrenamiento, ejecuta de nuevo con resume=True")
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO DURANTE EL ENTRENAMIENTO:")
        print(f"   {type(e).__name__}: {e}")
        print("\n💡 Posibles soluciones:")
        print("   • Verifica que el archivo YAML esté en data/raw/FoodSeg103/yolo_dataset/")
        print("   • Reduce el batch_size si hay error de memoria")
        print("   • Verifica que CUDA esté instalado si usas GPU")
        print("   • Revisa los logs arriba para más detalles")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌙 PREDATOR HELIOS 300 - ENTRENAMIENTO NOCTURNO")
    print("=" * 60)
    print("💡 Configuración optimizada para toda la noche")
    print("⏱️  Tiempo estimado: 8-10 horas")
    print("🔋 IMPORTANTE: Conecta el cargador")
    print("❄️  IMPORTANTE: Asegura buena ventilación")
    print("\n💾 El mejor modelo se guardará en: models/food_detector/best.pt")
    print("📊 Los resultados estarán en: runs/food_detector_train/")
    print("🔄 Se guardarán checkpoints cada 10 épocas")
    print("=" * 60)
    
    response = input("\n👉 Presiona ENTER para iniciar (o 'n' para cancelar): ")
    
    if response.lower() == 'n':
        print("❌ Entrenamiento cancelado")
        sys.exit(0)
    
    print("\n🚀 Iniciando entrenamiento nocturno...")
    print("💤 Puedes irte a dormir, esto estará listo en la mañana\n")
    
    # CONFIGURACIÓN PARA ENTRENAMIENTO NOCTURNO (200 épocas)
    train_yolo_model(
        model_size='n',   # Nano = más rápido pero efectivo
        epochs=100,       # 120 épocas para aprovechar toda la noche
        batch_size=None,  # Automático según tu GPU (probablemente 16-24)
        resume=False      # Cambiar a True si quieres continuar un entrenamiento previo
    )
    
    print("\n" + "=" * 60)
    print("🎉 ¡ENTRENAMIENTO COMPLETADO! Buenos días ☀️")
    print("=" * 60)