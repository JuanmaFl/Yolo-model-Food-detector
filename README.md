# IA Chef con Aprendizaje por Refuerzo

Sistema inteligente de detección de ingredientes y generación de recetas personalizadas que combina visión por computadora (YOLOv8), procesamiento de lenguaje natural (GPT-4o-mini) y aprendizaje por refuerzo (Thompson Sampling).

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características Principales](#características-principales)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración](#configuración)
- [Uso](#uso)
- [Dataset](#dataset)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Funcionamiento del Sistema](#funcionamiento-del-sistema)
- [Módulos Principales](#módulos-principales)
- [Limitaciones Conocidas](#limitaciones-conocidas)
- [Mejoras Futuras](#mejoras-futuras)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Licencia](#licencia)

## Descripción General

Este proyecto implementa un asistente culinario inteligente que:

1. Detecta ingredientes en imágenes usando un modelo YOLOv8 entrenado con el dataset FoodSeg103
2. Genera recetas personalizadas utilizando la API de OpenAI (GPT-4o-mini)
3. Aprende de las preferencias del usuario mediante un agente de Aprendizaje por Refuerzo basado en Thompson Sampling

El sistema mejora continuamente sus recomendaciones adaptándose al feedback del usuario, ofreciendo una experiencia cada vez más personalizada.

## Características Principales

### Detección de Ingredientes
- Modelo YOLOv8n entrenado específicamente en 103 clases de alimentos
- Detección en tiempo real con ajuste de umbral de confianza
- Complementación automática con ingredientes básicos cuando la detección es limitada

### Generación de Recetas
- Integración con GPT-4o-mini para generación de recetas contextuales
- Múltiples estilos de cocina: casual, gourmet, saludable, rápido, tradicional
- Tres sugerencias de recetas por consulta

### Aprendizaje Personalizado
- Agente RL basado en Thompson Sampling
- Adaptación dinámica según feedback positivo/negativo
- Persistencia de preferencias entre sesiones
- Estadísticas detalladas del aprendizaje

### Interfaz de Usuario
- Aplicación web desarrollada con Streamlit
- Interfaz intuitiva y responsive
- Visualización de estadísticas del agente
- Configuración ajustable de parámetros de detección

## Arquitectura del Sistema

```
Usuario
   |
   | (Sube imagen)
   v
[Interfaz Streamlit]
   |
   ├──> [Detector YOLOv8] --> Ingredientes detectados
   |         |
   |         └──> (confidence threshold, max_ingredients)
   |
   ├──> [Agente RL Thompson Sampling] --> Selecciona estilo de prompt
   |         |
   |         └──> (alpha/beta parameters)
   |
   └──> [OpenAI GPT-4o-mini] --> Genera recetas
         |
         v
   [Presenta recetas al usuario]
         |
         | (Feedback: like/dislike)
         v
   [Actualiza Agente RL]
```

## Requisitos del Sistema

### Hardware Mínimo
- CPU: Procesador multi-core moderno
- RAM: 8 GB (16 GB recomendado)
- Almacenamiento: 5 GB disponibles

### Hardware Recomendado (para entrenamiento)
- GPU: NVIDIA con 4+ GB VRAM (RTX series o superior)
- RAM: 16 GB
- Almacenamiento: 10 GB disponibles

### Software
- Python 3.8 o superior
- CUDA 11.x o superior (para entrenamiento con GPU)
- pip o conda para gestión de paquetes

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/IA_Cocina_RL.git
cd IA_Cocina_RL
```

### 2. Crear Entorno Virtual

```bash
# Usando venv
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
# Instalar paquetes principales
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install openai
pip install streamlit
pip install python-dotenv
pip install pillow
pip install numpy
pip install pycocotools
```

O usar un archivo requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=tu_clave_api_de_openai
```

### 5. Descargar el Dataset

El dataset FoodSeg103 no está incluido en el repositorio debido a su tamaño. Descárgalo desde:

[Supervisely - FoodSeg103](https://ecosystem.supervisely.com/projects/foodseg103)

Coloca el dataset descargado en:
```
data/raw/FoodSeg103/
```

La estructura debe ser:
```
data/raw/FoodSeg103/
├── meta.json
├── train/
│   ├── ann/
│   └── img/
└── test/
    ├── ann/
    └── img/
```

## Estructura del Proyecto

```
IA_Cocina_RL/
├── src/
│   ├── 01_prepare_data.py          # Conversión del dataset a formato YOLO
│   ├── 02_train_detector.py        # Entrenamiento del modelo YOLOv8
│   ├── app.py                       # Aplicación Streamlit principal
│   ├── detector_module.py           # Módulo de detección de ingredientes
│   ├── openai_chat.py              # Módulo de generación de recetas
│   ├── rl_agent_module.py          # Agente de aprendizaje por refuerzo
│   ├── explore_dataset.py          # Script de exploración del dataset
│   └── debug_bitmap.py             # Utilidad de debugging
├── data/
│   ├── raw/
│   │   └── FoodSeg103/             # Dataset original (no incluido)
│   └── yolo_dataset/                # Dataset convertido (no incluido)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── foodseg_yolo.yaml
├── models/
│   └── food_detector/
│       └── best.pt                  # Modelo entrenado (no incluido)
├── runs/                            # Resultados de entrenamiento
├── uploads/                         # Imágenes subidas por usuarios
├── rl_agent_stats.json             # Estadísticas del agente RL
├── .env                            # Variables de entorno (no incluido)
├── .gitignore
└── README.md
```

## Configuración

### Configuración del Detector

En `src/app.py` puedes ajustar:

```python
conf_threshold = 0.15  # Umbral de confianza (0.05 - 0.50)
max_ingredients = 15   # Máximo de ingredientes a detectar
```

### Configuración del Agente RL

En `src/rl_agent_module.py`:

```python
self.prompt_styles = [
    'casual',      # Recetas cotidianas y fáciles
    'gourmet',     # Recetas elaboradas y sofisticadas
    'saludable',   # Enfoque en alimentación saludable
    'rapido',      # Recetas rápidas (< 30 minutos)
    'tradicional'  # Recetas de cocina tradicional
]
```

### Configuración del Entrenamiento

En `src/02_train_detector.py`:

```python
model_size='n'      # Tamaño del modelo: 'n', 's', 'm', 'l', 'x'
epochs=100          # Número de épocas
batch_size=None     # Auto-detectar según GPU disponible
imgsz=640           # Tamaño de imagen de entrada
```

## Uso

### 1. Preparar el Dataset

Antes de entrenar, convierte el dataset al formato YOLO:

```bash
python src/01_prepare_data.py
```

Este script:
- Lee las anotaciones en formato Supervisely
- Convierte los bitmaps a bounding boxes
- Genera el formato YOLO (archivos .txt)
- Crea el archivo de configuración YAML

Salida esperada:
```
CONVERSIÓN FOODSEG103 → FORMATO YOLO
=======================================
Mapeo de clases cargado. Total de clases: 103.

Procesando: TRAIN
   Procesadas correctamente: 1,033
   Sin imagen: 0
   Sin objetos válidos: 45
   Errores: 0

Procesando: TEST
   Procesadas correctamente: 2,135
   Sin imagen: 0
   Sin objetos válidos: 89
   Errores: 0

Archivo YAML creado: data/yolo_dataset/foodseg_yolo.yaml
¡CONVERSIÓN COMPLETADA!
```

### 2. Entrenar el Modelo

```bash
python src/02_train_detector.py
```

Opciones de configuración:

```python
# Entrenamiento rápido (testing)
model_size='n', epochs=50

# Entrenamiento completo (recomendado)
model_size='n', epochs=100

# Entrenamiento de alta calidad
model_size='s', epochs=200
```

Tiempo estimado de entrenamiento (100 épocas, modelo nano):
- GPU RTX 3060: ~4-6 horas
- GPU RTX 4060: ~3-4 horas
- CPU: ~48+ horas (no recomendado)

### 3. Ejecutar la Aplicación

```bash
streamlit run src/app.py
```

La aplicación se abrirá en `http://localhost:8501`

### 4. Usar la Aplicación

1. **Subir imagen**: Usa el botón de carga para subir una foto de ingredientes
2. **Ajustar parámetros**: Opcionalmente, ajusta el umbral de confianza en la barra lateral
3. **Detectar**: Haz clic en "Detectar Ingredientes y Generar Recetas"
4. **Revisar resultados**: El sistema mostrará ingredientes detectados y 3 recetas sugeridas
5. **Dar feedback**: Usa los botones de like/dislike para mejorar futuras recomendaciones

## Dataset

### FoodSeg103

Dataset de segmentación semántica de alimentos con 103 categorías.

**Estadísticas:**
- Total de imágenes: 3,168
- Imágenes de entrenamiento: 1,033
- Imágenes de validación: 2,135
- Clases: 103 categorías de alimentos
- Formato original: Supervisely (bitmaps RLE)
- Formato convertido: YOLO (bounding boxes)

**Nota sobre la conversión:** 
El dataset original usa segmentación semántica (máscaras pixel-perfect). Este proyecto convierte las máscaras a bounding boxes para detección de objetos con YOLO.

## Entrenamiento del Modelo

### Proceso de Entrenamiento

El script `02_train_detector.py` implementa:

1. **Verificación del entorno**: Detecta GPU/CPU, CUDA, memoria disponible
2. **Carga del modelo**: Utiliza pesos pre-entrenados de COCO
3. **Configuración de hiperparámetros**: Optimizada para el hardware disponible
4. **Entrenamiento**: Con early stopping y guardado periódico de checkpoints
5. **Evaluación**: Genera métricas y gráficos de rendimiento
6. **Exportación**: Guarda el mejor modelo en `models/food_detector/best.pt`

### Hiperparámetros Principales

```python
optimizer='AdamW'
lr0=0.001              # Learning rate inicial
lrf=0.01               # Learning rate final
momentum=0.937
weight_decay=0.0005
warmup_epochs=3.0
patience=50            # Early stopping

# Data augmentation
mosaic=1.0
mixup=0.15
copy_paste=0.1
degrees=10.0
translate=0.1
scale=0.5
```

### Métricas de Evaluación

El modelo se evalúa usando:
- **mAP@50**: Mean Average Precision at IoU threshold 0.50
- **mAP@50-95**: Mean Average Precision across IoU thresholds 0.50-0.95
- **Precision**: Tasa de detecciones correctas
- **Recall**: Tasa de objetos detectados del total

### Resultados Esperados

Con 100 épocas de entrenamiento:
- mAP@50: ~15-25% (debido a la conversión segmentación → detección)
- mAP@50-95: ~8-15%

**Nota:** La precisión relativamente baja se debe a:
1. Conversión de máscaras de segmentación a bounding boxes
2. Alta complejidad del dataset (103 clases)
3. Variabilidad en la calidad de las imágenes

## Funcionamiento del Sistema

### 1. Detección de Ingredientes

```python
# detector_module.py

def detect_ingredients(image_path, conf_threshold=0.15, max_ingredients=15):
    """
    1. Carga el modelo YOLOv8 entrenado
    2. Procesa la imagen de entrada
    3. Aplica umbral de confianza
    4. Elimina duplicados
    5. Limita número de ingredientes
    6. Complementa con ingredientes básicos si es necesario
    """
```

### 2. Selección de Estilo (RL)

```python
# rl_agent_module.py

def select_prompt_style(self):
    """
    Implementa Thompson Sampling:
    1. Muestrea de distribución Beta(alpha, beta) para cada estilo
    2. Selecciona el estilo con mayor valor muestreado
    3. Actualiza contador de uso
    """
```

**Thompson Sampling:**
- Enfoque bayesiano para el problema de multi-armed bandit
- Balance entre exploración y explotación
- Parámetros Beta(alpha, beta) representan éxitos/fracasos

### 3. Generación de Recetas

```python
# openai_chat.py

def generate_recipe_chat(ingredients, prompt_style):
    """
    1. Construye prompt con ingredientes y estilo seleccionado
    2. Envía solicitud a GPT-4o-mini
    3. Recibe y procesa 3 sugerencias de recetas
    4. Retorna texto formateado
    """
```

### 4. Actualización del Agente

```python
# rl_agent_module.py

def update_model(self, feedback_type):
    """
    Actualiza parámetros Beta según feedback:
    - thumbs_up: alpha += 1 (incrementa éxitos)
    - thumbs_down: beta += 1 (incrementa fracasos)
    
    Persiste estadísticas en rl_agent_stats.json
    """
```

## Módulos Principales

### app.py
Aplicación principal de Streamlit. Coordina todos los módulos y maneja:
- Interfaz de usuario
- Carga de imágenes
- Visualización de resultados
- Recolección de feedback
- Estadísticas del agente

### detector_module.py
Módulo de detección de ingredientes:
- Carga y gestión del modelo YOLOv8
- Procesamiento de imágenes
- Filtrado y validación de detecciones
- Complementación con ingredientes básicos

### openai_chat.py
Interfaz con la API de OpenAI:
- Construcción de prompts
- Comunicación con GPT-4o-mini
- Procesamiento de respuestas
- Manejo de errores

### rl_agent_module.py
Agente de aprendizaje por refuerzo:
- Implementación de Thompson Sampling
- Gestión de parámetros Beta
- Persistencia de estadísticas
- Cálculo de métricas de rendimiento

### 01_prepare_data.py
Preprocesamiento del dataset:
- Lectura de anotaciones Supervisely
- Decodificación de bitmaps (Base64 + Zlib + PNG)
- Conversión a bounding boxes
- Generación de archivos YOLO
- Creación de archivo YAML de configuración

### 02_train_detector.py
Pipeline de entrenamiento:
- Verificación de requisitos
- Configuración de hiperparámetros
- Entrenamiento con validación
- Generación de métricas
- Guardado de checkpoints

## Limitaciones Conocidas

### Modelo de Detección
1. **Precisión moderada** (~15% mAP): Debido a la conversión segmentación → detección
2. **Falsos positivos**: El umbral bajo (0.15) puede generar detecciones incorrectas
3. **Clases similares**: Dificultad para distinguir alimentos visualmente similares
4. **Iluminación**: Sensible a condiciones de luz extremas

### Sistema de Recetas
1. **Dependencia de API**: Requiere conexión a internet y créditos de OpenAI
2. **Ingredientes limitados**: Puede no usar todos los ingredientes detectados
3. **Latencia**: Tiempo de respuesta depende de la API de OpenAI

### Agente RL
1. **Período de aprendizaje**: Requiere múltiples interacciones para personalización efectiva
2. **Exploración inicial**: Primeras recomendaciones pueden no ser óptimas
3. **Persistencia local**: Estadísticas se guardan localmente (no compartidas entre usuarios)

## Mejoras Futuras

### Corto Plazo
- [ ] Implementar caching de detecciones frecuentes
- [ ] Agregar más estilos de cocina
- [ ] Mejorar manejo de errores en la UI
- [ ] Implementar logging estructurado
- [ ] Agregar tests unitarios

### Mediano Plazo
- [ ] Entrenar modelo de segmentación completo
- [ ] Implementar base de datos para usuarios
- [ ] Agregar sistema de valoración con estrellas
- [ ] Implementar búsqueda de recetas históricas
- [ ] Agregar exportación de recetas a PDF

### Largo Plazo
- [ ] Despliegue en la nube (AWS/GCP/Azure)
- [ ] Aplicación móvil nativa
- [ ] Modelo de detección multi-idioma
- [ ] Sistema de recomendación colaborativo
- [ ] Integración con bases de datos nutricionales

## Tecnologías Utilizadas

### Visión por Computadora
- **YOLOv8**: Framework de detección de objetos de Ultralytics
- **PyTorch**: Backend de deep learning
- **Pillow**: Procesamiento de imágenes
- **NumPy**: Operaciones numéricas

### Procesamiento de Lenguaje Natural
- **OpenAI API**: GPT-4o-mini para generación de recetas
- **Python-dotenv**: Gestión de variables de entorno

### Aprendizaje por Refuerzo
- **Thompson Sampling**: Algoritmo de bandits multi-brazo
- **NumPy**: Muestreo de distribuciones Beta

### Interfaz de Usuario
- **Streamlit**: Framework de aplicaciones web
- **Markdown**: Formateo de contenido

### Herramientas de Desarrollo
- **Git**: Control de versiones
- **pip**: Gestión de paquetes
- **CUDA**: Aceleración por GPU (opcional)

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para preguntas, sugerencias o reportar problemas, por favor abre un issue en el repositorio de GitHub.

## Agradecimientos

- Dataset FoodSeg103 por Supervisely
- Ultralytics por el framework YOLOv8
- OpenAI por la API de GPT-4o-mini
- Streamlit por el framework de UI

---

**Nota:** Este proyecto fue desarrollado con fines educativos y de investigación. El rendimiento del modelo puede variar según las condiciones de uso y la calidad de las imágenes de entrada.
