# IA_Cocina_RL/src/app.py
import streamlit as st
from detector_module import detect_ingredients, get_model_info
from openai_chat import generate_recipe_chat
from rl_agent_module import ThompsonSamplingAgent
import os
import time
from pathlib import Path

# --- CONFIGURACIÓN ---
UPLOADS_DIR = Path('uploads')
UPLOADS_DIR.mkdir(exist_ok=True)
# ---------------------

# Inicializar Agente RL en el estado de Streamlit
if 'rl_agent' not in st.session_state:
    st.session_state.rl_agent = ThompsonSamplingAgent()
if 'recipe_response' not in st.session_state:
    st.session_state.recipe_response = None
if 'ingredients_list' not in st.session_state:
    st.session_state.ingredients_list = None
if 'current_style' not in st.session_state:
    st.session_state.current_style = None
if 'model_loaded' not in st.session_state:
    # Verificar que el modelo esté disponible al inicio
    model_info = get_model_info()
    st.session_state.model_loaded = model_info['loaded']
    st.session_state.model_info = model_info

def handle_feedback(feedback_type):
    """Maneja el feedback del usuario y actualiza el Agente RL."""
    st.session_state.rl_agent.update_model(feedback_type)
    st.toast("¡Gracias por tu feedback! El Chef IA ha aprendido algo nuevo.", icon='🧠')
    
    # Limpiar la receta actual para forzar una nueva recomendación
    st.session_state.recipe_response = None
    st.session_state.ingredients_list = None

# Configuración de la página
st.set_page_config(
    page_title="IA Chef con Aprendizaje", 
    layout="centered",
    page_icon="🍳"
)

# Título y descripción
st.title("🍳 IA Chef Personalizada con RL")
st.markdown("""
**Detector de Ingredientes con YOLOv8** entrenado en 103 clases de comida  
Sube una foto de tu nevera o ingredientes y obtén recetas personalizadas.
""")

# Verificar estado del modelo
if not st.session_state.model_loaded:
    st.error("⚠️ **Modelo de detección no encontrado**")
    st.warning(f"""
    El modelo entrenado no está disponible. Por favor, entrena el modelo primero:
    
    ```bash
    python src/02_train_detector.py
    ```
    
    Ruta esperada: `models/food_detector/best.pt`
    """)
    st.stop()

# Mostrar info del modelo en sidebar
with st.sidebar:
    st.header("ℹ️ Información del Sistema")
    
    if st.session_state.model_loaded:
        st.success("✅ Modelo de detección cargado")
        
        with st.expander("📊 Detalles del Modelo"):
            info = st.session_state.model_info
            st.write(f"**Clases:** {info['num_classes']}")
            st.write(f"**Device:** {info['device']}")
            st.write(f"**Modelo:** YOLOv8n")
            
            # Mostrar algunas clases
            st.write("**Ejemplos de clases detectables:**")
            sample_classes = info['classes'][:15]
            for cls in sample_classes:
                st.write(f"- {cls}")
    
    st.divider()
    
    # Configuración de detección
    st.subheader("⚙️ Configuración")
    conf_threshold = st.slider(
        "Umbral de confianza",
        min_value=0.05,
        max_value=0.50,
        value=0.15,
        step=0.05,
        help="Menor = más detecciones (puede incluir falsos positivos)"
    )
    
    max_ingredients = st.slider(
        "Máximo de ingredientes",
        min_value=5,
        max_value=20,
        value=15,
        step=1
    )
    
    st.divider()
    
    # Estadísticas del agente RL
    st.subheader("🤖 Estadísticas del Agente RL")
    agent_stats = st.session_state.rl_agent.get_stats()
    
    st.metric("Interacciones Totales", agent_stats['total_interactions'])
    st.metric("👍 Likes", agent_stats['total_likes'])
    st.metric("👎 Dislikes", agent_stats['total_dislikes'])
    
    if agent_stats['best_style']:
        st.write(f"**Mejor estilo:** {agent_stats['best_style']}")

# Área principal
st.divider()

uploaded_file = st.file_uploader(
    "📸 Sube una imagen de tus ingredientes",
    type=['jpg', 'jpeg', 'png'],
    help="Sube una foto clara de tus ingredientes o de tu nevera"
)

if uploaded_file is not None:
    # 1. Guardar y mostrar la imagen
    file_path = UPLOADS_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(uploaded_file, caption='Ingredientes Subidos', use_column_width=True)
    
    with col2:
        st.info("""
        **Tips:**
        - Foto con buena iluminación
        - Ingredientes visibles
        - No muy lejos ni muy cerca
        """)
    
    st.divider()
    
    if st.button("🔍 Detectar Ingredientes y Generar Recetas", key='generate_btn', type='primary', use_container_width=True):
        
        # 2. Detección de Ingredientes
        with st.spinner('🔍 Detectando ingredientes con YOLOv8...'):
            progress_bar = st.progress(0)
            
            ingredients = detect_ingredients(
                str(file_path),
                conf_threshold=conf_threshold,
                max_ingredients=max_ingredients
            )
            
            progress_bar.progress(50)
            st.session_state.ingredients_list = ingredients
            progress_bar.progress(100)
            time.sleep(0.3)
            progress_bar.empty()

        if ingredients and "Error" in ingredients[0]:
            st.error(ingredients[0])
            st.stop()
        
        # Mostrar ingredientes detectados
        st.success(f"✅ **{len(ingredients)} ingredientes detectados:**")
        
        # Mostrar en columnas
        cols = st.columns(3)
        for idx, ingredient in enumerate(ingredients):
            with cols[idx % 3]:
                st.write(f"🥘 {ingredient}")
        
        st.divider()

        # 3. Generación de recetas con RL
        with st.spinner('🤖 Aplicando Aprendizaje por Refuerzo...'):
            prompt_style = st.session_state.rl_agent.select_prompt_style()
            st.session_state.current_style = prompt_style
            
            st.info(f"✨ **Estilo seleccionado por RL:** {prompt_style}")
            time.sleep(0.5)

        with st.spinner('👨‍🍳 Generando recetas personalizadas con GPT-4o-mini...'):
            recipe_text = generate_recipe_chat(ingredients, prompt_style)
            st.session_state.recipe_response = recipe_text
            
    # 4. Mostrar la Receta y Feedback
    if st.session_state.recipe_response:
        st.divider()
        st.subheader("👨‍🍳 Recetas Sugeridas")
        
        # Mostrar receta en un contenedor con estilo
        with st.container():
            st.markdown(st.session_state.recipe_response)
        
        st.divider()
        
        # Sección de feedback
        st.subheader("💬 ¿Qué te parecieron las recetas?")
        st.write("Tu feedback ayuda al Chef IA a mejorar sus recomendaciones")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("👍 Me Gustó", key='like_btn', use_container_width=True):
                handle_feedback('thumbs_up')
                st.balloons()
        
        with col2:
            if st.button("👎 No Me Gustó", key='dislike_btn', use_container_width=True):
                handle_feedback('thumbs_down')
        
        with col3:
            if st.button("🔄 Generar Nuevas Recetas", key='regenerate_btn', use_container_width=True):
                st.session_state.recipe_response = None
                st.rerun()

else:
    # Instrucciones cuando no hay imagen
    st.info("""
    ### 📝 Cómo usar:
    
    1. **Sube una imagen** de tus ingredientes o tu nevera
    2. **Ajusta la configuración** en la barra lateral si lo deseas
    3. **Haz clic en "Detectar"** para que YOLOv8 identifique los ingredientes
    4. **Recibe recetas personalizadas** generadas por GPT-4o-mini
    5. **Da feedback** para que el sistema aprenda tus preferencias
    
    ⚠️ **Nota:** Debido a que el modelo tiene precisión moderada (~15% mAP),
    puede que no detecte todos los ingredientes o detecte algunos incorrectamente.
    El sistema complementará con ingredientes básicos según sea necesario.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
🤖 <b>Sistema de IA Chef</b><br>
Detector: YOLOv8n (FoodSeg103) | Generador: GPT-4o-mini | Agente: Thompson Sampling<br>
Desarrollado con Streamlit
</div>
""", unsafe_allow_html=True)