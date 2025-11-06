# IA_Cocina_RL/src/openai_chat.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_recipe_chat(ingredients: list, prompt_style: str) -> str:
    """
    Envía la lista de ingredientes y un estilo de prompt a la API de OpenAI
    para generar ideas de recetas en formato de chat.
    """
    if not ingredients:
        return "¡Vaya! No detecté ingredientes. Por favor, sube una foto más clara."

    ingredients_str = ", ".join(ingredients)
    
    # El prompt_style se inyecta desde el Agente RL para personalizar la experiencia
    system_prompt = f"""Eres un chef de IA experto en crear recetas prácticas. Tu tarea es generar 3 ideas de recetas distintas usando los ingredientes proporcionados y siguiendo el estilo: '{prompt_style}'.

Para cada una de las 3 recetas, DEBES incluir:
1.  **Título:** Un nombre atractivo para la receta.
2.  **Porciones:** Indica para cuántas personas es la receta (Asume 2 personas).
3.  **Lista de Ingredientes:** Proporciona una lista detallada de ingredientes con **cantidades exactas** (ej: 100g, 1 taza, 2 cucharadas) para las 2 porciones.
4.  **Pasos:** Una lista numerada y detallada de los pasos de preparación."""

    user_message = f"Mis ingredientes son: {ingredients_str}."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # O 'gpt-4' para mayor calidad
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7 
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error al conectar con la API de OpenAI: {e}"

if __name__ == '__main__':
    # Ejemplo de prueba
    test_ingredients = ["tomate", "cebolla", "queso", "pasta"]
    test_style = "recetas rápidas y económicas"
    print(f"Generando receta con estilo: {test_style}")
    response = generate_recipe_chat(test_ingredients, test_style)
    print(response)