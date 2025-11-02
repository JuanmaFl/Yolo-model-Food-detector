# IA_Cocina_RL/src/rl_agent_module.py
"""
Módulo de Agente de Aprendizaje por Refuerzo (Thompson Sampling)
Maneja la selección de estilos de prompt y aprende de feedback del usuario
"""
import json
import numpy as np
from pathlib import Path

# Ruta del archivo de estadísticas
STATS_FILE = Path('rl_agent_stats.json')

class ThompsonSamplingAgent:
    """
    Agente de RL que usa Thompson Sampling para seleccionar el mejor estilo de prompt
    """
    
    def __init__(self):
        """Inicializa el agente con estilos de prompt disponibles"""
        self.prompt_styles = [
            'casual',
            'gourmet', 
            'saludable',
            'rapido',
            'tradicional'
        ]
        
        # Inicializar parámetros Beta para Thompson Sampling
        # alpha = éxitos + 1, beta = fracasos + 1
        self.alpha = {style: 1 for style in self.prompt_styles}
        self.beta = {style: 1 for style in self.prompt_styles}
        
        # Cargar estadísticas previas si existen
        self.stats = self._load_stats()
        
    def _load_stats(self):
        """
        Carga estadísticas previas desde archivo JSON
        
        Returns:
            dict: Estadísticas cargadas o dict vacío si hay error
        """
        if not STATS_FILE.exists():
            print("ℹ️  No se encontró archivo de estadísticas. Creando uno nuevo.")
            return self._create_empty_stats()
        
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Verificar que el archivo no esté vacío
                if not content:
                    print("⚠️  Archivo de estadísticas vacío. Creando nuevo.")
                    return self._create_empty_stats()
                
                stats = json.loads(content)
                
                # Validar estructura
                if not isinstance(stats, dict):
                    print("⚠️  Formato de estadísticas inválido. Creando nuevo.")
                    return self._create_empty_stats()
                
                # Cargar alpha y beta si existen
                if 'alpha' in stats and 'beta' in stats:
                    self.alpha = stats['alpha']
                    self.beta = stats['beta']
                
                print(f"✅ Estadísticas cargadas: {stats.get('total_interactions', 0)} interacciones")
                return stats
                
        except json.JSONDecodeError as e:
            print(f"⚠️  Error al leer JSON: {e}")
            print("   Creando archivo nuevo.")
            return self._create_empty_stats()
        except Exception as e:
            print(f"⚠️  Error inesperado al cargar stats: {e}")
            return self._create_empty_stats()
    
    def _create_empty_stats(self):
        """
        Crea estructura de estadísticas vacía
        
        Returns:
            dict: Estadísticas inicializadas
        """
        stats = {
            'total_interactions': 0,
            'total_likes': 0,
            'total_dislikes': 0,
            'alpha': {style: 1 for style in self.prompt_styles},
            'beta': {style: 1 for style in self.prompt_styles},
            'style_counts': {style: 0 for style in self.prompt_styles}
        }
        
        # Guardar inmediatamente
        self._save_stats(stats)
        return stats
    
    def _save_stats(self, stats=None):
        """
        Guarda las estadísticas en archivo JSON
        
        Args:
            stats: Dict de estadísticas a guardar (usa self.stats si es None)
        """
        if stats is None:
            stats = self.stats
        
        try:
            with open(STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Error al guardar estadísticas: {e}")
    
    def select_prompt_style(self):
        """
        Selecciona un estilo de prompt usando Thompson Sampling
        
        Returns:
            str: Estilo de prompt seleccionado
        """
        # Muestrear de la distribución Beta para cada estilo
        samples = {}
        for style in self.prompt_styles:
            samples[style] = np.random.beta(
                self.alpha[style],
                self.beta[style]
            )
        
        # Seleccionar el estilo con el mayor sample
        selected_style = max(samples, key=samples.get)
        
        # Actualizar contador de uso
        if 'style_counts' not in self.stats:
            self.stats['style_counts'] = {style: 0 for style in self.prompt_styles}
        
        self.stats['style_counts'][selected_style] = \
            self.stats['style_counts'].get(selected_style, 0) + 1
        
        return selected_style
    
    def update_model(self, feedback_type):
        """
        Actualiza el modelo basado en el feedback del usuario
        
        Args:
            feedback_type: 'thumbs_up' o 'thumbs_down'
        """
        # Obtener el último estilo usado
        if 'style_counts' not in self.stats:
            self.stats['style_counts'] = {style: 0 for style in self.prompt_styles}
        
        last_style = max(
            self.stats['style_counts'],
            key=self.stats['style_counts'].get
        )
        
        # Actualizar parámetros Beta según feedback
        if feedback_type == 'thumbs_up':
            self.alpha[last_style] += 1
            self.stats['total_likes'] = self.stats.get('total_likes', 0) + 1
        elif feedback_type == 'thumbs_down':
            self.beta[last_style] += 1
            self.stats['total_dislikes'] = self.stats.get('total_dislikes', 0) + 1
        
        # Actualizar contador total
        self.stats['total_interactions'] = self.stats.get('total_interactions', 0) + 1
        
        # Guardar alpha y beta actualizados
        self.stats['alpha'] = self.alpha
        self.stats['beta'] = self.beta
        
        # Persistir cambios
        self._save_stats()
        
        print(f"✅ Modelo actualizado: {feedback_type} para estilo '{last_style}'")
    
    def get_stats(self):
        """
        Retorna estadísticas actuales del agente
        
        Returns:
            dict: Estadísticas con métricas adicionales calculadas
        """
        stats = self.stats.copy()
        
        # Calcular estilo con mejor desempeño
        best_style = None
        best_ratio = 0
        
        for style in self.prompt_styles:
            alpha = self.alpha[style]
            beta = self.beta[style]
            
            # Calcular tasa de éxito esperada
            expected_success = alpha / (alpha + beta)
            
            if expected_success > best_ratio:
                best_ratio = expected_success
                best_style = style
        
        stats['best_style'] = best_style
        stats['best_ratio'] = best_ratio
        
        # Agregar tasas individuales
        stats['style_performance'] = {}
        for style in self.prompt_styles:
            alpha = self.alpha[style]
            beta = self.beta[style]
            stats['style_performance'][style] = {
                'expected_success': alpha / (alpha + beta),
                'alpha': alpha,
                'beta': beta
            }
        
        return stats
    
    def reset_stats(self):
        """Reinicia todas las estadísticas del agente"""
        self.alpha = {style: 1 for style in self.prompt_styles}
        self.beta = {style: 1 for style in self.prompt_styles}
        self.stats = self._create_empty_stats()
        print("🔄 Estadísticas reiniciadas")


if __name__ == "__main__":
    # Prueba del módulo
    print("=" * 60)
    print("🧪 PRUEBA DEL AGENTE RL")
    print("=" * 60)
    
    agent = ThompsonSamplingAgent()
    
    print("\n📊 Estadísticas iniciales:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if key != 'style_performance':
            print(f"   • {key}: {value}")
    
    print("\n🎲 Seleccionando estilos (5 veces):")
    for i in range(5):
        style = agent.select_prompt_style()
        print(f"   {i+1}. {style}")
    
    print("\n👍 Simulando feedback positivo...")
    agent.update_model('thumbs_up')
    
    print("\n👎 Simulando feedback negativo...")
    agent.update_model('thumbs_down')
    
    print("\n📊 Estadísticas finales:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if key != 'style_performance':
            print(f"   • {key}: {value}")
    
    print("\n✅ Prueba completada")