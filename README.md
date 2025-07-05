# 🧠 Ajedrez con Red Neuronal Avanzada e Interfaz Gráfica

## 🎮 Características Principales

### 🖥️ Interfaz Gráfica Mejorada
- **Panel de Control Lateral**: Muestra estadísticas en tiempo real, controles y estado del juego
- **Botones Interactivos**: 
  - ✅ **Nueva Partida**: Reinicia el juego instantáneamente
  - 💾 **Guardar Modelo**: Guarda el progreso de la IA manualmente
  - 📊 **Ver Análisis**: Muestra análisis detallado del progreso de aprendizaje
  - 🤖 **Modo Auto**: Activa/desactiva el juego automático continuo
  - 🚪 **Salir**: Cierra el juego y guarda automáticamente
- **Información en Tiempo Real**:
  - 📈 Estadísticas de victorias/derrotas/empates
  - 🎯 Precisión del modelo
  - 🧠 Tasa de aprendizaje actual
  - ⏱️ Tiempo de partida
  - ♟️ Número de movimientos

### 🤖 Modo Automático
- **Aprendizaje Continuo**: Después de cada jaque mate, automáticamente inicia una nueva partida
- **Entrenamiento Intensivo**: Permite que la IA juegue cientos de partidas sin intervención
- **Tiempo de Espera**: 3 segundos entre partidas para visualizar resultados

### 🎯 Controles del Juego

#### 🖱️ Controles del Mouse
- **Clic Izquierdo**: Seleccionar pieza y realizar movimientos
- **Botones del Panel**: Interactuar con las funciones del juego

#### ⌨️ Controles del Teclado
- **R**: Reiniciar nueva partida
- **M**: Activar/desactivar modo automático
- **A**: Mostrar análisis de progreso en consola
- **S**: Guardar modelo manualmente
- **Q**: Salir del juego

### 🧠 Red Neuronal Avanzada

#### 🏗️ Arquitectura
- **Entrada**: 768 neuronas (8x8x12 - representación completa del tablero)
- **Capas Ocultas**: [512, 256, 128] neuronas con activación Leaky ReLU
- **Salida**: 1 neurona con activación Tanh para evaluación de posición
- **Funciones de Activación**: Leaky ReLU para capas ocultas, Tanh para salida

#### 🎯 Características de Evaluación
- **Balance Material**: Suma del valor de todas las piezas
- **Actividad de Piezas**: Bonificación por piezas en posiciones centrales
- **Seguridad del Rey**: Evaluación de la protección del rey
- **Estructura de Peones**: Análisis de peones doblados, aislados y pasados
- **Control del Centro**: Evaluación del dominio de casillas centrales
- **Fase del Juego**: Identificación automática de apertura, medio juego y final

#### 🔄 Aprendizaje Adaptativos
- **Experience Replay**: Entrena con experiencias pasadas (hasta 10,000 posiciones)
- **Tasa de Aprendizaje Adaptativa**: Decae gradualmente con la experiencia
- **Etiquetado Sofisticado**: Considera características especiales como jaque, amenaza de mate
- **Entrenamiento Multi-Época**: 15 épocas por partida para mejor convergencia

### 📊 Sistema de Análisis de Datos

#### 📁 Archivos CSV Generados
1. **game_statistics.csv**: Estadísticas por partida
   - Timestamp, número de partida, resultado, duración, tasa de aprendizaje
2. **move_analysis.csv**: Análisis detallado de movimientos
   - Evaluación neural vs tradicional, mejor movimiento, coordenadas
3. **position_evaluations.csv**: Evaluaciones de posiciones
   - FEN, evaluación neural, fase del juego, balance material
4. **learning_progress.csv**: Progreso del aprendizaje
   - Pérdida, precisión, tasa de aprendizaje por época

#### 🔍 Funciones de Análisis
- **Análisis de Progreso**: Tendencias de mejora a lo largo del tiempo
- **Exportación de Resumen**: Resumen completo del entrenamiento
- **Estadísticas en Tiempo Real**: Visualización continua del progreso

### 🎨 Mejoras Visuales

#### 🎨 Interfaz
- **Colores Intuitivos**: 
  - 🟡 Pieza seleccionada
  - 🟢 Movimientos legales
  - 🔴 Jaque mate
  - 🟠 Jaque/empate
- **Coordenadas del Tablero**: Letras y números para fácil referencia
- **Hover Effects**: Botones que cambian de color al pasar el mouse

#### 📱 Responsive Design
- **Ventana Expandida**: 950px de ancho (650px tablero + 300px panel)
- **Información Organizada**: Panel lateral con toda la información importante
- **Fuentes Apropiadas**: Diferentes tamaños para mejor legibilidad

## 🚀 Instalación y Uso

### 📋 Requisitos
```bash
pip install pygame chess numpy pandas
```

### 🏃 Ejecución
```bash
python main.py
```

### 🎮 Uso Básico
1. **Jugar**: Haz clic en las piezas blancas para seleccionar y mover
2. **Entrenar**: Juega partidas completas para que la IA aprenda
3. **Modo Automático**: Activa para entrenamiento continuo sin intervención
4. **Análisis**: Usa el botón "Ver Análisis" para revisar el progreso

### 🔧 Configuración Avanzada
- **Tasa de Aprendizaje**: Se adapta automáticamente (0.001 inicial)
- **Tamaño de Replay**: Hasta 10,000 experiencias almacenadas
- **Frecuencia de Guardado**: Cada 3 partidas automáticamente

## 📈 Resultados Esperados

### 🎯 Objetivos de Aprendizaje
- **Mejora Gradual**: La IA debe mejorar su tasa de victoria con el tiempo
- **Adaptación de Estilo**: Aprende diferentes estilos de juego según la oposición
- **Optimización Automática**: Ajusta su evaluación basándose en resultados

### 📊 Métricas de Éxito
- **Tasa de Victoria**: Porcentaje de partidas ganadas por la IA
- **Precisión del Modelo**: Exactitud en la evaluación de posiciones
- **Convergencia**: Estabilización de la pérdida durante el entrenamiento

## 🔮 Características Futuras

### 🎯 Mejoras Planificadas
- **Análisis de Aperturas**: Reconocimiento y evaluación de aperturas específicas
- **Evaluación de Finales**: Conocimiento especializado de finales teóricos
- **Diferentes Niveles**: Múltiples niveles de dificultad
- **Análisis de Partidas**: Replay y análisis de partidas guardadas

### 🎨 Mejoras Visuales
- **Temas Personalizables**: Diferentes colores y estilos de tablero
- **Animaciones**: Movimientos animados de piezas
- **Sonidos**: Efectos sonoros para movimientos y capturas

## 🛠️ Arquitectura Técnica

### 🧩 Componentes Principales
1. **main.py**: Interfaz gráfica, bucle del juego, manejo de eventos
2. **chess_neural_network.py**: Red neuronal, evaluación, entrenamiento
3. **neural_data/**: Directorio con modelos y datos CSV
4. **Images/**: Sprites de las piezas de ajedrez

### 🔗 Flujo de Datos
1. **Jugada del Usuario** → **Actualización del Tablero** → **Evaluación Neural**
2. **Movimiento de IA** → **Entrenamiento** → **Guardado de Datos**
3. **Fin de Partida** → **Análisis** → **Mejora del Modelo**

## 🤝 Contribuciones

Este proyecto está diseñado para ser educativo y demostrar conceptos de:
- **Inteligencia Artificial**: Redes neuronales aplicadas a juegos
- **Aprendizaje Automático**: Aprendizaje por refuerzo y experience replay
- **Análisis de Datos**: Visualización y análisis de progreso
- **Interfaz de Usuario**: Diseño intuitivo y funcional

¡Disfruta viendo cómo la IA aprende y mejora su juego! 🎉
