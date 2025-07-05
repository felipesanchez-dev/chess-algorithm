# 🧠 Ajedrez con Red Neuronal Avanzada e Interfaz Gráfica

Un sistema completo de ajedrez con inteligencia artificial que utiliza redes neuronales profundas para aprender y mejorar automáticamente jugando partidas.

## 📋 Tabla de Contenidos

- [🎮 Características Principales](#-características-principales)
- [🧠 Arquitectura de la Red Neuronal](#-arquitectura-de-la-red-neuronal)
- [🎯 Diagramas del Sistema](#-diagramas-del-sistema)
- [🚀 Instalación y Uso](#-instalación-y-uso)
- [📊 Sistema de Análisis de Datos](#-sistema-de-análisis-de-datos)
- [🔧 Configuración Avanzada](#-configuración-avanzada)
- [📈 Resultados Esperados](#-resultados-esperados)
- [🛠️ Arquitectura Técnica](#-arquitectura-técnica)

## 🎮 Características Principales

### 🖥️ Interfaz Gráfica Completa
- **Panel de Control Lateral (300px)**: Estadísticas en tiempo real y controles
- **Tablero Interactivo (650px)**: Visualización con coordenadas y piezas
- **Sistema de Coronación**: Menú interactivo para promoción de peones
- **Botones Inteligentes**: Con efectos hover y funcionalidad específica

### 🤖 Sistema de IA Avanzado
- **Red Neuronal Profunda**: 4 capas con 400,000+ parámetros
- **Aprendizaje Automático**: Mejora continua después de cada partida
- **Evaluación Híbrida**: Combina análisis neural y tradicional
- **Experience Replay**: Reutiliza hasta 10,000 experiencias previas

### 🎯 Controles y Funcionalidades

#### 🖱️ Controles del Mouse
- **Selección de Piezas**: Clic izquierdo para seleccionar piezas blancas
- **Movimientos Legales**: Casillas verdes muestran movimientos válidos
- **Coronación Interactiva**: Menú emergente para promoción de peones
- **Botones del Panel**: Interacción con todas las funciones

#### ⌨️ Controles del Teclado
- **R**: Reiniciar nueva partida
- **M**: Activar/desactivar modo automático
- **A**: Mostrar análisis de progreso
- **S**: Guardar modelo manualmente
- **Q**: Salir del juego
- **Q/R/B/N**: Coronación rápida (Dama/Torre/Alfil/Caballo)

## 🧠 Arquitectura de la Red Neuronal

### 🏗️ Estructura de la Red

```
┌─────────────────────────────────────────────────────────────────┐
│                    RED NEURONAL AVANZADA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                   INPUT LAYER (768 neuronas)                    │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │         Representación del Tablero: 8x8x12 = 768       │  │
│    │         • 8x8 casillas del tablero                     │  │
│    │         • 12 tipos de piezas (6 blancas + 6 negras)    │  │
│    │         • Codificación one-hot para cada pieza         │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                 │                               │
│                                 ▼                               │
│                  HIDDEN LAYER 1 (512 neuronas)                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │         Activación: Leaky ReLU (α=0.01)                │  │
│    │         Función: f(x) = max(0.01x, x)                  │  │
│    │         Inicialización: Xavier/He Normal                │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                 │                               │
│                                 ▼                               │
│                  HIDDEN LAYER 2 (256 neuronas)                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │         Activación: Leaky ReLU (α=0.01)                │  │
│    │         Extracción de patrones tácticos                │  │
│    │         Reconocimiento de estructuras                  │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                 │                               │
│                                 ▼                               │
│                  HIDDEN LAYER 3 (128 neuronas)                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │         Activación: Leaky ReLU (α=0.01)                │  │
│    │         Análisis estratégico de alto nivel              │  │
│    │         Evaluación de posiciones complejas             │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                 │                               │
│                                 ▼                               │
│                    OUTPUT LAYER (1 neurona)                    │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │         Activación: Tanh                               │  │
│    │         Rango: [-1, +1]                                │  │
│    │         • -1: Posición muy mala para blancas           │  │
│    │         •  0: Posición equilibrada                     │  │
│    │         • +1: Posición muy buena para blancas          │  │
│    └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```


### 🔢 Parámetros del Modelo

| Componente | Cantidad | Descripción |
|------------|----------|-------------|
| **Pesos Totales** | 409,345 | Conexiones entre neuronas |
| **Bias** | 897 | Términos de sesgo |
| **Parámetros Totales** | 410,242 | Parámetros entrenable |
| **Capas** | 4 | 1 entrada + 2 ocultas + 1 salida |
| **Conexiones** | 768→512→256→128→1 | Arquitectura completamente conectada |

### 🎯 Funciones de Activación

#### Leaky ReLU (Capas Ocultas)
```
f(x) = max(0.01x, x)

Gráfico:
     │
   1 │     ╱
     │    ╱
     │   ╱
   0 │──╱──────────
     │ ╱
-0.01│╱
     └──────────────
    -1  0    1
```

#### Tanh (Capa de Salida)
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Gráfico:
   1 │  ████████
     │ ██     ██
     │██       ██
   0 │──██───────██──
     │   ██     ██
     │    ████████
  -1 │
     └─────────────────
    -2    0    2
```


## 🎯 Diagramas del Sistema

### 🔄 Flujo de Aprendizaje

```
┌─────────────────────────────────────────────────────────────────┐
│                      CICLO DE APRENDIZAJE                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    JUEGO    │    │ RECOLECCIÓN │    │ENTRENAMIENTO│    │   MEJORA    │
│   ACTIVO    │───►│    DATOS    │───►│ RED NEURONAL│───►│   MODELO    │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│• Jugador    │    │• Posiciones │    │• Backprop   │    │• Pesos      │
│  mueve      │    │• Evaluación │    │• 15 épocas  │    │  actualizados│
│• IA evalúa  │    │• Movimientos│    │• Experience │    │• Estadísticas│
│• IA mueve   │    │• Resultado  │    │  Replay     │    │• Guardado   │
│• Repite     │    │• Contexto   │    │• Gradientes │    │• CSV        │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```


### 📊 Sistema de Evaluación Híbrida

```
┌─────────────────────────────────────────────────────────────────┐
│                EVALUACIÓN HÍBRIDA DE MOVIMIENTOS                │
└─────────────────────────────────────────────────────────────────┘

                    POSICIÓN DEL TABLERO
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        ┌─────────────┐         ┌─────────────┐
        │ EVALUACIÓN  │         │ EVALUACIÓN  │
        │   NEURAL    │         │TRADICIONAL  │
        │             │         │             │
        │• Red Neural │         │• Alfabeta   │
        │• Patrones   │         │• Material   │
        │• Aprende    │         │• Posición   │
        └─────────────┘         └─────────────┘
                │                       │
                │       PESOS           │
                │     ADAPTATIVOS       │
                │                       │
                ▼                       ▼
        ┌─────────────────────────────────────────┐
        │         COMBINACIÓN INTELIGENTE         │
        │                                         │
        │     Score = (w₁ × Neural) +             │
        │             (w₂ × Tradicional)          │
        │                                         │
        │     donde w₁ + w₂ = 1                   │
        │     w₁ aumenta con experiencia          │
        └─────────────────────────────────────────┘
                            │
                            ▼
                    MEJOR MOVIMIENTO
```


### 🧩 Análisis de Características

```
┌─────────────────────────────────────────────────────────────────┐
│                  EXTRACCIÓN DE CARACTERÍSTICAS                  │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │MATERIAL │     │POSICIÓN │     │TÁCTICA  │
    │         │     │         │     │         │
    │• Peones │     │• Centro │     │• Jaque  │
    │• Piezas │     │• Reyes  │     │• Mate   │
    │• Damas  │     │• Castillo│     │• Clavada│
    └─────────┘     └─────────┘     └─────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │    VECTOR DE ENTRADA    │
            │                         │
            │  [0.2, 0.8, 0.1, 0.9,   │
            │   0.3, 0.7, 0.4, 0.6,   │
            │   0.5, 0.5, 0.0, 1.0,   │
            │   ... 768 valores ...]  │
            └─────────────────────────┘
                         │
                         ▼
                  RED NEURONAL
```


### 🔄 Experience Replay

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXPERIENCE REPLAY                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PARTIDA 1  │    │  PARTIDA 2  │    │  PARTIDA N  │
│             │    │             │    │             │
│ Pos₁ → Mov₁ │    │ Pos₁ → Mov₁ │    │ Pos₁ → Mov₁ │
│ Pos₂ → Mov₂ │    │ Pos₂ → Mov₂ │    │ Pos₂ → Mov₂ │
│ Pos₃ → Mov₃ │    │ Pos₃ → Mov₃ │    │ Pos₃ → Mov₃ │
│    ...      │    │    ...      │    │    ...      │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
            ┌─────────────────────────┐
            │    BUFFER DE MEMORIA    │
            │                         │
            │ ┌─────┬─────┬─────┬───┐ │
            │ │Exp₁ │Exp₂ │Exp₃ │...│ │
            │ └─────┴─────┴─────┴───┘ │
            │                         │
            │   Capacidad: 10,000     │
            │   Política: FIFO        │
            └─────────────────────────┘
                           │
                           ▼
            ┌─────────────────────────┐
            │     ENTRENAMIENTO       │
            │                         │
            │ • Selección aleatoria   │
            │ • Batch de 32 ejemplos  │
            │ • 5 épocas por batch    │
            │ • Mejora continua       │
            └─────────────────────────┘
```

## 🚀 Instalación y Uso

### 📋 Requisitos del Sistema

```bash
# Dependencias principales
pip install pygame==2.6.1
pip install python-chess==1.999
pip install numpy==1.24.3
pip install pandas==2.0.3

# Opcional para análisis avanzado
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
```

### 🛠️ Instalación

```bash
# Clonar o descargar el proyecto
cd Chess-algoritmo

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el juego
python main.py
```

### 🎮 Guía de Uso

#### 🎯 Juego Básico
- **Iniciar**: Ejecuta `python main.py`
- **Seleccionar**: Haz clic en una pieza blanca
- **Mover**: Haz clic en una casilla verde resaltada
- **Coronar**: Usa el menú emergente para promoción de peones
- **Observar**: La IA aprende y mejora automáticamente

#### 🤖 Modo Automático
- **Activar**: Presiona el botón "Modo Auto" o tecla 'M'
- **Observar**: La IA jugará partidas continuas
- **Aprender**: Se reinicia automáticamente después de cada partida
- **Monitorear**: Observa las estadísticas en tiempo real

#### 👑 Sistema de Coronación
- **Automático**: Aparece menú cuando un peón llega al final
- **Opciones**: Dama (Q), Torre (R), Alfil (B), Caballo (N)
- **Teclas rápidas**: Q/R/B/N para selección rápida
- **Por defecto**: Dama si no se selecciona
## 📊 Sistema de Análisis de Datos

### 📁 Archivos CSV Generados

#### 1. game_statistics.csv
```csv
timestamp,game_number,result,moves_count,duration,avg_evaluation,learning_rate,win_rate
2024-01-01T10:00:00,1,1,45,120.5,0.32,0.001000,100.0
2024-01-01T10:02:30,2,-1,38,98.2,-0.18,0.000995,50.0
...
```

#### 2. move_analysis.csv
```csv
timestamp,game_number,move_number,from_square,to_square,piece_moved,piece_captured,neural_evaluation,traditional_evaluation,combined_evaluation,was_best_move
2024-01-01T10:00:15,1,1,e2,e4,P,,0.12,0.15,0.129,true
2024-01-01T10:00:18,1,2,e7,e5,p,,−0.08,−0.12,−0.092,false
...
```

#### 3. position_evaluations.csv
```csv
timestamp,fen,neural_evaluation,traditional_evaluation,game_phase,material_balance,piece_activity
2024-01-01T10:00:00,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0.00,0.00,0,0,0.5
...
```

#### 4. learning_progress.csv
```csv
timestamp,epoch,loss,accuracy,learning_rate,total_games,win_rate
2024-01-01T10:00:00,0,0.456,65.2,0.001000,1,100.0
2024-01-01T10:00:01,5,0.234,72.8,0.001000,1,100.0
...
```

### 📈 Métricas de Rendimiento

| Métrica | Descripción | Rango Esperado |
|---------|-------------|----------------|
| **Tasa de Victoria** | % de partidas ganadas | 0-100% |
| **Precisión** | % de evaluaciones correctas | 60-90% |
| **Pérdida** | Error del modelo | 0.1-0.5 |
| **Tiempo por Movimiento** | Milisegundos | 50-200ms |
| **Experiencias Almacenadas** | Posiciones en memoria | 0-10,000 |
## 🔧 Configuración Avanzada

### ⚙️ Parámetros de la Red Neuronal

```python
# En chess_neural_network.py
class AdvancedChessNeuralNetwork:
    def __init__(self, 
                 input_size=768,           # Tamaño de entrada
                 hidden_layers=[512, 256, 128],  # Capas ocultas
                 output_size=1,            # Salida
                 learning_rate=0.001):     # Tasa de aprendizaje
```

### 🎛️ Configuración del Aprendizaje

| Parámetro | Valor por Defecto | Descripción |
|-----------|------------------|-------------|
| `learning_rate` | 0.001 | Tasa de aprendizaje inicial |
| `learning_decay` | 0.995 | Factor de decaimiento |
| `min_learning_rate` | 0.0001 | Mínima tasa de aprendizaje |
| `max_replay_size` | 10,000 | Tamaño máximo del buffer |
| `batch_size` | 32 | Tamaño del lote de entrenamiento |
| `epochs_per_game` | 15 | Épocas por partida |

### 🎨 Personalización Visual

```python
# En main.py - Colores personalizables
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)      # Pieza seleccionada
LIGHT_GREEN = (144, 238, 144)  # Movimientos legales
RED = (255, 0, 0)           # Jaque mate
ORANGE = (255, 165, 0)      # Jaque/empate
```

### 🧩 Estructura del Proyecto

```
Chess-algoritmo/
├── main.py                    # Interfaz gráfica y lógica principal
├── chess_neural_network.py    # Red neuronal y aprendizaje
├── README.md                  # Documentación completa
├── requirements.txt           # Dependencias del proyecto
├── Images/                    # Sprites de las piezas
│   ├── wK.png, wQ.png, wR.png, wB.png, wN.png, wP.png
│   └── bK.png, bQ.png, bR.png, bB.png, bN.png, bP.png
└── neural_data/              # Datos de entrenamiento
    ├── advanced_chess_model.json
    ├── experience_replay.json
    ├── game_statistics.csv
    ├── move_analysis.csv
    ├── position_evaluations.csv
    ├── learning_progress.csv
    └── training_summary.json
```

### 🔗 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DEL SISTEMA                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INTERFAZ      │    │   LÓGICA DE     │    │   INTELIGENCIA  │
│   GRÁFICA       │◄──►│     JUEGO       │◄──►│   ARTIFICIAL    │
│                 │    │                 │    │                 │
│• Pygame         │    │• Tablero        │    │• Red Neuronal   │
│• Botones        │    │• Movimientos    │    │• Evaluación     │
│• Tablero        │    │• Reglas         │    │• Aprendizaje    │
│• Animaciones    │    │• Validación     │    │• Experience     │
│• Menús          │    │• Estado         │    │  Replay         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EVENTOS       │    │   DATOS DE      │    │   PERSISTENCIA  │
│                 │    │   ANÁLISIS      │    │                 │
│• Mouse          │    │• Estadísticas   │    │• Modelo         │
│• Teclado        │    │• Movimientos    │    │• Experiencias   │
│• Temporizador   │    │• Evaluaciones   │    │• Configuración  │
│• Ventana        │    │• Progreso       │    │• Logs           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 📡 Flujo de Datos

```
    ENTRADA DEL USUARIO
            │
            ▼
    ┌─────────────────┐
    │   VALIDACIÓN    │
    │   MOVIMIENTO    │
    └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │   ACTUALIZAR    │
    │    TABLERO      │
    └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │   EVALUACIÓN    │
    │   POSICIÓN      │
    └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │   GUARDAR       │
    │    DATOS        │
    └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │   ENTRENAMIENTO │
    │   RED NEURONAL  │
    └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │   ACTUALIZAR    │
    │    MODELO       │
    └─────────────────┘
```
## 🔮 Características Futuras

### 🎯 Roadmap de Desarrollo

#### Versión 2.0 - Análisis Avanzado
- [ ] **Análisis de Aperturas**: Base de datos de aperturas clásicas
- [ ] **Evaluación de Finales**: Conocimiento teórico de finales
- [ ] **Análisis de Partidas**: Replay con evaluación paso a paso
- [ ] **Diferentes Niveles**: Múltiples niveles de dificultad

#### Versión 2.1 - Mejoras Visuales
- [ ] **Temas Personalizables**: Diferentes estilos de tablero
- [ ] **Animaciones**: Movimientos suaves de piezas
- [ ] **Efectos Sonoros**: Audio para movimientos y capturas
- [ ] **Modo 3D**: Tablero en tres dimensiones

#### Versión 2.2 - Multijugador
- [ ] **Juego Online**: Partidas contra otros jugadores
- [ ] **Torneos**: Sistema de competición
- [ ] **Ranking**: Sistema de puntuación ELO
- [ ] **Chat**: Comunicación entre jugadores

#### Versión 3.0 - IA Avanzada
- [ ] **Redes Convolucionales**: CNN para reconocimiento de patrones
- [ ] **Attention Mechanisms**: Mecanismos de atención
- [ ] **Transfer Learning**: Transferencia de conocimiento
- [ ] **Ensemble Methods**: Combinación de múltiples modelos

## 📚 Referencias y Agradecimientos

### 🧠 Inspiración Técnica
- **AlphaZero**: Arquitectura de autoaprendizaje
- **Stockfish**: Motor de ajedrez tradicional
- **Leela Chess Zero**: Red neuronal para ajedrez
- **DeepMind**: Investigación en IA para juegos

### 📖 Librerías Utilizadas
- **pygame**: Interfaz gráfica y manejo de eventos
- **python-chess**: Lógica del juego de ajedrez
- **numpy**: Operaciones matemáticas y matrices
- **pandas**: Análisis y manipulación de datos

### 🤝 Contribuciones
Este proyecto es de código abierto y educativo. Contribuciones bienvenidas:

- 🐛 Reportar bugs
- 💡 Sugerir mejoras
- 🔧 Añadir funcionalidades
- 📝 Mejorar documentación

## 🎉 Conclusión

Este proyecto demuestra la aplicación práctica de:

- **Redes Neuronales Profundas** en juegos de estrategia
- **Aprendizaje por Refuerzo** con experience replay
- **Análisis de Datos** para el seguimiento del progreso
- **Interfaces Gráficas** intuitivas y funcionales
- **Arquitecturas Híbridas** que combinan técnicas tradicionales y modernas

¡Disfruta viendo cómo la inteligencia artificial aprende y domina el juego más antiguo del mundo! 🎯♟️🧠

---

## 🚀 Instalación y Uso

### 📋 Requisitos del Sistema

```bash
# Dependencias principales
pip install pygame==2.6.1
pip install python-chess==1.999
pip install numpy==1.24.3
pip install pandas==2.0.3

# Opcional para análisis avanzado
pip install matplotlib==3.7.1
pip install seaborn==0.12.2

# Clonar o descargar el proyecto
cd Chess-algoritmo

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el juego
python main.py

```


### 📊 Métricas de Éxito

#### 🏆 Objetivos a Corto Plazo (10 partidas)
- [ ] Tasa de victoria > 30%
- [ ] Precisión > 65%
- [ ] Pérdida < 0.5
- [ ] Tiempo de respuesta < 200ms

#### 🎯 Objetivos a Mediano Plazo (50 partidas)
- [ ] Tasa de victoria > 60%
- [ ] Precisión > 80%
- [ ] Pérdida < 0.3
- [ ] Reconocimiento de patrones básicos

#### 🚀 Objetivos a Largo Plazo (100+ partidas)
- [ ] Tasa de victoria > 80%
- [ ] Precisión > 90%
- [ ] Pérdida < 0.2
- [ ] Juego estratégico avanzado
