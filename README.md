# 🍽️ Restaurant Vision Demo

Detección y análisis de ocupación de mesas en restaurantes usando Computer Vision y Deep Learning.

## 📋 Descripción

Este proyecto utiliza YOLO v8 para detectar personas en tiempo real y analizar la ocupación de mesas en un restaurante. El sistema puede procesar videos, detectar clientes sentados, clasificar personal (staff) vs clientes, y generar reportes de ocupación con visualización en tiempo real.

![screen](/figs/screenshot.png)

## ✨ Características

- 🎯 **Detección de personas** con YOLO v8 y análisis de pose
- 👥 **Tracking inteligente** de personas con tolerancia a oclusiones
- 🏠 **Análisis de ocupación** de mesas por ROI (Region of Interest)
- 👔 **Clasificación staff vs clientes** usando patrones de movimiento
- 📊 **Panel de visualización** en tiempo real con estado de mesas
- 📹 **Procesamiento de video** con salida grabada
- 📈 **Exportación de datos** en formato CSV para análisis

## 🏗️ Arquitectura del Sistema

### Estrategia Modular

El proyecto utiliza una arquitectura modular limpia con separación clara de responsabilidades:

```
🎭 main.py           → Orquestador principal (coordinator/recipe)
📋 config.py         → Gestión de configuración y argumentos
📊 data_manager.py   → Manejo de entrada/salida de datos
⚙️ processor.py      → Motor de procesamiento de video
📦 detector/         → Módulos de detección especializados
🧠 logic/           → Lógica de negocio y análisis
🎨 visualization.py  → Renderizado y visualización
```

### Componentes Especializados

#### 🔍 **Detector** (`detector/`)
- `person_detector.py`: Detector principal YOLO v8
- `pose_analyzer.py`: Análisis de poses para clasificación
- `segment_validator.py`: Validación de segmentos de personas

#### 🧠 **Logic** (`logic/`)
- `occupancy_engine.py`: Motor principal de análisis de ocupación
- `mesa_analyzer.py`: Análisis específico por mesa
- `person_classifier.py`: Clasificación staff vs cliente
- `models.py`: Modelos de datos (Mesa, LogicParams, etc.)

#### 🎨 **Visualization**
- Panel translúcido con estado de mesas en tiempo real
- Indicadores visuales de ocupación por colores
- Información de conteo de personas por mesa

## 📁 Estructura de Directorios

```
restaurant-vision-demo/
├── 📄 README.md                    # Documentación principal
├── 🚀 run.sh                       # Script de ejecución
├── 📋 requirements.txt             # Dependencias Python
├── 🤖 yolov8n.pt                   # Modelo YOLO v8 base
├── 🤖 yolov8n-pose.pt             # Modelo YOLO v8 pose
│
├── 📁 app/                         # Código fuente principal
│   ├── 🎭 main.py                  # Orquestador principal
│   ├── 📋 config.py                # Configuración del sistema
│   ├── 📊 data_manager.py          # Gestión de datos E/O
│   ├── ⚙️ processor.py             # Motor de procesamiento
│   ├── 🎨 visualization.py         # Visualización y UI
│   ├── 📦 tracker.py               # Sistema de tracking
│   │
│   ├── 🔍 detector/                # Módulos de detección
│   │   ├── person_detector.py      # Detector YOLO principal
│   │   ├── pose_analyzer.py        # Análisis de poses
│   │   └── segment_validator.py    # Validación de segmentos
│   │
│   ├── 🧠 logic/                   # Lógica de negocio
│   │   ├── occupancy_engine.py     # Motor de ocupación
│   │   ├── mesa_analyzer.py        # Análisis por mesa
│   │   ├── person_classifier.py    # Clasificación personas
│   │   └── models.py               # Modelos de datos
│   │
│   └── detector.py, logic.py       # Wrappers de compatibilidad
│
├── 📁 data/                        # Datos y configuración
│   ├── 🎬 video.mov                # Video de entrada
│   ├── 📐 rois.yaml                # ROIs de mesas (configuración)
│   ├── 📊 events.csv               # Eventos exportados
│   └── 🎥 out.mp4                  # Video procesado de salida
│
├── 📁 env/                         # Entorno virtual Python
└── 📁 tools/                       # Herramientas auxiliares
    └── roi_tagger.py               # Herramienta para definir ROIs
```

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.11+
- OpenCV
- PyTorch
- Ultralytics YOLO v8

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/joelibaceta/restaurant-vision-demo.git
cd restaurant-vision-demo

# 2. Crear entorno virtual
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Los modelos YOLO se descargan automáticamente en la primera ejecución
```

### Ejecución Básica

```bash
# Ejecutar con configuración por defecto
bash run.sh
```

### Ejecución Avanzada

```bash
cd app
python main.py [opciones]
```

#### Opciones Disponibles

```bash
--video PATH          # Video de entrada (default: ../data/video.mov)
--rois PATH           # Archivo YAML con ROIs (default: ../data/rois.yaml)  
--conf FLOAT          # Umbral de confianza YOLO (default: 0.5)
--save_video PATH     # Video de salida (default: ../data/out.mp4)
--events PATH         # Archivo CSV eventos (default: ../data/events.csv)
--display             # Mostrar ventana en tiempo real (default: True)
--no-display         # No mostrar ventana
```

#### Ejemplos de Uso

```bash
# Procesamiento con alta confianza y sin visualización
python main.py --conf 0.8 --no-display

# Usando video personalizado
python main.py --video /path/to/my/video.mp4 --rois /path/to/my/rois.yaml

# Solo generar CSV, sin video de salida
python main.py --save_video "" --events results.csv
```

### Configuración de ROIs (Regiones de Interés)

El archivo `data/rois.yaml` define las mesas y sus polígonos:

```yaml
tables:
- id: '01'
  capacity: 4
  polygon:
    - [156, 375]  # Coordenadas [x, y] de cada vértice
    - [490, 370]
    - [486, 571]
    - [98, 569]
  iop_thr: 0.12   # Umbral de ocupación (opcional)
```

**Herramienta de configuración:**
```bash
cd tools
python roi_tagger.py  # Interfaz gráfica para definir ROIs
```

## 📊 Salidas del Sistema

### Visualización en Tiempo Real
- Panel translúcido con estado actual de todas las mesas
- Indicadores de color: 🔴 ocupada / 🟢 libre
- Conteo de personas por mesa
- Detección de staff vs clientes

### Archivo CSV (`events.csv`)
```csv
frame,time,mesa_id,occupied,people_seated
1,0.066,01,True,2
1,0.066,02,False,0
...
```

### Video Procesado (`out.mp4`)
Video con overlay de detecciones, tracking y panel de estado.

## ⚙️ Configuración Avanzada

### Parámetros de Detección

En `config.py` se pueden ajustar parámetros del sistema:

```python
LogicParams(
    conf_thr=0.5,              # Confianza mínima detección
    min_bbox_frac=0.001,       # Tamaño mínimo bbox (fracción)
    v_thr_px_s=32.0,          # Velocidad máxima (px/s)
    sit_seconds=2.0,          # Tiempo para confirmar "sentado"
    ttl_lost=11.0,            # Tiempo mantener track perdido
    # ... más parámetros
)
```

### Personalización de Visualización

En `visualization.py`:
- Colores y estilos del panel
- Tamaño y transparencia
- Información mostrada

## 📈 Rendimiento

- **FPS típico**: 1.5-2.0 fps (CPU)
- **Precisión**: >90% detección personas sentadas
- **Memoria**: ~2GB RAM
- **GPU**: Opcional, mejora significativamente el rendimiento

## 🔧 Desarrollo

### Estructura Modular

El código está organizado para facilitar el mantenimiento:

1. **Orquestador** (`main.py`): Coordina el flujo principal
2. **Configuración** (`config.py`): Centraliza parámetros
3. **Procesamiento** (`processor.py`): Encapsula la lógica de video
4. **Datos** (`data_manager.py`): Maneja E/O de archivos

### Testing

```bash
# Test básico de componentes
cd app
python test_simple.py --video ../data/video.mov --rois ../data/rois.yaml
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---

## 🎯 Casos de Uso

- **Restaurantes**: Optimizar asignación de mesas y servicio
- **Cafeterías**: Monitorear ocupación en tiempo real  
- **Análisis de flujo**: Estudiar patrones de ocupación
- **Automatización**: Integrar con sistemas de reservas

## 🔮 Roadmap

- [ ] Soporte para múltiples cámaras
- [ ] Dashboard web en tiempo real
- [ ] API REST para integración
- [ ] Análisis predictivo de ocupación
- [ ] Soporte para GPU optimization
- [ ] Mobile app para monitoring