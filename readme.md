# 🛰️ Simulador Multi-UAV para Misiones SAR — Entorno y Generación de Escenarios

Este proyecto implementa un **motor de simulación discreta** para misiones de **búsqueda y rescate (SAR)** utilizando múltiples vehículos aéreos no tripulados (UAVs).
El sistema genera entornos aleatorios, valida su factibilidad energética y temporal, y ejecuta simulaciones *tick a tick* que replican el comportamiento cooperativo de una flota de drones.

Forma parte de la **Etapa 4.1 — Modelado del entorno y generación de escenarios** de la tesis:

> *“Modelo de aprendizaje por refuerzo multiagente (MARL) para coordinación de UAVs en misiones SAR en entornos complejos.”*

---

## 📂 Estructura general del proyecto

```
tesis-envgen/
│
├── config.json                # Archivo de configuración global del entorno
├── instances/                 # Carpeta de salida para escenarios generados
│   ├── inst_train_000_.../    # Carpeta individual de simulación (mapa + reportes + snapshots)
│   ├── index.csv              # Índice consolidado de escenarios validados
│
├── envgen/
│   ├── cli.py                 # CLI principal (controlador de generación + simulación)
│   ├── config.py              # Cargador de parámetros desde config.json
│   ├── sampling.py            # Muestreo de dimensiones, densidades y POIs
│   ├── obstacles.py           # Generación Bernoulli de obstáculos
│   ├── base.py                # Selección de la celda base en el perímetro
│   ├── pois.py                # Colocación y atributos de puntos de interés
│   ├── energy.py              # Cálculo de factibilidad energética
│   ├── gridsearch.py          # BFS/A* para conectividad
│   ├── qa.py                  # Validación QA (conectividad, viabilidad, distribución)
│   ├── persist.py             # Guardado en .npz, .json y actualización de index.csv
│   ├── viz.py                 # Funciones de visualización (mapas, energía, trayectorias)
│   └── sim_engine/            # Motor temporal de simulación
│       ├── engine.py          # simulate_episode(): núcleo discreto del simulador
│       ├── entities.py        # Clases UAV, POI, BaseStation
│       ├── planner.py         # Planificación de rutas (BFS + greedy)
│       └── utils.py           # Utilidades temporales y espaciales
│
└── README.md                  # Este documento
```

---

## ⚙️ Dependencias

El proyecto está desarrollado en **Python 3.10+** y requiere las siguientes librerías principales:

```
pip install numpy matplotlib tqdm
```

Opcionalmente, puedes instalar paquetes de soporte para análisis y visualización avanzada:

```
pip install pandas seaborn scikit-learn
```

💡 **Sugerencia:** crea un entorno virtual con Anaconda o venv para aislar dependencias:

```
conda create -n envgen python=3.10
conda activate envgen
```

---

## 🚀 Ejecución paso a paso

### 1. Generar instancias del entorno

Este comando crea mapas aleatorios, valida su conectividad y guarda las instancias en la carpeta `/instances/`:

```
python -m envgen.cli --config config.json --n-train 1 --n-val 1 --plot
```

### 2. Ejecutar simulación temporal (motor discreto)

Para correr una simulación completa en los escenarios validados:

```
python -m envgen.cli --config config.json --simulate --plot --mission-report
```

Esto generará carpetas individuales con:

* `snap_t000.png`, `snap_t100.png`, … → snapshots cada 100 ticks
* `map_...png` → mapa base con POIs y obstáculos
* `mission_...json` → reporte final de la misión
* `index.csv` → resumen general de métricas por instancia

### 3. Modo solo validación QA

Si deseas verificar únicamente la conectividad y viabilidad sin correr simulaciones:

```
python -m envgen.cli --config config.json --qa-only
```

---

## 📈 Ejemplo de salida de misión

```
[simulate] ticks=613 | served=26/26 | violations=10 | RTB=0
```

| Métrica          | Descripción                       | Valor               |
| ---------------- | --------------------------------- | ------------------- |
| `ticks_used`     | Duración total de la simulación   | 613                 |
| `served / total` | POIs atendidos / totales          | 26 / 26             |
| `violations`     | Ventanas temporales excedidas     | 10                  |
| `n_rtb`          | Retornos a base                   | 0                   |
| `energy_spent`   | Energía total consumida (por UAV) | 174.35 u / 173.20 u |

---

## 🧠 Fundamento del modelo

El simulador combina **procesos estocásticos y deterministas**:

* **Obstáculos:** muestreados mediante un proceso *Bernoulli(p_obs)*.
* **POIs:** generados con densidad ajustable y atributos aleatorios (prioridad, duración, ventana temporal).
* **Energía:** evaluada con umbrales dinámicos (E_{\text{max}}, E_{\text{reserve}}).
* **RTB:** activado cuando (E \le e_{\text{move,ortho}},d(\text{pos,base}) + E_{\text{reserve}}).
* **Snapshots:** guardados cada 100 ticks y figura final con trayectorias completas.

La arquitectura completa constituye el entorno base para entrenar políticas MARL en la **Etapa 4.2: Aprendizaje multiagente**.


