# CrewAI · Automatización de procesos por roles — Grupo 4

Electiva Profesional en TI/SI · CECAR · Exposición 15 y 16 de octubre de 2026
Integrantes: Luis Acuña Rodríguez · Sebastián Sierra Torres · Sergio Guzmán Barragán

## Qué contiene

| Archivo | Qué hace |
|---|---|
| `ejemplo1_reporte_ventas.py` | **Demo principal.** Crew de 3 agentes (analista → generador → validador) que produce el reporte mensual de ventas de la *Panadería La Espiga Dorada*. |
| `ejemplo2_inventario_flow.py` | **Flow con decisión.** Lee el inventario de la *Ferretería El Progreso*; un `@router` decide si hay que reabastecer. Si sí, una Crew genera y valida la orden de compra. |
| `taller_plantilla.py` | Plantilla con `TODO` para el **taller evaluable del Día 2**. |
| `herramientas.py` | Tools personalizadas (`BaseTool`): calculan las cifras con Python. |
| `comun.py` | Configuración del LLM, guardrails reutilizables y modo simulado. |
| `datos/` | CSV de ventas (1.064 registros, sep-2026), inventario (12 productos) y reseñas (18). |

Principio de diseño: **"la IA redacta, el código calcula"**. Las cifras salen de herramientas
deterministas y un *guardrail* rechaza cualquier reporte cuyo total no coincida con los datos;
si falla, CrewAI devuelve el error al agente y este reintenta (hasta 3 veces).

## Instalación (una sola vez, ~3 minutos)

Requisito: **Python 3.10 a 3.13** (CrewAI 1.15 aún no soporta 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Abra `.env` y pegue su clave gratuita de Google AI Studio o de Groq.

> Importante: Google apaga `gemini-2.5-flash` el **16 de octubre de 2026**. Este proyecto ya usa
> `gemini/gemini-3.6-flash`. Si su cuenta no lo tiene, cambie `MODEL` a `gemini/gemini-3.5-flash-lite`.

## Ejecución

```bash
python ejemplo1_reporte_ventas.py
python ejemplo2_inventario_flow.py
python ejemplo2_inventario_flow.py --csv datos/inventario_ferreteria_ok.csv
```

Los resultados quedan en `salidas/` (análisis, borrador y reporte final validado).

### Modo simulado (plan B)

Si en clase no hay internet o se agota la cuota gratuita, agregue `--simulado` a cualquier comando.
CrewAI orquesta igual (agentes, tareas, herramientas, guardrails, Flow y router), pero las
respuestas de los agentes son textos preparados a partir de los datos, **sin IA**. Si lo usan,
díganlo abiertamente al público.