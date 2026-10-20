"""
Herramientas (Tools) personalizadas para los agentes de CrewAI.

Principio de diseño: "la IA redacta, el código calcula".
Todos los números (totales, promedios, cantidades a pedir) se calculan con
Python de forma determinista; los agentes los consultan a través de estas
herramientas y se concentran en interpretar, redactar y validar.
"""
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def formato_cop(valor: float) -> str:
    """Formatea un valor en pesos colombianos: 1234567 -> '$1.234.567'."""
    return "$" + f"{round(valor):,}".replace(",", ".")


# --------------------------------------------------------------------------- #
#  EJEMPLO 1: KPI de ventas de la panadería
# --------------------------------------------------------------------------- #
def calcular_kpis_ventas(ruta_csv: str) -> dict:
    """Lee el CSV de ventas y calcula los indicadores clave del periodo."""
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de ventas: {ruta}")

    total = 0.0
    unidades = 0
    registros = 0
    por_producto = defaultdict(lambda: {"unidades": 0, "ventas": 0.0})
    por_categoria = defaultdict(float)
    por_dia_semana = defaultdict(float)
    dias_por_semana = defaultdict(set)
    por_fecha = defaultdict(float)
    por_pago = defaultdict(float)
    quincena = {"1ra quincena (1-15)": 0.0, "2da quincena (16-fin)": 0.0}

    with ruta.open(encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            cantidad = int(fila["cantidad"])
            valor = cantidad * float(fila["precio_unitario"])
            f = date.fromisoformat(fila["fecha"])
            total += valor
            unidades += cantidad
            registros += 1
            por_producto[fila["producto"]]["unidades"] += cantidad
            por_producto[fila["producto"]]["ventas"] += valor
            por_categoria[fila["categoria"]] += valor
            por_dia_semana[DIAS[f.weekday()]] += valor
            dias_por_semana[DIAS[f.weekday()]].add(f)
            por_fecha[f] += valor
            por_pago[fila["metodo_pago"]] += valor
            quincena["1ra quincena (1-15)" if f.day <= 15 else "2da quincena (16-fin)"] += valor

    if registros == 0:
        raise ValueError("El archivo de ventas está vacío.")

    ranking = sorted(por_producto.items(), key=lambda kv: kv[1]["ventas"], reverse=True)
    mejor_fecha = max(por_fecha, key=por_fecha.get)
    peor_fecha = min(por_fecha, key=por_fecha.get)
    promedio_dia_semana = {
        d: por_dia_semana[d] / len(dias_por_semana[d]) for d in DIAS if d in por_dia_semana
    }
    q1, q2 = quincena.values()

    return {
        "periodo": f"{min(por_fecha).isoformat()} a {max(por_fecha).isoformat()}",
        "dias_con_ventas": len(por_fecha),
        "ventas_totales": round(total),
        "ventas_totales_fmt": formato_cop(total),
        "unidades_vendidas": unidades,
        "registros_de_venta": registros,
        "venta_promedio_diaria_fmt": formato_cop(total / len(por_fecha)),
        "valor_promedio_por_registro_fmt": formato_cop(total / registros),
        "top5_productos": [
            {"producto": p, "unidades": d["unidades"], "ventas_fmt": formato_cop(d["ventas"]),
             "participacion_pct": round(100 * d["ventas"] / total, 1)}
            for p, d in ranking[:5]
        ],
        "productos_menor_venta": [
            {"producto": p, "unidades": d["unidades"], "ventas_fmt": formato_cop(d["ventas"]),
             "participacion_pct": round(100 * d["ventas"] / total, 1)}
            for p, d in ranking[-3:]
        ],
        "ventas_por_categoria": {
            c: {"ventas_fmt": formato_cop(v), "participacion_pct": round(100 * v / total, 1)}
            for c, v in sorted(por_categoria.items(), key=lambda kv: kv[1], reverse=True)
        },
        "promedio_por_dia_de_semana": {d: formato_cop(v) for d, v in promedio_dia_semana.items()},
        "mejor_dia_semana": max(promedio_dia_semana, key=promedio_dia_semana.get),
        "peor_dia_semana": min(promedio_dia_semana, key=promedio_dia_semana.get),
        "mejor_fecha": {"fecha": mejor_fecha.isoformat(), "ventas_fmt": formato_cop(por_fecha[mejor_fecha])},
        "peor_fecha": {"fecha": peor_fecha.isoformat(), "ventas_fmt": formato_cop(por_fecha[peor_fecha])},
        "metodos_de_pago_pct": {
            m: round(100 * v / total, 1)
            for m, v in sorted(por_pago.items(), key=lambda kv: kv[1], reverse=True)
        },
        "ventas_por_quincena": {k: formato_cop(v) for k, v in quincena.items()},
        "variacion_2da_vs_1ra_quincena_pct": round(100 * (q2 - q1) / q1, 1),
    }


class EntradaRutaCSV(BaseModel):
    """Esquema de entrada de las herramientas: la ruta del archivo CSV."""
    ruta_csv: str = Field(..., description="Ruta del archivo CSV a analizar.")


class CalculadoraKPIVentas(BaseTool):
    name: str = "calculadora_kpi_ventas"
    description: str = (
        "Calcula con exactitud los indicadores de ventas (total, promedio diario, "
        "top de productos, categorías, días de la semana, métodos de pago y "
        "quincenas) a partir de un CSV de ventas. Devuelve JSON. Úsala SIEMPRE "
        "en lugar de calcular cifras de memoria."
    )
    args_schema: Type[BaseModel] = EntradaRutaCSV

    def _run(self, ruta_csv: str) -> str:
        return json.dumps(calcular_kpis_ventas(ruta_csv), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
#  EJEMPLO 2: inventario y reabastecimiento de la ferretería
# --------------------------------------------------------------------------- #
def analizar_inventario(ruta_csv: str, semanas_cobertura: int = 2) -> dict:
    """
    Detecta productos bajo el stock mínimo y calcula la cantidad sugerida:
        cantidad = stock_minimo + consumo_semanal * semanas_cobertura - stock_actual
    """
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de inventario: {ruta}")

    criticos, en_riesgo, sanos = [], [], 0
    por_proveedor = defaultdict(lambda: {"items": [], "subtotal": 0.0})

    with ruta.open(encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            stock, minimo = int(f["stock_actual"]), int(f["stock_minimo"])
            consumo, costo = int(f["consumo_semanal"]), float(f["costo_unitario"])
            cobertura = round(stock / consumo, 1) if consumo else None
            if stock < minimo:
                cantidad = minimo + consumo * semanas_cobertura - stock
                subtotal = cantidad * costo
                item = {
                    "codigo": f["codigo"], "producto": f["producto"],
                    "stock_actual": stock, "stock_minimo": minimo,
                    "consumo_semanal": consumo, "semanas_de_cobertura_actual": cobertura,
                    "cantidad_sugerida": cantidad, "costo_unitario_fmt": formato_cop(costo),
                    "subtotal": round(subtotal), "subtotal_fmt": formato_cop(subtotal),
                    "proveedor": f["proveedor"],
                }
                criticos.append(item)
                por_proveedor[f["proveedor"]]["items"].append(f["codigo"])
                por_proveedor[f["proveedor"]]["subtotal"] += subtotal
            elif stock < minimo * 1.2:
                en_riesgo.append({"codigo": f["codigo"], "producto": f["producto"],
                                  "stock_actual": stock, "stock_minimo": minimo,
                                  "semanas_de_cobertura_actual": cobertura})
            else:
                sanos += 1

    total = sum(p["subtotal"] for p in por_proveedor.values())
    criticos.sort(key=lambda i: i["semanas_de_cobertura_actual"] or 0)
    return {
        "regla": f"cantidad = stock_minimo + consumo_semanal x {semanas_cobertura} - stock_actual",
        "productos_criticos": criticos,
        "productos_en_riesgo": en_riesgo,
        "productos_sin_novedad": sanos,
        "ordenes_por_proveedor": {
            prov: {"codigos": d["items"], "subtotal_fmt": formato_cop(d["subtotal"])}
            for prov, d in por_proveedor.items()
        },
        "total_orden": round(total),
        "total_orden_fmt": formato_cop(total),
    }


class AnalizadorInventario(BaseTool):
    name: str = "analizador_inventario"
    description: str = (
        "Analiza un CSV de inventario: identifica productos por debajo del stock "
        "mínimo, productos en riesgo, la cantidad sugerida a pedir, subtotales por "
        "proveedor y el total de la orden. Devuelve JSON con cifras exactas."
    )
    args_schema: Type[BaseModel] = EntradaRutaCSV

    def _run(self, ruta_csv: str) -> str:
        return json.dumps(analizar_inventario(ruta_csv), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
#  TALLER (Día 2): reseñas de clientes del restaurante
# --------------------------------------------------------------------------- #
def resumir_resenas(ruta_csv: str) -> dict:
    """Cuenta reseñas por calificación y canal, calcula el promedio y lista las negativas."""
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de reseñas: {ruta}")
    with ruta.open(encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    if not filas:
        raise ValueError("El archivo de reseñas está vacío.")
    notas = [int(f["calificacion"]) for f in filas]
    por_canal = defaultdict(list)
    for f in filas:
        por_canal[f["canal"]].append(int(f["calificacion"]))
    return {
        "total_resenas": len(filas),
        "promedio": round(sum(notas) / len(notas), 2),
        "distribucion_estrellas": {str(e): notas.count(e) for e in range(5, 0, -1)},
        "promedio_por_canal": {c: round(sum(v) / len(v), 2) for c, v in por_canal.items()},
        "positivas_4_5": sum(n >= 4 for n in notas),
        "neutras_3": notas.count(3),
        "negativas_1_2": sum(n <= 2 for n in notas),
        "comentarios_negativos": [
            {"fecha": f["fecha"], "canal": f["canal"], "calificacion": int(f["calificacion"]),
             "comentario": f["comentario"]}
            for f in filas if int(f["calificacion"]) <= 2
        ],
        "comentarios_positivos": [f["comentario"] for f in filas if int(f["calificacion"]) >= 4],
    }


class ResumenResenas(BaseTool):
    name: str = "resumen_resenas"
    description: str = (
        "Lee un CSV de reseñas de clientes y devuelve en JSON: total, promedio de "
        "estrellas, distribución, promedio por canal y los comentarios negativos y positivos."
    )
    args_schema: Type[BaseModel] = EntradaRutaCSV

    def _run(self, ruta_csv: str) -> str:
        return json.dumps(resumir_resenas(ruta_csv), ensure_ascii=False, indent=2)
