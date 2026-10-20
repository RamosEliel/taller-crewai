"""
EJEMPLO 1 (demo principal) - Automatización de generación de reportes con
agentes por roles: ANALISTA -> GENERADOR -> VALIDADOR.

Contexto: "Panadería La Espiga Dorada" (microempresa de Sincelejo, Sucre).
Cada fin de mes la dueña exporta las ventas del POS a un CSV y gasta una
tarde armando el informe. Esta Crew lo produce en minutos.

Uso:
    python ejemplo1_reporte_ventas.py              # con el LLM configurado en .env
    python ejemplo1_reporte_ventas.py --simulado   # ensayo sin IA ni internet
"""
import argparse
import json
import time

from comun import (DATOS, banner, crear_guardrail, obtener_llm, preparar_salidas)
from crewai import Agent, Crew, Process, Task
from herramientas import CalculadoraKPIVentas, calcular_kpis_ventas

NEGOCIO = "Panadería La Espiga Dorada (Sincelejo, Sucre)"
PERIODO = "septiembre de 2026"
RUTA_CSV = (DATOS / "ventas_panaderia_sep2026.csv").as_posix()
SECCIONES = ["Resumen ejecutivo", "Indicadores clave", "Productos y categorías",
             "Comportamiento por día", "Recomendaciones", "Validación"]


def construir_crew(simulado: bool = False) -> Crew:
    kpis = calcular_kpis_ventas(RUTA_CSV)          # cifra de control para el guardrail
    llm = obtener_llm(simulado, respuestas=lambda rol, p: respuestas_simuladas(rol, kpis))
    salidas = preparar_salidas()
    calculadora = CalculadoraKPIVentas()

    # ------------------------------ AGENTES ------------------------------ #
    analista = Agent(
        role="Analista de datos de ventas",
        goal="Obtener cifras exactas de {negocio} para {periodo} e interpretar qué significan para el negocio.",
        backstory=("Eres contador público con 10 años asesorando microempresas de la región Caribe. "
                   "Nunca inventas cifras: siempre usas la calculadora de KPI. Respondes en español."),
        tools=[calculadora], llm=llm, allow_delegation=False, max_iter=5, verbose=True,
    )
    generador = Agent(
        role="Generador de reportes ejecutivos",
        goal="Convertir el análisis en un reporte claro, breve y accionable para la dueña del negocio.",
        backstory=("Eres comunicador organizacional. Escribes para personas sin formación técnica, "
                   "con tablas sencillas y recomendaciones concretas. Respondes en español."),
        llm=llm, allow_delegation=False, max_iter=3, verbose=True,
    )
    validador = Agent(
        role="Validador de calidad de reportes",
        goal="Garantizar que cada cifra del reporte coincida con los datos y que el documento esté completo.",
        backstory=("Eres auditor interno, meticuloso. Recalculas con la herramienta y corriges cualquier "
                   "cifra, sección o recomendación sin sustento. Respondes en español."),
        tools=[calculadora], llm=llm, allow_delegation=False, max_iter=5, verbose=True,
    )

    # ------------------------------- TAREAS ------------------------------- #
    tarea_analisis = Task(
        description=("Usa la herramienta calculadora_kpi_ventas con ruta_csv='{ruta_csv}' para obtener los "
                     "indicadores de {negocio} en {periodo}. Luego interpreta: 3 a 5 hallazgos, 2 alertas "
                     "(productos o días débiles) y 3 oportunidades. Copia las cifras tal cual las entrega "
                     "la herramienta."),
        expected_output=("Análisis en español con: (1) las cifras clave exactas, (2) hallazgos, "
                         "(3) alertas, (4) oportunidades."),
        agent=analista,
        output_file=str(salidas / "ej1_1_analisis.md"),
    )
    tarea_reporte = Task(
        description=("Con el análisis recibido redacta el reporte mensual de {negocio} ({periodo}) en Markdown "
                     "con estas secciones exactas como títulos '##': Resumen ejecutivo; Indicadores clave "
                     "(tabla); Productos y categorías; Comportamiento por día; Recomendaciones (3 a 5, "
                     "concretas y realizables por una microempresa). Máximo 450 palabras."),
        expected_output="Borrador del reporte en Markdown con las 5 secciones indicadas.",
        agent=generador,
        context=[tarea_analisis],
        output_file=str(salidas / "ej1_2_borrador.md"),
    )
    tarea_validacion = Task(
        description=("Revisa el borrador del reporte. Recalcula con calculadora_kpi_ventas (ruta_csv='{ruta_csv}') "
                     "y compara CADA cifra. Corrige errores, completa secciones faltantes y elimina afirmaciones "
                     "sin sustento. Entrega la versión FINAL completa del reporte y agrega al final una sección "
                     "'## Validación' con la lista de verificaciones realizadas y el veredicto (APROBADO)."),
        expected_output=("Reporte final en Markdown con las secciones: " + ", ".join(SECCIONES) + "."),
        agent=validador,
        context=[tarea_analisis, tarea_reporte],
        guardrail=crear_guardrail(kpis["ventas_totales"], SECCIONES, "total de ventas"),
        guardrail_max_retries=3,
        output_file=str(salidas / "ej1_3_reporte_final_ventas.md"),
    )

    # ------------------------------ LA CREW ------------------------------- #
    return Crew(
        agents=[analista, generador, validador],
        tasks=[tarea_analisis, tarea_reporte, tarea_validacion],
        process=Process.sequential,   # analista -> generador -> validador
        verbose=True,
        max_rpm=8,                    # respeta los límites de la capa gratuita
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reporte de ventas con CrewAI (Grupo 4)")
    parser.add_argument("--simulado", action="store_true", help="ensayo sin IA ni internet")
    args = parser.parse_args()

    banner("EJEMPLO 1 · Reporte de ventas con agentes: analista, generador, validador", args.simulado)
    inicio = time.time()
    resultado = construir_crew(args.simulado).kickoff(
        inputs={"negocio": NEGOCIO, "periodo": PERIODO, "ruta_csv": RUTA_CSV}
    )
    print("\n" + "=" * 78 + "\nREPORTE FINAL\n" + "=" * 78)
    print(resultado.raw)
    print("\n" + "-" * 78)
    print(f"Tiempo total: {time.time() - inicio:.1f} s")
    print(f"Tokens usados: {resultado.token_usage}")
    print("Archivos generados en la carpeta salidas/: ej1_1_analisis.md, ej1_2_borrador.md, "
          "ej1_3_reporte_final_ventas.md")


# --------------------------------------------------------------------------- #
#  Respuestas del modo simulado (construidas con los datos reales, sin IA)
# --------------------------------------------------------------------------- #
def respuestas_simuladas(rol: str, k: dict) -> str:
    top = k["top5_productos"]
    cat = k["ventas_por_categoria"]
    tabla_top = "\n".join(f"| {p['producto']} | {p['unidades']} | {p['ventas_fmt']} | {p['participacion_pct']}% |"
                          for p in top)
    tabla_cat = "\n".join(f"- **{c}:** {d['ventas_fmt']} ({d['participacion_pct']}%)" for c, d in cat.items())
    bajos = ", ".join(p["producto"] for p in k["productos_menor_venta"])
    if rol.startswith("Analista"):
        return (f"Cifras exactas de la herramienta:\n```json\n{json.dumps(k, ensure_ascii=False, indent=2)}\n```\n"
                f"Hallazgos: ventas de {k['ventas_totales_fmt']}; el mejor día es {k['mejor_dia_semana']} y el "
                f"más débil {k['peor_dia_semana']}; {top[0]['producto']} lidera; la 2da quincena creció "
                f"{k['variacion_2da_vs_1ra_quincena_pct']}%.\nAlertas: bajo movimiento de {bajos}.\n"
                f"Oportunidades: combos en el día débil, promover pagos digitales, reforzar producción de fin de semana.")
    cuerpo = (
        f"# Reporte mensual de ventas · Panadería La Espiga Dorada · {PERIODO}\n\n"
        f"## Resumen ejecutivo\nEn {PERIODO} la panadería vendió **{k['ventas_totales_fmt']}** en "
        f"{k['dias_con_ventas']} días de operación (promedio diario de {k['venta_promedio_diaria_fmt']}). "
        f"El producto estrella fue **{top[0]['producto']}** y la segunda quincena superó a la primera en "
        f"{k['variacion_2da_vs_1ra_quincena_pct']}%.\n\n"
        f"## Indicadores clave\n| Indicador | Valor |\n|---|---|\n"
        f"| Ventas totales | {k['ventas_totales_fmt']} |\n"
        f"| Unidades vendidas | {k['unidades_vendidas']:,} |\n".replace(",", ".") +
        f"| Venta promedio diaria | {k['venta_promedio_diaria_fmt']} |\n"
        f"| Mejor fecha | {k['mejor_fecha']['fecha']} ({k['mejor_fecha']['ventas_fmt']}) |\n"
        f"| Fecha más baja | {k['peor_fecha']['fecha']} ({k['peor_fecha']['ventas_fmt']}) |\n\n"
        f"## Productos y categorías\n| Producto | Unidades | Ventas | % |\n|---|---|---|---|\n{tabla_top}\n\n"
        f"{tabla_cat}\n\nProductos de menor venta: {bajos}.\n\n"
        f"## Comportamiento por día\nEl mejor día de la semana es **{k['mejor_dia_semana']}** y el más débil "
        f"**{k['peor_dia_semana']}**. Métodos de pago: "
        + ", ".join(f"{m} {v}%" for m, v in k["metodos_de_pago_pct"].items()) + ".\n\n"
        f"## Recomendaciones\n"
        f"1. Lanzar un combo de desayuno ({top[0]['producto']} + Café tinto) los {k['peor_dia_semana'].lower()}.\n"
        f"2. Aumentar la producción del {k['mejor_dia_semana'].lower()} para evitar agotados.\n"
        f"3. Revisar receta, precio o exhibición de: {bajos}.\n"
        f"4. Incentivar pagos con Nequi/Daviplata para agilizar la caja y tener trazabilidad.\n"
    )
    if rol.startswith("Generador"):
        return cuerpo
    return (cuerpo + "\n## Validación\n"
            f"- [x] Total de ventas recalculado con la herramienta: {k['ventas_totales_fmt']} (coincide).\n"
            "- [x] Top 5 de productos y participación por categoría verificados.\n"
            "- [x] Mejor y peor día de la semana verificados.\n"
            "- [x] Las 5 secciones obligatorias están presentes.\n"
            "- [x] Cada recomendación tiene sustento en los datos.\n\n**Veredicto: APROBADO**\n")


if __name__ == "__main__":
    main()
