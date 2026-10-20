"""
EJEMPLO 2 - Flow de CrewAI con decisión automática: control de inventario y
orden de compra para "Ferretería El Progreso" (microempresa de Corozal, Sucre).

Arquitectura (Flow + Crew):
    @start   cargar_inventario   -> Python puro: lee el CSV y calcula faltantes
    @router  decidir             -> "reabastecer" si hay productos bajo el mínimo,
                                    "sin_novedad" en caso contrario
    @listen  generar_orden       -> Crew de 3 agentes (analista de inventario,
                                    generador de órdenes, validador)
    @listen  informe_sin_novedad -> mensaje corto, sin gastar tokens
    @listen  cerrar              -> guarda el resultado y muestra el resumen

Uso:
    python ejemplo2_inventario_flow.py                       # inventario con faltantes
    python ejemplo2_inventario_flow.py --csv datos/inventario_ferreteria_ok.csv
    python ejemplo2_inventario_flow.py --simulado            # ensayo sin IA
    python ejemplo2_inventario_flow.py --plot                # diagrama HTML del flujo
"""
import argparse
import shutil
from pathlib import Path

from comun import BASE, DATOS, banner, crear_guardrail, obtener_llm, preparar_salidas
from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, or_, router, start
from herramientas import AnalizadorInventario, analizar_inventario
from pydantic import BaseModel

NEGOCIO = "Ferretería El Progreso (Corozal, Sucre)"
SECCIONES = ["Diagnóstico", "Orden de compra", "Total", "Validación"]


class EstadoInventario(BaseModel):
    """Estado estructurado que viaja por todos los pasos del Flow."""
    ruta_csv: str = (DATOS / "inventario_ferreteria.csv").as_posix()
    simulado: bool = False
    criticos: int = 0
    en_riesgo: int = 0
    total_orden: int = 0
    resultado: str = ""
    archivo_salida: str = ""


class FlujoInventario(Flow[EstadoInventario]):

    @start()
    def cargar_inventario(self):
        datos = analizar_inventario(self.state.ruta_csv)
        self.state.criticos = len(datos["productos_criticos"])
        self.state.en_riesgo = len(datos["productos_en_riesgo"])
        self.state.total_orden = datos["total_orden"]
        print(f"\n[Flow] Inventario leído: {self.state.criticos} críticos, "
              f"{self.state.en_riesgo} en riesgo, {datos['productos_sin_novedad']} sin novedad.")

    @router(cargar_inventario, emit=["reabastecer", "sin_novedad"])
    def decidir(self):
        if self.state.criticos > 0:
            print("[Flow] Decisión del router: REABASTECER")
            return "reabastecer"
        print("[Flow] Decisión del router: SIN_NOVEDAD")
        return "sin_novedad"

    @listen("reabastecer")
    def generar_orden(self):
        crew = construir_crew(self.state.simulado, self.state.ruta_csv)
        resultado = crew.kickoff(inputs={"negocio": NEGOCIO, "ruta_csv": self.state.ruta_csv})
        self.state.resultado = resultado.raw
        self.state.archivo_salida = "ej2_orden_compra_validada.md"

    @listen("sin_novedad")
    def informe_sin_novedad(self):
        self.state.resultado = (f"# {NEGOCIO}\n\nInventario revisado: no hay productos por debajo del "
                                f"stock mínimo. Productos en riesgo a vigilar: {self.state.en_riesgo}.\n"
                                "No se genera orden de compra (no se consumieron tokens del LLM).")
        self.state.archivo_salida = "ej2_inventario_sin_novedad.md"

    @listen(or_(generar_orden, informe_sin_novedad))
    def cerrar(self):
        destino = preparar_salidas() / self.state.archivo_salida
        destino.write_text(self.state.resultado, encoding="utf-8")
        print("\n" + "=" * 78 + "\nRESULTADO DEL FLOW\n" + "=" * 78)
        print(self.state.resultado)
        print("-" * 78 + f"\nGuardado en: salidas/{destino.name}")
        return self.state.resultado


def construir_crew(simulado: bool, ruta_csv: str) -> Crew:
    datos = analizar_inventario(ruta_csv)     # cifras de control para el guardrail
    llm = obtener_llm(simulado, respuestas=lambda rol, p: respuestas_simuladas(rol, datos))
    salidas = preparar_salidas()
    analizador = AnalizadorInventario()

    analista = Agent(
        role="Analista de inventario",
        goal="Identificar con cifras exactas qué productos de {negocio} deben reabastecerse y por qué.",
        backstory=("Llevas años manejando bodegas de ferreterías en Sucre. Siempre usas el analizador "
                   "de inventario y nunca estimas cantidades de memoria. Respondes en español."),
        tools=[analizador], llm=llm, allow_delegation=False, max_iter=5, verbose=True,
    )
    generador = Agent(
        role="Generador de órdenes de compra",
        goal="Redactar órdenes de compra claras, agrupadas por proveedor, listas para enviar.",
        backstory=("Eres el auxiliar administrativo de la ferretería. Redactas documentos formales, "
                   "breves y sin errores. Respondes en español."),
        llm=llm, allow_delegation=False, max_iter=3, verbose=True,
    )
    validador = Agent(
        role="Validador de compras",
        goal="Verificar que cantidades, subtotales y total de la orden coincidan con el inventario.",
        backstory=("Eres el revisor financiero: ningún peso sale sin verificación. Recalculas con el "
                   "analizador y corriges cualquier diferencia. Respondes en español."),
        tools=[analizador], llm=llm, allow_delegation=False, max_iter=5, verbose=True,
    )

    t_diagnostico = Task(
        description=("Usa analizador_inventario con ruta_csv='{ruta_csv}'. Explica qué productos de {negocio} "
                     "están bajo el mínimo, cuántas semanas de cobertura les quedan y cuáles están en riesgo."),
        expected_output="Diagnóstico con la lista de productos críticos y en riesgo, con cifras exactas.",
        agent=analista, output_file=str(salidas / "ej2_1_diagnostico.md"),
    )
    t_orden = Task(
        description=("Redacta en Markdown la orden de compra con los títulos '##': Diagnóstico (2 líneas); "
                     "Orden de compra (una tabla por proveedor: código, producto, cantidad, costo unitario, "
                     "subtotal); Total (valor total de la orden)."),
        expected_output="Orden de compra en Markdown agrupada por proveedor.",
        agent=generador, context=[t_diagnostico], output_file=str(salidas / "ej2_2_orden_borrador.md"),
    )
    t_validacion = Task(
        description=("Recalcula con analizador_inventario (ruta_csv='{ruta_csv}') y verifica cada cantidad, "
                     "subtotal y el total. Corrige lo necesario y entrega la orden FINAL completa, agregando "
                     "al final '## Validación' con las verificaciones hechas y el veredicto."),
        expected_output="Orden de compra final con secciones: " + ", ".join(SECCIONES) + ".",
        agent=validador, context=[t_diagnostico, t_orden],
        guardrail=crear_guardrail(datos["total_orden"], SECCIONES, "total de la orden"),
        guardrail_max_retries=3,
    )
    return Crew(agents=[analista, generador, validador], tasks=[t_diagnostico, t_orden, t_validacion],
                process=Process.sequential, verbose=True, max_rpm=8)


def respuestas_simuladas(rol: str, d: dict) -> str:
    criticos = d["productos_criticos"]
    if rol.startswith("Analista"):
        lineas = "\n".join(f"- {c['codigo']} {c['producto']}: stock {c['stock_actual']} / mínimo "
                           f"{c['stock_minimo']} ({c['semanas_de_cobertura_actual']} semanas de cobertura)"
                           for c in criticos)
        riesgo = ", ".join(r["producto"] for r in d["productos_en_riesgo"]) or "ninguno"
        return f"Productos bajo el mínimo:\n{lineas}\nEn riesgo: {riesgo}.\nRegla: {d['regla']}."
    tablas = []
    for prov, info in d["ordenes_por_proveedor"].items():
        filas = "\n".join(f"| {c['codigo']} | {c['producto']} | {c['cantidad_sugerida']} | "
                          f"{c['costo_unitario_fmt']} | {c['subtotal_fmt']} |"
                          for c in criticos if c["proveedor"] == prov)
        tablas.append(f"### {prov} (subtotal {info['subtotal_fmt']})\n| Código | Producto | Cantidad | "
                      f"Costo unitario | Subtotal |\n|---|---|---|---|---|\n{filas}")
    cuerpo = (f"# Orden de compra · {NEGOCIO}\n\n## Diagnóstico\n{len(criticos)} productos están bajo el "
              f"stock mínimo; se pide para cubrir el mínimo más 2 semanas de consumo.\n\n## Orden de compra\n"
              + "\n\n".join(tablas) + f"\n\n## Total\n**{d['total_orden_fmt']}**\n")
    if rol.startswith("Generador"):
        return cuerpo
    return (cuerpo + "\n## Validación\n- [x] Cantidades recalculadas con la regla de reabastecimiento.\n"
            "- [x] Subtotales por proveedor verificados.\n"
            f"- [x] Total verificado: {d['total_orden_fmt']}.\n\n**Veredicto: APROBADO**\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow de inventario con CrewAI (Grupo 4)")
    parser.add_argument("--csv", default=str(DATOS / "inventario_ferreteria.csv"))
    parser.add_argument("--simulado", action="store_true", help="ensayo sin IA ni internet")
    parser.add_argument("--plot", action="store_true", help="genera el diagrama HTML del Flow")
    args = parser.parse_args()

    flujo = FlujoInventario()
    if args.plot:
        html = Path(flujo.plot("diagrama_flujo_inventario.html", show=True))
        destino = preparar_salidas() / "diagrama_flujo"
        shutil.copytree(html.parent, destino, dirs_exist_ok=True)
        print(f"Diagrama abierto en el navegador y copiado en salidas/diagrama_flujo/{html.name}")
        return
    banner("EJEMPLO 2 · Flow de inventario con router y Crew de compras", args.simulado)
    ruta = Path(args.csv)
    ruta = ruta if ruta.is_absolute() else (BASE / ruta)
    flujo.kickoff(inputs={"ruta_csv": ruta.resolve().as_posix(), "simulado": args.simulado})


if __name__ == "__main__":
    main()
