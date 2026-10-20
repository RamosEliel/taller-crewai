"""
TALLER EVALUABLE - SOLUCIÓN

Proceso real: cada lunes el "Restaurante Sabor Sabanero" (Sincelejo) recibe
reseñas por Google, Instagram y WhatsApp. Esta Crew de 3 agentes produce el
INFORME SEMANAL DE SATISFACCIÓN:

    clasificador (usa la herramienta) -> redactor -> supervisor (recalcula + guardrail)

Uso:
    python taller_plantilla.py              # con el LLM configurado en .env
    python taller_plantilla.py --simulado   # ensayo sin IA ni internet (tokens = 0)

La herramienta 'resumen_resenas' ya está hecha (herramientas.py):
calcula total, promedio, distribución y lista las reseñas negativas.
"""
import re
import sys
import unicodedata
from collections import Counter

from comun import DATOS, obtener_llm, preparar_salidas
from crewai import Agent, Crew, Process, Task
from herramientas import ResumenResenas, resumir_resenas

NEGOCIO = "Restaurante Sabor Sabanero (Sincelejo)"
RUTA_CSV = (DATOS / "resenas_restaurante.csv").as_posix()
SECCIONES = ["Satisfacción general", "Problemas principales", "Plan de mejora", "Respuestas sugeridas"]
SIMULADO = "--simulado" in sys.argv

# Evidencia para el PASO 5 (se llena sola mientras corre la Crew).
USO_HERRAMIENTA: Counter = Counter()      # (agente, herramienta) -> veces
INTENTOS_GUARDRAIL: list[tuple[bool, str]] = []   # (aprobó?, motivo)

try:
    from crewai.events import ToolUsageStartedEvent, crewai_event_bus

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _registrar_uso_herramienta(source, event):
        USO_HERRAMIENTA[(event.agent_role, event.tool_name)] += 1
except Exception:  # el registro es un extra: si esta versión no lo soporta, la Crew corre igual
    pass


# --------------------------------------------------------------------------
# Modo simulado (plan B): respuestas armadas con los datos reales, SIN IA.
# --------------------------------------------------------------------------
def respuestas_simuladas(rol: str, _prompt: str) -> str:
    d = resumir_resenas(RUTA_CSV)
    prom = f"{d['promedio']:.2f}".replace(".", ",")
    neg = d["comentarios_negativos"]
    pedidos = [n for n in neg if re.search(r"domicilio|pedí|dirección", n["comentario"], re.I)]
    cobros = [n for n in neg if n not in pedidos]
    canal_bajo = min(d["promedio_por_canal"], key=d["promedio_por_canal"].get)
    canal_alto = max(d["promedio_por_canal"], key=d["promedio_por_canal"].get)

    if rol.startswith("Clasificador"):
        return (f"Cifras de la herramienta: {d['total_resenas']} reseñas, promedio {prom}, "
                f"distribución {d['distribucion_estrellas']}, por canal {d['promedio_por_canal']}.\n"
                f"Temas negativos: domicilios y pedidos por WhatsApp ({len(pedidos)} reseñas); "
                f"cobros y precios ({len(cobros)} reseñas).\n"
                "Positivos a mantener: sazón (posta negra, sancocho, suero costeño), atención del "
                "personal y almuerzo ejecutivo.")

    respuestas = "\n".join(
        f"- **{n['canal']} · {n['fecha']} ({n['calificacion']}★):** Lamentamos lo ocurrido; "
        "queremos corregirlo. Escríbanos para resolverlo y compensarle la molestia."
        for n in neg)
    cuerpo = (
        f"# Informe semanal de satisfacción · {NEGOCIO}\n\n"
        f"## Satisfacción general\nSe recibieron **{d['total_resenas']} reseñas** con promedio de "
        f"**{prom} estrellas**. Positivas (4-5★): {d['positivas_4_5']}; neutras: {d['neutras_3']}; "
        f"negativas (1-2★): {d['negativas_1_2']}. Mejor canal: {canal_alto}; canal más débil: {canal_bajo}.\n\n"
        f"## Problemas principales\n1. **Domicilios y pedidos por WhatsApp** ({len(pedidos)} reseñas): "
        "demoras, errores de dirección y pedidos mal armados.\n"
        f"2. **Cobros y precios** ({len(cobros)} reseñas): doble cobro y porción pequeña para el precio.\n\n"
        "## Plan de mejora\n1. Confirmar dirección y observaciones de cada domicilio antes de despachar.\n"
        "2. Designar un responsable de contestar WhatsApp en menos de 10 minutos.\n"
        "3. Revisar la cuenta en caja antes de cobrar para evitar dobles cobros.\n"
        "4. Revisar la porción y el precio del pescado.\n\n"
        f"## Respuestas sugeridas\n{respuestas}\n")
    if rol.startswith("Redactor"):
        return cuerpo
    return cuerpo + f"\n**Verificación:** total y promedio ({prom}) recalculados con la herramienta resumen_resenas.\n"


llm = obtener_llm(SIMULADO, respuestas=respuestas_simuladas)
herramienta = ResumenResenas()

# --------------------------------------------------------------------------
# PASO 1: los 3 agentes (role + goal + backstory; respuestas en español).
# --------------------------------------------------------------------------
clasificador = Agent(
    role="Clasificador de reseñas de clientes",
    goal=("Leer con la herramienta resumen_resenas las reseñas de {negocio}, obtener las cifras exactas "
          "y clasificar los comentarios por tema (domicilio, comida, atención, precio, instalaciones, pagos)."),
    backstory=("Eres analista de experiencia del cliente y has trabajado años con restaurantes de la región "
               "Caribe. Nunca inventas ni estimas cifras: siempre las tomas de la herramienta "
               "resumen_resenas. Respondes siempre en español."),
    tools=[herramienta],   # este agente SÍ usa la herramienta
    llm=llm, allow_delegation=False, max_iter=5, verbose=True,
)
redactor = Agent(
    role="Redactor del informe semanal",
    goal=("Convertir la clasificación de las reseñas de {negocio} en un informe claro, breve y accionable "
          "para el administrador, con un plan de mejora y respuestas sugeridas para los clientes."),
    backstory=("Eres comunicador organizacional. Escribes para personas sin formación técnica: frases cortas, "
               "acciones concretas que un restaurante pequeño pueda hacer esta semana, y respuestas "
               "cordiales a los clientes, con respeto y calidez. Solo usas las cifras que recibes. "
               "Respondes siempre en español."),
    llm=llm, allow_delegation=False, max_iter=3, verbose=True,
)
supervisor = Agent(
    role="Supervisor de calidad del informe",
    goal=("Garantizar que el informe de {negocio} esté completo, que cada cifra coincida con los datos "
          "reales y que las respuestas a los clientes no prometan nada que el restaurante no pueda cumplir."),
    backstory=("Eres auditor interno, meticuloso y con ojo para los detalles. Recalculas con la herramienta "
               "resumen_resenas, corriges cualquier cifra errónea, completas lo que falte y no dejas pasar "
               "afirmaciones sin sustento. Respondes siempre en español."),
    tools=[herramienta],   # recalcula para verificar
    llm=llm, allow_delegation=False, max_iter=5, verbose=True,
)

# --------------------------------------------------------------------------
# PASO 2: las 3 tareas (description + expected_output + agent).
# --------------------------------------------------------------------------
t1 = Task(
    description=("Usa la herramienta resumen_resenas con ruta_csv='{ruta_csv}' para leer las reseñas de "
                 "{negocio}. Con lo que devuelve la herramienta: (1) copia tal cual el total de reseñas, el "
                 "promedio de estrellas, la distribución y el promedio por canal; (2) clasifica los "
                 "comentarios negativos por tema, indicando cuántas reseñas corresponden a cada tema y un "
                 "ejemplo textual; (3) señala 2 o 3 aspectos positivos que conviene mantener; (4) copia la "
                 "lista COMPLETA de reseñas negativas (todas, no solo ejemplos), cada una con su fecha, "
                 "canal, calificación y comentario, tal cual la entrega la herramienta. No inventes "
                 "ninguna cifra."),
    expected_output=("Análisis en español con: (1) las cifras exactas de la herramienta (total, promedio, "
                     "distribución y promedio por canal), (2) los temas de las quejas con su cantidad de "
                     "reseñas y un ejemplo cada uno, (3) los aspectos positivos a mantener, (4) la lista "
                     "completa de reseñas negativas con fecha, canal, calificación y comentario."),
    agent=clasificador,
)
t2 = Task(
    description=("Con el análisis recibido, redacta el informe semanal de satisfacción de {negocio} en "
                 "Markdown, con estas cuatro secciones como títulos '##' exactos: Satisfacción general "
                 "(total de reseñas, promedio de estrellas, distribución y canal mejor y peor calificado); "
                 "Problemas principales (los más frecuentes, con cantidad de reseñas y un ejemplo); Plan de "
                 "mejora (3 a 5 acciones concretas, con responsable y plazo sugeridos); Respuestas sugeridas "
                 "(EXACTAMENTE una respuesta corta y cordial por cada reseña negativa de la lista del "
                 "análisis, sin omitir ninguna, cada una en una línea que empiece con "
                 "'- **Canal · AAAA-MM-DD (N★):**' usando el canal y la fecha tal cual aparecen en la lista). "
                 "Máximo 700 palabras y solo cifras del análisis."),
    expected_output=("Borrador del informe en Markdown, en español, con exactamente las secciones "
                     "'## Satisfacción general', '## Problemas principales', '## Plan de mejora' y "
                     "'## Respuestas sugeridas', esta última con una línea por cada reseña negativa."),
    agent=redactor,
    context=[t1],         # recibe el resultado del clasificador
)


# --------------------------------------------------------------------------
# PASO 3: el guardrail. Devuelve (True, texto) o (False, "motivo").
# --------------------------------------------------------------------------
_DATOS = resumir_resenas(RUTA_CSV)
PROMEDIO_REAL = _DATOS["promedio"]
CANALES = list(_DATOS["promedio_por_canal"])                       # Google, WhatsApp, Instagram
ESPERADAS = Counter((n["fecha"], n["canal"]) for n in _DATOS["comentarios_negativos"])


def _sin_tildes(texto: str) -> str:
    """Minúsculas y sin tildes, para comparar 'Satisfacción' con 'satisfaccion'."""
    base = unicodedata.normalize("NFD", texto)
    return "".join(c for c in base if unicodedata.category(c) != "Mn").lower()


def _aparece_promedio(texto: str) -> bool:
    """True si el promedio real está escrito como 3.44 o 3,44 (el LLM escribe en español)."""
    patron = re.escape(str(PROMEDIO_REAL)).replace(r"\.", "[.,]")
    return re.search(rf"(?<![\d.,]){patron}0*(?!\d)", texto) is not None


def _respuestas_encontradas(normalizado: str) -> Counter:
    """Cuenta las respuestas de la sección 'Respuestas sugeridas' como pares (fecha, canal).

    Una respuesta es una línea que trae una fecha AAAA-MM-DD y un nombre de canal; si hay varios
    canales en la línea, vale el que aparece primero (el del rótulo).
    """
    inicio = re.search(r"^\s*#{2,}\s*(?:\d+[.)]\s*)?[*_]*\s*respuestas sugeridas.*$", normalizado, re.M)
    if not inicio:
        return Counter()
    resto = normalizado[inicio.end():]
    fin = re.search(r"^\s*(?:#{2,}\s|[*_]{2}verificacion|-{3,}\s*$)", resto, re.M)
    encontradas: Counter = Counter()
    for linea in (resto[:fin.start()] if fin else resto).splitlines():
        fecha = re.search(r"\d{4}-\d{2}-\d{2}", linea)
        posiciones = {c: linea.find(_sin_tildes(c)) for c in CANALES if _sin_tildes(c) in linea}
        if fecha and posiciones:
            encontradas[(fecha.group(0), min(posiciones, key=posiciones.get))] += 1
    return encontradas


def guardrail_informe(salida):
    texto = salida.raw
    normalizado = _sin_tildes(texto)
    # 3a: cada sección debe existir como título '##' (admite '## 1. Plan de mejora' o negritas)
    faltantes = [s for s in SECCIONES
                 if not re.search(rf"^\s*#{{2,}}\s*(?:\d+[.)]\s*)?[*_]*\s*{re.escape(_sin_tildes(s))}",
                                  normalizado, re.M)]
    if faltantes:
        motivo = f"Faltan secciones (deben ser títulos '##'): {', '.join(faltantes)}."
        INTENTOS_GUARDRAIL.append((False, motivo))
        return False, motivo
    # 3b: el promedio real de la herramienta debe aparecer en el informe
    if not _aparece_promedio(texto):
        motivo = f"El promedio de estrellas debe ser {PROMEDIO_REAL} (el valor que entrega la herramienta)."
        INTENTOS_GUARDRAIL.append((False, motivo))
        return False, motivo
    # 3c: una respuesta por cada reseña negativa, con su canal y fecha correctos
    encontradas = _respuestas_encontradas(normalizado)
    faltan, sobran = ESPERADAS - encontradas, encontradas - ESPERADAS
    if faltan or sobran:
        partes = [f"Debe haber exactamente una respuesta sugerida por cada reseña negativa "
                  f"({sum(ESPERADAS.values())}), cada una en una línea que empiece con "
                  "'- **Canal · AAAA-MM-DD (N★):**' usando el canal y la fecha de la herramienta."]
        if faltan:
            partes.append("Faltan respuestas para: "
                          + ", ".join(f"{c} · {f}" for f, c in faltan.elements()) + ".")
        if sobran:
            partes.append("Estas respuestas no corresponden a ninguna reseña negativa (revisen el canal "
                          "y la fecha): " + ", ".join(f"{c} · {f}" for f, c in sobran.elements()) + ".")
        motivo = " ".join(partes)
        INTENTOS_GUARDRAIL.append((False, motivo))
        return False, motivo
    INTENTOS_GUARDRAIL.append((True, "ok"))
    return True, texto


t3 = Task(
    description=("Revisa el borrador del informe. Vuelve a consultar resumen_resenas (ruta_csv='{ruta_csv}') "
                 "y compara CADA cifra: total de reseñas, promedio de estrellas, distribución y cantidades "
                 "citadas. Comprueba también que haya EXACTAMENTE una respuesta sugerida por cada reseña "
                 "negativa de la herramienta, con su canal y fecha correctos (cada reseña negativa tiene "
                 "su canal en la herramienta; no los confundas). Corrige lo que no coincida, completa las "
                 "secciones que falten y elimina "
                 "promesas o afirmaciones sin sustento. Entrega la versión FINAL completa del informe (no un "
                 "resumen de tus cambios), con las cuatro secciones como títulos '##' y el promedio exacto "
                 "de estrellas escrito en el texto. Cierra con una línea '**Verificación:**' que diga qué "
                 "recalculaste."),
    expected_output=("Informe final en Markdown, en español, con las secciones '## Satisfacción general', "
                     "'## Problemas principales', '## Plan de mejora' y '## Respuestas sugeridas', con el "
                     "promedio de estrellas exacto de la herramienta, listo para entregar al administrador."),
    agent=supervisor,
    context=[t1, t2],
    guardrail=guardrail_informe,
    guardrail_max_retries=3,
    output_file=str(preparar_salidas() / "taller_informe_resenas.md"),
)


# --------------------------------------------------------------------------
# PASO 4: la Crew con proceso secuencial.
# --------------------------------------------------------------------------
def armar_crew() -> Crew:
    pendientes = [n for n, x in [("clasificador", clasificador), ("redactor", redactor),
                                 ("supervisor", supervisor)] if not x.role or not x.goal]
    pendientes += [n for n, t in [("t1", t1), ("t2", t2), ("t3", t3)] if not t.description]
    if pendientes:
        raise SystemExit(f"Aún faltan TODO por completar en: {', '.join(pendientes)}")
    return Crew(
        agents=[clasificador, redactor, supervisor],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=True,
        max_rpm=8,        # respeta los límites de la capa gratuita
    )


def resumen_paso_5(resultado) -> None:
    """Imprime la evidencia para responder el PASO 5 del taller."""
    rechazos = [m for ok, m in INTENTOS_GUARDRAIL if not ok]
    print("\n" + "=" * 78 + "\nEVIDENCIA PARA EL PASO 5\n" + "=" * 78)
    print("1) ¿Qué agentes usaron la herramienta?")
    if USO_HERRAMIENTA:
        for (agente, herr), n in USO_HERRAMIENTA.items():
            print(f"   - {agente}: {herr} x {n}")
    else:
        print("   (sin registro: revisen en la salida verbose los bloques 'Tool Execution')")
    print(f"2) ¿Se activó el guardrail? Se ejecutó {len(INTENTOS_GUARDRAIL)} vez/veces; "
          f"rechazó {len(rechazos)}.")
    for i, motivo in enumerate(rechazos, 1):
        print(f"   - rechazo {i}: {motivo}")
    print(f"3) Tokens usados: {resultado.token_usage}")
    if SIMULADO:
        print("   NOTA: ejecución SIMULADA (sin IA): los tokens son 0 y no sirve como evidencia del taller.")


if __name__ == "__main__":
    resultado = armar_crew().kickoff(inputs={"negocio": NEGOCIO, "ruta_csv": RUTA_CSV})
    print(resultado.raw)
    print(resultado.token_usage)
    resumen_paso_5(resultado)
