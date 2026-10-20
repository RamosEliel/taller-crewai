"""
Utilidades compartidas: configuración del LLM, validaciones (guardrails)
y un LLM de simulación para ensayar la demo sin conexión.
"""
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

# Desactiva telemetría y la pregunta interactiva de "trazas" de CrewAI,
# para que la demo en clase no se detenga esperando una respuesta.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv  # noqa: E402
from crewai import LLM, BaseLLM  # noqa: E402

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos"
SALIDAS = BASE / "salidas"
load_dotenv(BASE / ".env")

# Compatibilidad entre CrewAI y Groq
import litellm


_completion_original = litellm.completion


def completion_groq_corregida(*args, **kwargs):
    modelo = str(kwargs.get("model", ""))


    if modelo.startswith("groq/"):
        mensajes = kwargs.get("messages")


        if isinstance(mensajes, list):
            kwargs["messages"] = [
                {k: v for k, v in mensaje.items()
                 if k != "cache_breakpoint"}
                if isinstance(mensaje, dict) else mensaje
                for mensaje in mensajes
            ]


    return _completion_original(*args, **kwargs)


litellm.completion = completion_groq_corregida

# En Windows la consola puede no usar UTF-8: evita errores con tildes y emojis.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except Exception:
        pass


def obtener_llm(simulado: bool = False, respuestas: Callable[[str, str], str] | None = None):
    """
    Devuelve el modelo de lenguaje que usarán los agentes.

    - Modo normal: lee MODEL del archivo .env (por defecto Gemini 3.6 Flash).
      CrewAI usa el formato "proveedor/modelo", p. ej.:
        gemini/gemini-3.6-flash   (requiere GEMINI_API_KEY)
        openai/gpt-4o-mini        (requiere OPENAI_API_KEY)
        groq/llama-3.3-70b-versatile (requiere GROQ_API_KEY y crewai[litellm])
        ollama/llama3.1           (local, requiere Ollama y crewai[litellm])
    - Modo simulado: respuestas preparadas a partir de los datos reales,
      SIN inteligencia artificial. Solo para ensayar o como plan de respaldo.
    """
    if simulado:
        return LLMSimulado(model="simulado/sin-ia", respuestas=respuestas)

    modelo = os.getenv("MODEL") or "gemini/gemini-3.6-flash"
    proveedor = modelo.split("/")[0]
    claves = {"gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "openai": ["OPENAI_API_KEY"],
              "groq": ["GROQ_API_KEY"], "anthropic": ["ANTHROPIC_API_KEY"]}
    if proveedor in claves and not any(os.getenv(c) for c in claves[proveedor]):
        sys.exit(
            f"\n[ERROR] Falta la clave del proveedor '{proveedor}'. Copie .env.example a .env "
            f"y escriba su {claves[proveedor][0]}.\n"
            f"        Para ensayar sin clave ejecute el mismo comando con --simulado.\n"
        )
    extra = {"max_tokens": 4096} if proveedor == "anthropic" else {}
    return LLM(model=modelo, temperature=float(os.getenv("TEMPERATURE", "0.2")), **extra)


def numeros_en_texto(texto: str) -> set[int]:
    """Extrae los números de un texto ignorando separadores: '$1.234.567' -> 1234567."""
    return {int(re.sub(r"[.,]", "", n)) for n in re.findall(r"\d[\d.,]*\d|\d", texto)
            if re.sub(r"[.,]", "", n).isdigit()}


def crear_guardrail(cifra_obligatoria: int, secciones: list[str], nombre_cifra: str):
    """
    Construye un guardrail de función (validación determinista) para una Task.
    CrewAI lo ejecuta sobre la salida del agente: si devuelve (False, motivo),
    el agente recibe el motivo y vuelve a intentarlo (hasta guardrail_max_retries).
    """
    def guardrail(salida) -> tuple[bool, Any]:
        texto = salida.raw
        faltantes = [s for s in secciones if s.lower() not in texto.lower()]
        if faltantes:
            return False, f"Faltan estas secciones obligatorias: {', '.join(faltantes)}."
        if cifra_obligatoria not in numeros_en_texto(texto):
            return False, (f"El {nombre_cifra} no coincide con la herramienta. Debe aparecer "
                           f"exactamente como ${cifra_obligatoria:,}".replace(",", ".") + ".")
        return True, texto
    return guardrail


def preparar_salidas() -> Path:
    SALIDAS.mkdir(exist_ok=True)
    return SALIDAS


def banner(titulo: str, simulado: bool) -> None:
    print("=" * 78)
    print(f"  {titulo}")
    if simulado:
        print("  MODO SIMULADO: los agentes responden con textos preparados a partir de")
        print("  los datos (sin IA). Sirve para ensayar; la demo real usa un LLM.")
    else:
        print(f"  Modelo: {os.getenv('MODEL') or 'gemini/gemini-3.6-flash'}")
    print("=" * 78)


class LLMSimulado(BaseLLM):
    """
    LLM de ensayo. Recibe una función respuestas(rol, prompt) -> texto.
    Imita el ciclo ReAct de CrewAI: si el agente tiene herramientas, primero
    pide usarlas (Action / Action Input) y, con la observación, da la respuesta.
    """
    respuestas: Any = None

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None, **kwargs) -> str:
        if isinstance(messages, str):
            prompt, conversacion = messages, messages
        else:
            prompt = "\n".join(str(m.get("content", "")) for m in messages)
            # El prompt de sistema ya menciona "Observation:"; solo cuenta la conversación.
            conversacion = "\n".join(str(m.get("content", "")) for m in messages
                                     if m.get("role") != "system")
        rol = from_agent.role if from_agent is not None else ""
        herramientas = list(getattr(from_agent, "tools", None) or [])
        ya_uso_herramienta = "Observation:" in conversacion or "Action:" in conversacion
        if herramientas and not ya_uso_herramienta:
            ruta = re.search(r"[\w:\\/. -]+\.csv", prompt)
            entrada = ruta.group(0).strip() if ruta else ""
            entrada = entrada.replace("\\", "/")
            return (f"Thought: Necesito las cifras exactas, usaré la herramienta.\n"
                    f"Action: {herramientas[0].name}\n"
                    f'Action Input: {{"ruta_csv": "{entrada}"}}')
        texto = self.respuestas(rol, prompt) if self.respuestas else "Sin respuesta."
        return f"Thought: Tengo todo lo necesario.\nFinal Answer: {texto}"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128000
