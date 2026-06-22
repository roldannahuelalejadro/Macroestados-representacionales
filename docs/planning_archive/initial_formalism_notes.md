# Notas iniciales para BMAD

## Proyecto

Nombre tentativo: Thermodynamic Representation Monitoring for LLM Safety

Fuente conceptual: `C:\Users\User\Desktop\formalismo_llm_macroestados_safety.pdf`

Quiero construir una herramienta local para estudiar, medir y visualizar macroestados representacionales en modelos de lenguaje, con foco en safety. La idea es convertir el formalismo matematico del PDF en un producto experimental usable: primero notebook o CLI, luego dashboard local si el MVP lo justifica.

## Problema

En LLM safety suele ser dificil conectar estados internos del modelo con comportamientos observables. Quiero una herramienta que permita mapear activaciones o representaciones internas a macroestados interpretables, medir transiciones entre esos macroestados y detectar senales de riesgo, inestabilidad o drift.

## Hipotesis

Si podemos agrupar representaciones internas en macroestados y medir dificultad, energia o masa poblacional de esos macroestados, entonces podemos construir monitores de seguridad mas interpretables que solo mirar el output final del modelo.

## Formalismo base

La herramienta deberia poder representar una masa poblacional por macroestado:

\[
M_\sigma = \sum_i \mathbf{1}[s_i \in \sigma]
\]

Tambien deberia soportar una nocion de energia local o dificultad geometrica:

\[
D_H = \Delta s^\top H \Delta s
\]

Y una dificultad estadistica de transicion entre macroestados:

\[
D^{LLM}_{stat}(\sigma_a \to \sigma_b) = -\log P(\sigma_b \mid \sigma_a)
\]

Estas ecuaciones son una base inicial. El PM y el Architect pueden pedir ajustes si el PDF define variantes mas precisas.

## Usuarios objetivo

- Investigadores independientes de interpretabilidad y safety.
- Estudiantes o equipos pequenos que quieren experimentar localmente.
- Desarrolladores que quieren monitorear modelos abiertos sin depender de APIs cloud.

## MVP sugerido

1. Cargar datos de activaciones o representaciones desde archivos locales.
2. Definir o aprender macroestados representacionales.
3. Calcular metricas como masa poblacional, transiciones, dificultad estadistica y energia local.
4. Generar reportes reproducibles en Markdown/JSON.
5. Visualizar trayectorias y cambios de macroestado en una vista simple.
6. Incluir ejemplos con datos sinteticos si todavia no tenemos activaciones reales.

## Alcance inicial

Local first. No depender de servicios pagos. Priorizar trazabilidad experimental, reproducibilidad y claridad matematica. Evitar construir una app demasiado grande antes de tener un pipeline cientifico confiable.

## Preguntas para el PM

- Cual es el usuario primario del MVP: investigador, estudiante o developer?
- El primer producto debe ser notebook, CLI, dashboard o libreria Python?
- Que datos reales vamos a soportar primero?
- Que parte del formalismo es indispensable en v1 y que puede quedar para roadmap?
- Como medimos exito: reproduccion de experimentos, calidad de visualizacion, deteccion de casos de riesgo, o facilidad de uso?

## Restricciones tecnicas

- Debe correr en Windows local.
- Debe poder integrarse con notebooks y scripts Python.
- Debe guardar resultados en formatos simples: Markdown, JSON, CSV o Parquet.
- Debe poder empezar con datos sinteticos.
- Debe separar claramente formalismo matematico, pipeline experimental y visualizacion.
