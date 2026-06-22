# Planes previos y puente hacia BMAD

Esta carpeta contiene los documentos de planificacion escritos antes o durante el intento inicial de pasar el proyecto a BMAD.

## Orden recomendado de lectura

1. `dstat_llm_phase1_plan_v5.md`

   Version final aprobada del plan de fase 1. Es el documento mas importante. Define:

   - \(D_{stat}^{LLM}\);
   - prompts indexados \(q_\alpha = (H_\alpha, x_\alpha, m_\alpha)\);
   - canaries/refusal;
   - transformaciones \(T_j(q)\);
   - scoring por componentes \(e_b\);
   - separacion entre fase 1 conductual y fase 2 representacional.

2. `dstat_llm_phase1_plan_v4.md`, `v3`, `v2`

   Historial de iteracion conceptual. Sirven para ver como se fueron resolviendo dudas de notacion, contexto, metadata, canaries, curvas \(\gamma_b\), y definicion de transformaciones.

3. `dstat_llm_bmad_plan.md`

   Documento puente que queriamos darle a BMAD para convertir el plan cientifico en tareas de implementacion.

4. `notas-iniciales-formalismo.md`

   Notas iniciales del proyecto antes de formalizar el plan. Son utiles para reconstruir la motivacion general desde el PDF.

## Relacion con los analisis recientes

Los analisis recientes implementan una parte concreta del plan:

- Fase 1: medir \(D_{stat}^{LLM}\) desde rollouts observables.
- Subconjunto focalizado: tres familias de transformaciones.
- Resultado fuerte: `partial_extraction` y `transform_extraction` concentran alta dificultad.
- Fase 2 preliminar/proxy: hidden states de un modelo Qwen pequeno para explorar geometria representacional.

La fase 2 definitiva todavia requiere representaciones del mismo modelo que produce la conducta, idealmente `qwen3-coder:30b-32k` o un equivalente cargado en Transformers.
