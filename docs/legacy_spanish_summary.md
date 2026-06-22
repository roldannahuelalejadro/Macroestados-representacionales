# Ultimos analisis D_stat LLM

Este paquete ordena los artefactos recientes para leerlos sin perderse en el workspace.

## 01_run_qwen3_coder_B16_3_transformaciones

Corrida conductual real con el modelo usado en fase 1:

- Modelo: `qwen3-coder:30b-32k` via Ollama.
- B: `16`.
- Rollouts: `1600`.
- Transformaciones: `context_injection`, `partial_extraction`, `transform_extraction`.
- Control: `none`.

Archivos importantes:

- `prompts.csv`: prompts usados.
- `rollouts.jsonl`: respuestas generadas.
- `scores.jsonl`: valores `e_b` por rollout.
- `dstat_by_prompt.csv`: dificultad por prompt.
- `report.html`: reporte HTML del run.
- `condition_gen.json`: condicion experimental.

## 02_reanalisis_conductual_qwen3_coder_B16

Tablas y graficos interpretativos del run anterior.

Archivos importantes:

- `secret_only_prompt_summary_by_family_strength.csv`: resumen excluyendo controles benignos.
- `comparison_with_general_b8.csv`: comparacion contra la corrida grande `B=8`.
- `top_40_prompts_by_D_stat_max.csv`: prompts mas dificiles.
- `plots/`: graficos de distribucion y componentes de error.

Resultado central:

- `transform_extraction:2` fue el caso mas fuerte.
- `partial_extraction:3` tambien fue fuerte.
- `context_injection:3` quedo casi al nivel de control.

## 03_proxy_hidden_states_qwen2p5_0p5B

Hidden states extraidos con `Qwen/Qwen2.5-0.5B-Instruct`.

Esto es un analisis proxy: no son activaciones internas de `qwen3-coder:30b-32k`.

Archivos importantes:

- `samples.csv`: prompts analizados.
- `hidden_vectors_by_layer.npz`: vectores por capa.
- `layer_metrics_transform_family.csv`: separabilidad por capa.
- `pca_best_layer_transform_family.png`: PCA crudo por familia.

Resultado central:

- Mejor capa PCA cruda: `8`.
- Silhouette PCA2: `0.549`.
- Las transformaciones son linealmente separables en el proxy.

## 04_proxy_deltas_vs_control

Deltas de activacion proxy contra el control `none` emparejado por `secret_id` y `base_prompt_index`.

Archivos importantes:

- `delta_samples.csv`: muestras con control asociado.
- `delta_vectors_by_layer.npz`: deltas por capa.
- `delta_layer_metrics_transform_family_noncontrol.csv`: separabilidad excluyendo controles.
- `delta_pca_best_layer_transform_family_noncontrol.png`: PCA de deltas.

Resultado central:

- Mejor capa para deltas: `20`.
- Los deltas separan `context_injection` de las familias de extraccion.

## 05_proxy_prediccion_no_lineal_Dstat

Modelos que intentan predecir `D_stat` usando metadata y hidden states proxy.

Archivos importantes:

- `regression_metrics.csv`: metricas completas de regresion.
- `classification_metrics.csv`: metricas completas de clasificacion.
- `best_regression_by_family.csv`: mejores regresores por familia.
- `best_classification_by_family.csv`: mejores clasificadores por familia.
- `dstat_regression_random_kfold.png`: resumen grafico.

Resultado central:

- Hidden proxy lineal predice `D_stat_max` mucho mejor que metadata sola.
- Random k-fold: `R2` aprox. `0.68`.
- Leave-secret-out: `R2` aprox. `0.71` para delta hidden lineal.

## 06_proxy_graficos_Dstat

Graficos PCA coloreados por `D_stat_max`.

Archivos importantes:

- `raw_pca_dstat.png`.
- `delta_noncontrol_pca_dstat.png`.

Lectura:

- La zona de alto `D_stat` se concentra en `partial_extraction` y `transform_extraction`.
- `context_injection` se separa representacionalmente, pero muchas veces queda con bajo `D_stat`.

## 07_reanalisis_corrida_grande_B8

Reanalisis del run grande original `B=8` con `qwen3-coder:30b-32k`.

Sirve como contexto para comparar estabilidad.

## 08_scripts_usados

Scripts principales usados para generar los analisis.

## Advertencia metodologica

La conducta fue medida realmente en `qwen3-coder:30b-32k`.

El analisis de representacion actual es proxy porque Ollama no expuso embeddings ni hidden states para ese modelo en esta sesion.

La afirmacion defensible es:

La geometria proxy de un modelo Qwen pequeno reproduce y predice bastante bien la estructura conductual medida en Qwen3-Coder, especialmente para transformaciones de extraccion.
