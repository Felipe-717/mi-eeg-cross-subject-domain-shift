# Inventario de publicacion

## Contenido

La publicacion incluye codigo de `src/`, `scripts/`, `pod/` y `tests/`, fuentes y materiales del informe, documentacion de reproduccion y CSV seleccionados de resultados. Se excluyeron datos EEG originales, caches, checkpoints/pesos, binarios generados y documentos internos de trabajo para evitar distribuir datos de participantes, artefactos grandes o material ajeno a la reproduccion. De los resultados se incluyeron reportes seleccionados, priorizando `punto6b`, avance 7 y archivos pequenos; otros CSV se excluyeron por tamano.

Tamano total de los archivos publicados, sin el escaneo eliminado: **286,576,259 bytes (286.6 MB)** en 984 archivos.

## Resolucion del escaneo

Los avisos de posible dato, referencia o secreto se revisaron en su linea. Los avisos señalados como secretos eran nombres de archivos/canales EEG; los avisos restantes eran rutas del pod, argumentos/comentarios con la palabra token y valores de resultados. No se encontro una credencial literal en los avisos listados. El escaneo tambien registro archivos CSV omitidos por tamano; se detallan como excluidos.

| Aviso o archivo omitido | Resolucion | Detalle |
| --- | --- | --- |
| `src\adapters.py:59` | Falso positivo | Falso positivo: token del modelo, argumento CLI o referencia de codigo; no es credencial. |
| `src\adapters.py:168` | Falso positivo | Falso positivo: token del modelo, argumento CLI o referencia de codigo; no es credencial. |
| `src\training.py:196` | Falso positivo | Falso positivo: token del modelo, argumento CLI o referencia de codigo; no es credencial. |
| `src\training.py:208` | Falso positivo | Falso positivo: token del modelo, argumento CLI o referencia de codigo; no es credencial. |
| `scripts\avance7_norma_gradiente.py:22` | Falso positivo | Falso positivo: token del modelo, argumento CLI o referencia de codigo; no es credencial. |
| `scripts\train_baseline.py:184` | Falso positivo | Falso positivo: token del modelo, argumento CLI o referencia de codigo; no es credencial. |
| `scripts\wbcic\build_cache.py:70` | Falso positivo | Falso positivo: nombre de archivo/canal del conjunto EEG; no es secreto. |
| `scripts\wbcic\descargar.py:63` | Falso positivo | Falso positivo: nombre de archivo/canal del conjunto EEG; no es secreto. |
| `scripts\wbcic\descargar.py:64` | Falso positivo | Falso positivo: nombre de archivo/canal del conjunto EEG; no es secreto. |
| `scripts\wbcic\exportar.py:109` | Falso positivo | Falso positivo: referencia textual a token/licencia de acceso; no hay valor de credencial. |
| `scripts\wbcic\exportar.py:169` | Falso positivo | Falso positivo: referencia textual a token/licencia de acceso; no hay valor de credencial. |
| `pod\avance_7\avance_7.py:25` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\avance_7.py:43` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\bajar.sh:17` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\preparar.sh:8` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\preparar.sh:16` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\preparar.sh:17` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\preparar.sh:22` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\preparar.sh:30` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\avance_7\ver_epocas.sh:5` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\MANANA.md:32` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\preparar.sh:24` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\preparar.sh:28` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\punto6b\preparar.sh:3` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\punto6b\preparar.sh:10` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\setup.sh:4` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\setup.sh:15` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\setup.sh:87` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\setup.sh:89` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\subir.sh:22` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `pod\subir.sh:64` | Falso positivo | Falso positivo: ruta, variable de entorno o instruccion de autenticacion; no contiene un secreto. |
| `informe\herramientas\equivalencia.txt:1` | Excluido | El archivo no forma parte del paquete publicado. |
| `informe\herramientas\equivalencia.txt:3` | Excluido | El archivo no forma parte del paquete publicado. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:3` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica_ea45.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\descartados\baseline_eegnet_smoke_completo.csv:2` | Falso positivo | Falso positivo: celda de datos o resultado numerico, sin credencial. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica_ea45.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica_ea45.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_conformer_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_conformer_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_eegmamba_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_eegmamba_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_eegnet_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_eegnet_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_labram_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\cruzado_labram_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_conformer_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_conformer_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegmamba_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegmamba_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegnet_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegnet_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_labram_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_openbmi_labram_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_conformer_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_conformer_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegmamba_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegmamba_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegnet_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegnet_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_labram_s2027.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\base\holdout_wbcic_labram_s2028.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\cruzado_conformer_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\cruzado_eegmamba_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\cruzado_eegnet_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\cruzado_labram_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_openbmi_conformer_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_openbmi_eegmamba_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_openbmi_eegnet_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_openbmi_labram_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_wbcic_conformer_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_wbcic_eegmamba_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_wbcic_eegnet_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\avance_7\transferencia\ea\holdout_wbcic_labram_s2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod\within_gpu_16sujetos_semilla2026.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_baseline\baseline_eegnet_loso_completo_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_baseline\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_bloque_a_parcial\baseline_conformer_loso_completo_global_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_bloque_a_parcial\baseline_conformer_loso_completo_global_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_bloque_a_parcial\baseline_conformer_loso_completo_global_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_bloque_a_parcial\baseline_eegnet_loso_completo_global_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_bloque_a_parcial\baseline_eegnet_loso_completo_global_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_bloque_a_parcial\baseline_eegnet_loso_completo_global_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_conformer\openbmi\baseline_conformer_loso_completo_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_conformer\openbmi\baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_conformer\openbmi\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba\baseline_eegmamba_within_completo_global_porsesion_avgpool_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w0.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w1.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w10.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w11.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w12.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w13.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w14.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w15.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w2.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w3.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w4.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w5.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w6.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w7.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w8.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w9.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_labram\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_labram\baseline_labram_loso_completo_global_linear_uv1_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_labram\baseline_labram_within_completo_global_mezcla_full_uv1_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_labram\baseline_labram_within_completo_global_mezcla_linear_uv1_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\pod_labram\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\cruzado\cruzado_conformer.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\cruzado\cruzado_eegmamba.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\cruzado\cruzado_eegnet.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\cruzado\cruzado_labram.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\curvas\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\curvas\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\openbmi\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_openbmi_conformer.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_openbmi_eegmamba.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_openbmi_eegnet.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_openbmi_labram.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_wbcic_conformer.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_wbcic_eegmamba.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_wbcic_eegnet.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\holdout44\holdout_wbcic_labram.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\curvas\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\curvas\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\curvas\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\punto6b\openbmi\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\bootstrap_ic.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\baseline_eegnet_comparanorm_completo.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\baseline_eegnet_smoke_completo.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\baseline_eegnet_smokeloso_completo.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\baseline_eegnet_within_completo_global.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\control_cache_within_fbcsp.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\control_ica_ocular.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\control_ocular_vs_motor.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\eda_calidad_30sujetos.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\eda_erd_por_canal.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\eda_inclusion_exclusion.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\eda_senal_erd_resumen.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\reference_fbcsp_loso.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\mind\tensor_cache_log.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\baseline_eegnet_within_completo_global_porsesion.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\control_ocular.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\control_ocular_epochs_cache_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\diag_banda_epochs_cache.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\puerta1_csp_replica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\tensor_cache_log.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |
| `results\reports\openbmi\tensor_cache_log_ica.csv` | Excluido por tamano | CSV omitido por el limite de tamano/seleccion del paquete publicado. |

- Excluidos por el orquestador (2026-11-16): `pod/MANANA*.md` (notas internas de sesion de RunPod).
