# Publication inventory

## Contents

The publication includes code from `src/`, `scripts/`, `pod/`, and `tests/`, report source files and materials, reproduction documentation, and selected result CSV files. Original EEG data, caches, checkpoints/weights, generated binaries, and internal working documents were excluded to avoid distributing participant data, large artifacts, or material unrelated to reproduction. Selected reports were included, prioritizing `punto6b`, `avance_7/`, and small files; other CSV files were excluded because of their size.

Total size of published files, excluding the removed scan: **286,576,259 bytes (286.6 MB)** across 984 files.

## Scan findings

Potential data, reference, or secret findings were reviewed at the reported lines. Findings flagged as secrets were EEG file/channel names; the remaining findings were pod paths, arguments/comments containing the word token, and result values. No literal credential was found among the listed findings. The scan also recorded CSV files omitted because of their size; these are listed as excluded.

| Finding or omitted file | Resolution | Details |
| --- | --- | --- |
| `src\adapters.py:59` | False positive | False positive: model token, CLI argument, or code reference; not a credential. |
| `src\adapters.py:168` | False positive | False positive: model token, CLI argument, or code reference; not a credential. |
| `src\training.py:196` | False positive | False positive: model token, CLI argument, or code reference; not a credential. |
| `src\training.py:208` | False positive | False positive: model token, CLI argument, or code reference; not a credential. |
| `scripts\avance7_norma_gradiente.py:22` | False positive | False positive: model token, CLI argument, or code reference; not a credential. |
| `scripts\train_baseline.py:184` | False positive | False positive: model token, CLI argument, or code reference; not a credential. |
| `scripts\wbcic\build_cache.py:70` | False positive | False positive: EEG dataset file/channel name; not a secret. |
| `scripts\wbcic\descargar.py:63` | False positive | False positive: EEG dataset file/channel name; not a secret. |
| `scripts\wbcic\descargar.py:64` | False positive | False positive: EEG dataset file/channel name; not a secret. |
| `scripts\wbcic\exportar.py:109` | False positive | False positive: textual reference to an access token/license; no credential value is present. |
| `scripts\wbcic\exportar.py:169` | False positive | False positive: textual reference to an access token/license; no credential value is present. |
| `pod\avance_7\avance_7.py:25` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\avance_7.py:43` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\bajar.sh:17` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\preparar.sh:8` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\preparar.sh:16` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\preparar.sh:17` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\preparar.sh:22` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\preparar.sh:30` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\avance_7\ver_epocas.sh:5` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\MANANA.md:32` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\preparar.sh:24` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\preparar.sh:28` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\punto6b\preparar.sh:3` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\punto6b\preparar.sh:10` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\setup.sh:4` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\setup.sh:15` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\setup.sh:87` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\setup.sh:89` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\subir.sh:22` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `pod\subir.sh:64` | False positive | False positive: path, environment variable, or authentication instruction; contains no secret. |
| `informe\herramientas\equivalencia.txt:1` | Excluded | The file is not part of the published package. |
| `informe\herramientas\equivalencia.txt:3` | Excluded | The file is not part of the published package. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv:3` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica_ea45.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\descartados\baseline_eegnet_smoke_completo.csv:2` | False positive | False positive: data cell or numeric result; no credential. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_beta__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mu__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_mubeta__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugcr_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugfft_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vaugnone_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw02_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_eegnet_within_completo_global_porsesion_vw03_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_loso_completo_global_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugcr_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vaugfft_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\curvas\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_vw02_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_conformer_loso_completo_global_epochs_cache_ica_ea__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica_ea__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_eegnet_loso_completo_global_epochs_cache_ica_ea__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\openbmi\fewshot\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica_ea__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica_ea45.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\reports\wbcic_2c\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica_ea45.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_conformer_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_conformer_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_eegmamba_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_eegmamba_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_eegnet_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_eegnet_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_labram_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\cruzado_labram_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_conformer_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_conformer_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegmamba_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegmamba_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegnet_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_eegnet_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_labram_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_openbmi_labram_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_conformer_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_conformer_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegmamba_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegmamba_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegnet_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_eegnet_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_labram_s2027.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\base\holdout_wbcic_labram_s2028.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\cruzado_conformer_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\cruzado_eegmamba_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\cruzado_eegnet_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\cruzado_labram_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_openbmi_conformer_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_openbmi_eegmamba_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_openbmi_eegnet_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_openbmi_labram_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_wbcic_conformer_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_wbcic_eegmamba_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_wbcic_eegnet_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\avance_7\transferencia\ea\holdout_wbcic_labram_s2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod\within_gpu_16sujetos_semilla2026.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_baseline\baseline_eegnet_loso_completo_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_baseline\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_bloque_a_parcial\baseline_conformer_loso_completo_global_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_bloque_a_parcial\baseline_conformer_loso_completo_global_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_bloque_a_parcial\baseline_conformer_loso_completo_global_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_bloque_a_parcial\baseline_eegnet_loso_completo_global_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_bloque_a_parcial\baseline_eegnet_loso_completo_global_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_bloque_a_parcial\baseline_eegnet_loso_completo_global_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_conformer\openbmi\baseline_conformer_loso_completo_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_conformer\openbmi\baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_conformer\openbmi\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba\baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba\baseline_eegmamba_within_completo_global_porsesion_avgpool_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegmamba_flatten\baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w0.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w1.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w10.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w11.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w12.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w13.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w14.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w15.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w2.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w3.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w4.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w5.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w6.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w7.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w8.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_eegnet_parcial\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica__w9.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_labram\baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_labram\baseline_labram_loso_completo_global_linear_uv1_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_labram\baseline_labram_within_completo_global_mezcla_full_uv1_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_labram\baseline_labram_within_completo_global_mezcla_linear_uv1_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\pod_labram\baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\cruzado\cruzado_conformer.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\cruzado\cruzado_eegmamba.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\cruzado\cruzado_eegnet.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\cruzado\cruzado_labram.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\curvas\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\curvas\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\openbmi\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6\reports\wbcic_2c\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_openbmi_conformer.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_openbmi_eegmamba.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_openbmi_eegnet.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_openbmi_labram.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_wbcic_conformer.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_wbcic_eegmamba.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_wbcic_eegnet.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\holdout44\holdout_wbcic_labram.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\curvas\baseline_conformer_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\curvas\baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\curvas\baseline_eegnet_all_comun_wbcic_global_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\punto6b\openbmi\curvas\baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_hold10s2026_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\bootstrap_ic.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\baseline_eegnet_comparanorm_completo.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\baseline_eegnet_smoke_completo.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\baseline_eegnet_smokeloso_completo.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\baseline_eegnet_within_completo_global.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\control_cache_within_fbcsp.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\control_ica_ocular.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\control_ocular_vs_motor.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\eda_calidad_30sujetos.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\eda_erd_por_canal.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\eda_inclusion_exclusion.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\eda_senal_erd_resumen.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\reference_fbcsp_loso.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\mind\tensor_cache_log.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\baseline_eegnet_within_completo_global_porsesion.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\control_ocular.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\control_ocular_epochs_cache_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\diag_banda_epochs_cache.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\puerta1_csp_replica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\tensor_cache_log.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |
| `results\reports\openbmi\tensor_cache_log_ica.csv` | Excluded due to size | CSV omitted due to the published package size/selection limit. |

- Excluded by the orchestrator (2026-11-16): `pod/MANANA*.md` (internal RunPod session notes).
