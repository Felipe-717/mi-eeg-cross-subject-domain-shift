# `pod/` — todo lo que va a la nube

Ejecución en RunPod: preparar el entorno, regenerar el caché desde cero, **verificar
que reproduce la Fase 2**, y medir cuánto cuesta de verdad entrenar.

Se regenera allí en lugar de subirlo porque **el pod descarga a velocidad de centro de
datos**. Tu internet solo mueve los CSV de métricas, que son kilobytes.

---

## Modelo de costes — leer una vez

**Se cobra por segundo mientras el pod está ENCENDIDO**, esté ocupado o no. Un pod
inactivo cuesta lo mismo que uno al 100 %.

> **Los comandos son gratis. El pod olvidado encendido, no.** No te cortes en pedir
> comprobaciones; lo que hay que vigilar es apagar.

| acción | GPU | disco |
|---|---|---|
| running | se cobra | $0.10/GB/mes |
| **stopped** | no | **$0.20/GB/mes — el doble** |
| terminated | no | no, pero **`/workspace` se borra** |

**Parar sale más caro que destruir**, y no protege: si el saldo llega a $0, los pods sin
network volume se terminan y los datos no se recuperan. La regla es **destruir, no
parar** — y por eso el caché tiene que salir del pod antes (fase `export`).

**GPU recomendada: A40 48 GB (~$0.44/h).** Es Ampere `sm_86`, la misma arquitectura que
la RTX 3050 local, así que `mamba-ssm` compila y torch cu121 funciona sin tocar nada.
Una H100 con EEGNet (2.898 parámetros) es tirar dinero.

---

## Qué sobrevive y qué no

| ruta | disco | tras *stop* | tras *terminate* |
|---|---|:--:|:--:|
| `/workspace/**` | volume | ✅ | ❌ **se borra** |
| todo lo demás (`/`, `~`, `/tmp`, `/raw`) | container | ❌ | ❌ |

```
/workspace/tesis/     código                    persiste hasta terminate
/workspace/venv/      entorno virtual           persiste hasta terminate
/workspace/cache/     caché procesado           persiste hasta terminate
/workspace/results/   CSV de métricas           persiste hasta terminate
/workspace/logs/      logs de los procesos      persiste hasta terminate

/raw/mne_data/        crudo de MOABB, decenas GB   EFÍMERO, y así lo queremos
~/.cache/             cachés de librerías          efímero
```

Dos trampas que se derivan:

- **`pip install` no sobrevive a un stop** (escribe en el Python del sistema, fuera de
  `/workspace`). Por eso `setup.sh` crea el entorno virtual **dentro** de `/workspace`.
- **Los directorios por defecto de MNE y HF son efímeros.** `env.sh` los redirige.

---

## Primera sesión

```bash
# 1. en tu portátil: subir el repo (o clonar en el pod con REPO_URL)
scp -P <puerto> -r . root@<ip>:/workspace/tesis

# 2. en el pod
bash pod/setup.sh
source /workspace/env.sh

# 3. todo de una vez (~2 h, ~$0.90 en A40)
python pod/run.py --fase gpu download cache verify export bench --hf-repo usuario/openbmi-ica
```

| fase | qué hace | tiempo |
|---|---|---|
| `gpu` | verifica tarjeta, arquitectura y VRAM; **avisa si está sobredimensionada** | seg |
| `download` | MOABB, 54 sujetos, reanudable | ~20-30 min |
| `cache` | preprocesado + ICA, **paralelo por sujeto** | ~30 min |
| `verify` | **¿reproduce la Fase 2?** | ~5 min |
| `export` | consolida a fp16 (2.68 GB) y sube a HF | ~5 min |
| `bench` | mide 1 fold LOSO y 1 sujeto within | ~30 min |

Todas son repetibles: `download` y `cache` saltan lo hecho, `verify` no escribe nada.

### Sesiones siguientes

```bash
bash pod/setup.sh && source /workspace/env.sh
huggingface-cli download usuario/openbmi-ica --repo-type dataset --local-dir /workspace/cache
```

Bajar 2.68 GB son un par de minutos y ya se entrena.

---

## La fase `verify` no se salta

Si se reconstruye el caché en otra máquina, **hay que demostrar que es el mismo** que el
validado en local, o toda la Fase 2 queda en el aire.

Comprueba tres cosas: integridad (54 sujetos, 400×62, balanceados, sin NaN), que el
`cfg_hash` sea **`6391bc7fe8d4`**, y que el Control A reproduzca valores conocidos:

| sujeto | proxy EOG | motor | por qué este sujeto |
|---|---|---|---|
| sub-29 | 0.705 | 0.943 | muscular irreducible: debe **seguir alto** |
| sub-04 | 0.518 | 0.608 | ocular resuelto por el ICA |
| sub-35 | 0.543 | 0.608 | ocular resuelto |
| sub-01 | 0.478 | 0.808 | limpio: control negativo, no debe moverse |

Tolerancia ±0.03. **El ICA es estocástico**: fijamos `random_state`, pero una versión
distinta de `mne` o `mne-icalabel` puede converger a otros componentes y dar otros
números *aunque el hash coincida*. Por eso `requirements.txt` fija versiones exactas y
`verify` mide el resultado, no solo la configuración.

Si falla, el script **se detiene a propósito**. `--seguir-si-falla-verify` existe para
depurar, no para producir resultados de tesis.

---

## Qué esperar de `bench`

Es la fase que convierte estimaciones en datos. Mide un fold LOSO real (53 sujetos de
entrenamiento) y un sujeto de within, y extrapola a las combinaciones que importan.

Con esos dos números se deciden **GPU, número de folds y presupuesto**. Hasta entonces
todo lo que hemos dicho sobre horas y coste son extrapolaciones de una regla de tres
sobre un LOSO de 6 sujetos, y pueden estar equivocadas por un factor de dos.

Las cifras que imprime son **secuenciales**. EEGNet no satura una GPU grande, así que
con folds en paralelo bajan bastante — pero eso se decide después de ver el número.

---

## Lo que todavía NO está aquí

Las fases `within` y `loso` de producción **no se han incluido a propósito**: dependen
de dos decisiones pendientes del director.

- **Definición del DSI (H2).** La propuesta es medir *sujeto visto vs no visto con los
  mismos datos*, lo que exige un tercer conjunto de evaluación (10 % reservado de cada
  sujeto de entrenamiento, estratificado por clase y sesión). Implementarlo antes de que
  se apruebe sería construir sobre una definición que puede cambiar.
- **Alcance del LOSO.** 18 folds estratificados frente a 54 completos.

`bench` no depende de ninguna de las dos: mide el coste de un fold con el esquema
actual, y el tiempo no cambia por añadir un conjunto de evaluación más.
