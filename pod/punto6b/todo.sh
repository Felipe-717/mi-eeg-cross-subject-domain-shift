#!/usr/bin/env bash
# Preparacion + experimento 44 + 10, encadenados. Si la preparacion falla, NO entrena.
set -uo pipefail
T0=$(date +%s)
bash /workspace/tesis/pod/punto6b/preparar.sh || { echo ""; echo "!! FALLO LA PREPARACION: no se entrena nada"; exit 1; }
echo ""
echo "############################################################"
echo "###  preparacion lista en $(( ($(date +%s)-T0)/60 )) min. Entrenando."
echo "############################################################"
bash /workspace/tesis/pod/punto6b/correr.sh
echo ""
echo "=== TOTAL $(( ($(date +%s)-T0)/60 )) min ==="
