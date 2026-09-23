#!/bin/bash
#SBATCH -p gg
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH -t 06:00:00
#SBATCH -J resbench_wells
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# Mine the five well conditions of every environment, one environment per node,
# with tools/mine_wells.py (the published rule, C.2). Pools resume: if an
# environment comes out SHORT, raise its draws and resubmit; only the new seeds run.
# Draws (2026-09-23, windowed dataset): a 20,000-run PV pool gave the best
# informative column 30 exact matches, so PV needs about 60,000 for five wells
# with >= 50; the other channel presets start at 40,000, MEANDER and delta at
# 6,000 (100 s and 20 s a run), lobe at 200,000 (6 s a run, columns rarely repeat).
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
PY=/work/08405/ilgar/vista/conda_libraries/resmill/bin/python
RB=/work/08405/ilgar/vista/codes/ResBench
REF=${REF:-$SCRATCH/resbench_v1_ref_new}
WELLS=${WELLS:-$SCRATCH/resbench_new_wells}
mkdir -p "$WELLS" "$RB/logs"
cd "$RB" || exit 1
for spec in lobe=10=200000 channel:PV_SHOESTRING=13=60000 channel:CB_LABYRINTH=15=40000 channel:CB_JIGSAW=15=40000 \
            channel:SH_DISTAL=11=40000 channel:SH_PROXIMAL=7=40000 channel:MEANDER_OXBOW=14=6000 delta=0=6000; do
  IFS== read -r env ri n <<< "$spec"
  slug=${env//:/_}
  srun --exact -N 1 -n 1 -c 144 env -u LD_PRELOAD "$PY" -u tools/mine_wells.py --ref "$REF" --env "$env" --row-index "$ri" \
       --draws "$n" --out "$WELLS" --jobs 140 > "$WELLS/mine_$slug.log" 2>&1 &
done
wait
for f in "$WELLS"/mine_*.log; do echo "== $f"; grep -v "^\s*[0-9]*/" "$f" | tail -8; done
