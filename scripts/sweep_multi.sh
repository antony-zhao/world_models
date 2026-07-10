#!/bin/bash
# sweep_multi.sh — seed sweep across diverse Atari100k games.
# Games chosen for diverse variance profiles:
#   pong    (6 act)  low variance, near-solved -> sanity check
#   boxing  (18 act) full action space test
#   freeway (3 act)  hard exploration test
#   krull   (18 act) large reward-scale test

GAMES="pong boxing freeway krull"
SEEDS="1"

for game in $GAMES; do
    for seed in $SEEDS; do
        echo "=== game=${game} seed=${seed} ==="
        python -m world_models.torch.agents.train \
            --config configs/atari100k.yaml \
            env.game=${game} \
            seed=${seed}
    done
done
