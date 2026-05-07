# Ablation Experiments for TRUSTEE

For each experiment, first ensure you are in the `verl` directory and the simulator server is running.

## Environment
We ablate the environment by removing the expected tool calls, or the user intent, or the user persona from the environment. After setting the appropriate flags in the script, run the following command to start the training:
```bash
bash recipe/trustee/scripts/ablations/ablation_environment.sh
```

## Curriculum
### General Ablation
We ablate the whole curriculum. Run the following command to start the training:
```bash
bash recipe/trustee/scripts/ablations/ablation_no_curriculum.sh
```

### Dimensions of Difficulty
We ablate the five dimensions of difficulty: tool, turn, user, system prompt, and criteria. After setting the appropriate flags in the script, run the following command to start the training:
```bash
bash recipe/trustee/scripts/ablations/ablation_curriculum.sh
```

### Soft Curriculum
We ablate the soft curriculum. Run the following command to start the training:
```bash
bash recipe/trustee/scripts/ablations/ablation_soft_curriculum.sh
```

## Hyper-parameters
We ablate the hyperparameters. After setting the hyperparameters in the script, run the following command to start the training:
```bash
bash recipe/trustee/scripts/ablations/ablation_hyperparameters.sh
```
