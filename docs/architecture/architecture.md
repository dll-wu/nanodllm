# Implementation map

Editable and frozen baseline profiles enter the same [training loop](../../nanodllm/runtime/training_loop.py). `nanodllm-pretrain` accepts a JSON file and dotted overrides. The [recipe launcher](../../recipes/launch.py) binds dataset and parent paths, writes the resolved config, and invokes the same command under one process or `torchrun`. The [component replications](../../recipes/components/README.md) have a specialized single-GPU driver for their objective and compute controls. They reuse the native model, optimizer and data loader.

| To change | Read first |
| --- | --- |
| Which document tokens are visible and which contribute to the loss | [Packing](../../nanodllm/runtime/data.py), [noise](../../nanodllm/runtime/noise.py), [masked objective](../../nanodllm/runtime/mdlm_objective.py) |
| Attention within a block and across its prefix | [Model](../../nanodllm/runtime/model.py), [attention](../../nanodllm/runtime/model_attention.py), [block training](../../nanodllm/runtime/training_bd3lm.py) |
| Which positions are filled on each generation pass | [Block sampling](../../nanodllm/runtime/block_recipe_sampling.py) and [prefix cache](../../nanodllm/runtime/bd3_prefix_cache.py) |
| Batch, optimizer, schedule, or parent initialization | [Config](../../nanodllm/runtime/config.py), [optimizer](../../nanodllm/runtime/training_optim.py), [initialization](../../nanodllm/runtime/training_initialization.py) |
| How a checkpoint is scored | [Corpus evaluator](../../nanodllm/evaluation/cli.py), [generation](../../nanodllm/generation/cli.py), [tasks](../../nanodllm/tasks/cli.py) |

The reusable code lives in `nanodllm/`. An editable experiment belongs in `recipes/`; a frozen study profile, reference panel or figure input belongs in `paper/`. The paper's exact profiles call the shared operators. A custom experiment can train, generate and score with its own data and no paper assets. The top-level Python modules retain the preview's public import paths and delegate to those same operators.

See [configuration](configuration.md), [data and tokenization](data-and-tokenization.md), [training](training.md), [evaluation and sampling](evaluation-and-sampling.md), or [packaging](packaging.md) for details.
