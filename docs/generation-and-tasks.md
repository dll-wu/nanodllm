# Generation and task scoring

`nanodllm-generate` and `nanodllm-tasks` use the native causal-clean block transformer. By default they accept a local portable `.weights.pt` file, verify the tokenizer identity embedded in it, and load the model state strictly. For a trusted native training checkpoint, pass `--training-checkpoint` explicitly. HF training files need their matching data identity sidecar beside the checkpoint or supplied with `--data-identity`. Pass `--checkpoint-sha256` to bind either input to an expected digest. HF checkpoints require `--tokenizer` with a local tokenizer directory. Byte checkpoints use the built-in tokenizer.

## Generate from a training run

After training `recipes/tiny-block.json`, give the model a literal prompt. This command prints a JSON report to stdout. Its first continuation is in `samples[0].text`.

```sh
nanodllm-generate --checkpoint runs/tiny-block/model_latest.pt --training-checkpoint --prompt "The lamp" --capacity 16 --seed 7
```

Pass `--output FILE` to save the report and print a summary instead. The cap counts generated positions. With custom prompts, the default is the smaller of 256 and the available context after the longest prompt. For a portable export, supply the `.weights.pt` file and omit `--training-checkpoint`. [Train and score the tiny model](evaluation.md#score-a-tiny-block-checkpoint-on-local-text) · [Portable weights](portable-weights.md)

## Offline CPU demo

The demo creates two randomly initialized, one-layer block checkpoints, three original synthetic tasks and two prompts in an empty directory. It exercises model loading, scoring, paired comparison and decoding without a dataset download.

```sh
demo=$(mktemp -d)
python scripts/tasks_demo.py "$demo"

for seed in 11 12; do
  nanodllm-tasks score --bundle "$demo" --roster "$demo/roster.json" \
    --checkpoint "$demo/demo-$seed.weights.pt" --output "$demo/tasks-$seed.json"
done
nanodllm-tasks compare "$demo/tasks-11.json" "$demo/tasks-12.json" \
  --output "$demo/paired.json"

nanodllm-generate --checkpoint "$demo/demo-11.weights.pt" \
  --prompts "$demo/prompts.jsonl" --capacity 12 --output "$demo/generation.json"
```

The last command produces four continuations, one per prompt and sampling seed. Each output row includes the generated text and actual forward counts. The task reports contain numeric scores, selection and input hashes, and row indices. Keep local reports with their inputs.

## Prepare the corrected 21-task panel

Obtain a locally authorized copy of the original nanochat evaluation bundle under the terms for its constituent datasets. The observed ZIP has SHA-256 `90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200`. Supply the ZIP directly or a directory with its `eval_data/` subtree. The evaluator opens the 20 listed JSONL files. Two HellaSwag tasks share one file. Jeopardy is excluded. Obtain the matching 32k checkpoint tokenizer separately. The task roster records paths, exact delimiters, shot counts, raw and selected counts, file hashes, input hashes and both centering baselines. [Bundle source](https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip) · [Task roster](../paper/tasks/core21.json) · [nanoLLaDA reference revision](https://github.com/Lukas-Xue/nanoLLaDA/tree/1769106a2c9a767de5058022e7940cffb0a40e0a) · [DCLM v2 baseline source](https://github.com/mlfoundations/dclm/blob/361714bdd60bb9b7f4b2d8354cebbf0dec0c329e/eval/eval_meta_data.csv)

Set `BUNDLE` and `TOKENIZER` to those local inputs. Verify all rows and tokenized spans without loading a checkpoint.

```sh
nanodllm-tasks audit --bundle "$BUNDLE" --tokenizer "$TOKENIZER"
```

A complete audit covers 8,845 items. The 11 multiple-choice tasks contribute 4,830, two schema tasks contribute 773 and eight completion tasks contribute 3,242. Each task shuffles its original row indices with Python RNG seed 1337 and retains up to 500. For each shuffled position `i`, seed `1234 + i` draws demonstrations without replacement from the other retained positions. Use `--task LABEL` and `--limit N` on the score command for a bounded run. The limit applies after the demonstration pool has been fixed.

Each demonstration supplies its gold response and ends with two newlines. Multiple-choice examples join the query, literal task delimiter and option. The common token prefix includes the context-only encoding so shared answer tokens stay in the scored span. Schema examples vary the context and score the common token suffix. Completion tasks preserve context whitespace and score every token in the full encoding that overlaps the answer characters, including a boundary merge. Each input prepends BOS. Left crops round up to a multiple of BL32 to fit 1,024 tokens. A cropped input replaces the first retained context token with BOS, keeping the answer and its block alignment intact.

## Score and compare checkpoints

```sh
for recipe in scratch conversion; do
  nanodllm-tasks score --bundle "$BUNDLE" --tokenizer "$TOKENIZER" \
    --checkpoint "$recipe.weights.pt" --device cuda --output "$recipe.tasks.json"
done
nanodllm-tasks compare scratch.tasks.json conversion.tasks.json \
  --output conversion-minus-scratch.json
```

The primary readout is `mean_nelbo__block_full_mask`. Candidate scoring uses the native doubled stream with a causal clean prefix. Prior blocks are gold. For each target group of at most seven tokens, the score integrates all nonempty masked subsets. Larger groups use 32 stratified mask-cardinality draws in each of four replicates with seed 1337. Equal-length choices and paired checkpoints share plans. Ranking minimizes mean NELBO per target token, with the first minimum winning a tie. Summed NELBO is retained as a control. Inference batches eight corruption rows, with eager BF16 on CUDA, TF32 disabled, FP32 cross entropy and higher-precision reductions. CPU runs use FP32.

Completion uses raw-vocabulary argmax and a supplied gold answer length. It masks each entire active target group and supplies earlier blocks as gold. Exact match requires every target token to match. The sequential control queries one masked position at a time with the preceding gold tokens. The decoder never chooses an answer length in either task readout.

The report gives all four fixed ranking/completion combinations. Raw accuracy averages task accuracies with equal weight. Centered v1 and v2 apply `(accuracy - baseline) / (1 - baseline)` within each task before averaging. The paired report gives right-minus-left differences in percentage points, with 10,000 item resamples within each dataset at seed `20261118 + depth`. It resamples the two HellaSwag variants jointly and checks item order, demonstrations, prepared spans, mask plans, tokenizer and evaluation settings across models. The primary per-task breakdown includes gained and lost items and each task’s contribution to the raw mean difference. The intervals condition on these checkpoints, tasks and draws. The study used depths 12 and 20. Custom subsets have their own denominators.

## Generate with the causal-prefix KV cache

For the larger-model study, supply the local ClimbMix validation file `shard_06542.parquet`. The 32 generation prompts use 192 tokens including BOS, with 16 documents in each of the short and long continuation strata. The manifest stores row numbers, strata, stable IDs and a joint prefix hash. The runner verifies the shard, tokenizer and reconstructed prompt tokens. The associated predictive panel uses different rows. [Generation panel](../paper/tasks/generation32.json) · [ClimbMix source](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)

```sh
nanodllm-generate --checkpoint block.weights.pt --tokenizer "$TOKENIZER" \
  --study-parquet "$SHARD" --device cuda --policy left-to-right --k 1 --top-p 1 \
  --block-length 32 --output k1.json

nanodllm-generate --checkpoint block.weights.pt --tokenizer "$TOKENIZER" \
  --study-parquet "$SHARD" --device cuda --policy sample-redraw --k 4 --top-p 0.9 \
  --block-length 32 --output k4.json
```

Alternatively, pass literal `--prompt` arguments or `--prompts` with one JSON object per line. Each row supplies a unique stable `id` and either `text` or `prefix_ids`. For text the CLI prepends BOS. Explicit token IDs include the original BOS and preserve the exact prompt. Prompt IDs participate in RNG derivation, so preserve them across runs. This example uses an original prompt.

```json
{"id":"dial","text":"The dial settled near "}
```

The study recipe defaults to four commitments per pass, sampled-token confidence with a fresh draw, temperature 1, top-p 0.9 and capacity 256. The default block length comes from the checkpoint. Override it with `--block-length`. The study used 16, 32 and 64 for K4. The K1 left-to-right comparison used BL32 and unrestricted sampling. Both use proposal seeds 2026110201 and 2026110202, for 64 samples per study setting. Supply one or more `--seed` flags to replace that pair.

For the keep/redraw comparison use `--k 4 --top-p 1 --block-length 32` with `--policy sample-keep` or `--policy sample-redraw`. Both select positions by the untempered, legal-vocabulary probability of the sampled proposal. The former keeps that token. The latter independently redraws from the same forward logits. The CLI uses the block-recipe sampler for every mode. The earlier 20M experiment used its own reporting prompts, BF16 cache, a different commitment RNG stream and a different tie rule. These commands reproduce the comparison design. Exact random trajectories for that earlier experiment require its original sampler.

The model finishes the active block and returns content before its first EOS. The output cap counts available generation positions. An immediate EOS gives zero content tokens. A sample with fewer than four content tokens has no repeat-four-gram denominator. Other samples count repeat occurrences divided by the number of four-grams in the generating tokenizer. The summary averages those fractions across eligible samples. Content tokens per forward includes the prompt prefill, all active passes and clean cache appends. The final block needs no append. Seconds and content tokens per second use the generation interval, synchronized on CUDA. Study summaries average per-sample rates. Timing depends on hardware and prefix length.

## Score the generated text with a local reference

Add `--reference-model "$REFERENCE"` to a generation command. Supply a complete local Hugging Face causal model and tokenizer directory with BOS and PAD IDs. The paper used the Gemma 4 31B base checkpoint and a Transformers build with `Gemma4ForConditionalGeneration` support. Reference-model scoring runs after generation and outside its timing interval. [Gemma 4 31B base](https://huggingface.co/google/gemma-4-31B)

The scorer independently encodes the decoded prefix and continuation as ordinary text, prepends one reference BOS, and scores only continuation tokens at the preceding causal positions. It applies no chat template or terminal EOS. Inputs must fit the 4,096-token protocol limit and the reference’s context window. For Gemma it applies the model’s final softcap in FP32 before cross entropy, projects up to 512 target positions per chunk, and batches two samples. The summary pools total NLL in nats, reference tokens and unchanged UTF-8 bytes. PPL is `exp(total NLL / tokens)`. BPB is `total NLL / (bytes * log(2))`. Empty continuations add zero to all denominators.
