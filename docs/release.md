# Local paper release candidate

This candidate updates the July preview at `d7690ed21384caf3ea5a0a1fb2fd8eab23b85b95`. Its reusable paper trainer and numerical operators were selected from research revision `0a046040e47542040c332d3d1881ca522b83577e`. The package and citation metadata identify the candidate as `1.0.0rc1`. No published tag is implied.

| Area | Public implementation and evidence |
| --- | --- |
| Training | `nanodllm.paper_runtime` owns the native paper operators. The profile catalog supplies per-size settings, budgets, world sizes, parent dependencies and historical controls. The original preview CLI remains available for existing configs. |
| Optimizer | The paper block recipe uses the pinned nanochat Muon matrix update with separate embedding and readout groups. Its source and license travel with the package. The adapter verifies their hashes. See the [notices](../THIRD_PARTY_NOTICES.md). |
| Evaluation | The project supplies corpus panels, conditional and cached block scoring, task scoring, generation diagnostics and paired uncertainty. Panel fingerprints bind the comparison to specific inputs. |
| Evidence | The CPU companion rebuilds the paper displays from curated numerical data. It records where only aggregates survived. Checkpoint or tokenizer identities identify required artifacts independently of whether their bytes are distributed. |
| Distribution | Code, manifests, numerical summaries and TeX exhibits are eligible for the source distribution. The wheel includes the native runtime, installed paper assets and numerical companion. Data shards, full trainer pickles, model weights, local experiment logs and cluster operations stay outside both packages. |

## Source and asset provenance

The original public CLI is retained at its existing entry point. The new runtime uses first-party model, attention, optimizer-adapter, data and evaluation code from the pinned research revision above. The nanochat optimizer and tokenizer training adaptation are covered by the [MIT notice](../THIRD_PARTY_NOTICES.md). The numerical companion ships derived losses, counts, statistical calculations and first-party figure sources. External benchmark items, corpus passages, generated excerpts, reference model files and the research tokenizer binaries are obtained separately according to the [data](paper-data.md) and [task](paper-tasks-generation.md) guides.

## Repeating the local checks

Run the lint, test and distribution checks from [packaging](architecture/packaging.md). The tiny corpus exercises CPU training, export, evaluation and fresh-parent initialization without external artifacts. The deterministic two-update CPU comparison with the pinned trainer produced identical model tensors for full-sequence AdamW and block Muon. Two-rank CPU training under Gloo resumed from step two and matched every tensor after four updates. A fresh AR parent initialized the diffusion child from both its trainer checkpoint and a portable export.

The GPU runs underlying the paper are historical measurements. Accelerator throughput, BF16 compilation, peak memory, NCCL and the large checkpoint replays require separate target-hardware qualification of this export. The checkpoint and data guides specify the material needed for those checks.

The numerical companion checks 67 TeX files, covering seven figures, thirteen tables and supplementary exhibits. Figure 2 has a parent-lineage caption correction, and Figure 3 omits the prompt and generation excerpts. See [reproduction boundaries](paper-results.md). The portable tokenizer digest was also verified on the original tokenizer with a locally trained tiny checkpoint. Its exported model produced identical CPU logits.

## Maintaining the public export

For an update, identify the source revision, port the required numerical and public workflow changes, and compare tiny runs with that source under the same inputs, seeds and software. Review both source and wheel inventories. Keep scheduler scripts, workstation paths, private logs and workspace-specific configuration out of the release. If upstream optimizer code changes, update its pinned hash and attribution together. Re-run the numerical companion when measurements, display precision or uncertainty calculations change.
