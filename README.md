# TheorySol — learning to read a fixed map

24 November 2026. Companion to [Theory](https://github.com/anttiluode/Theory), especially its Pythia access results and the question of whether learning flexible access to a representation becomes harder as the number of distinct possible questions increases.

**Scope:** controlled synthetic reader experiments, not a model of child development or an explanation of Pythia. The map is given and frozen; only the reader learns. Both experiments were executed, not merely proposed. The results are descriptive five-seed pilots, not a claim of novelty.

## Gate 1 — more questions, same map

Each held-out object has 16 independent binary attributes and 16 nuisance coordinates, mixed by a randomly signed/permuted orthonormal Hadamard matrix into 32 numbers. Attribute noise SD is 0.2. The known inverse (not available to trained readers) classifies the tested attributes perfectly. Train and test objects are separate.

Vary the number of active tasks K = 1, 4, 8, 16. Each task sees exactly the same 256 unique training objects, independently of K, and the query vocabulary always has 16 slots. Each reader uses logistic loss and online SGD with the same learning rate (0.025). Five seeds, widths 8 and 24, no hyperparameter selection on test scores.

- **Concat:** tanh hidden layer, query-dependent additive biases (equivalent to a one-hot query concatenated with the input), one shared output.
- **Heads:** same-size shared tanh hidden layer, explicitly query-selected output head.
- **Independent:** direct linear logistic head for each query, no shared hidden layer. A strong ordinary baseline, not a matched-FLOP ablation.

Mean held-out accuracy at 256 examples per task (width 24 where applicable):

| Active tasks | Concat | Heads | Independent |
|---:|---:|---:|---:|
| 1 | 99.53% | 98.28% | 100.00% |
| 4 | 69.49% | 97.66% | 99.96% |
| 8 | 63.50% | 96.41% | 99.94% |
| 16 | 58.00% | 91.68% | 99.95% |

Concat has 1,201 parameters; heads 1,192; independent 528 (all 16 query slots allocated even when some are unused). At equal total training presentations, the gap also reflects fewer examples per task; that comparison is reported separately in JSON. At width 8, heads reach only 74.32% with 16 tasks. Width 24 improves heads, but not the concat reader within the tested exposure and optimizer.

The independent baseline's task-0 learning curve is checked to be exactly invariant to K in every seed. This guards against changing the map or task exposure accidentally.

## Gate 2 — labels are not the same as relationships

Hold 16 query labels, width 24, 512 training objects, updates per label, and initialization fixed. Vary the number U of genuinely different attributes requested: 1, 4, or 16. For U=1 all queries ask the same thing; for U=4 every four queries share a function. The readout labels do not encode the answer itself.

At evaluation, shift the query by one position while scoring the original question. If a reader actually selects the requested attribute, shifting to a different attribute should destroy the answer. This is an intervention on the query, not an attention visualization.

| Distinct relationships | Reader | Correct query | Wrong query | Mean worst-query accuracy |
|---:|---|---:|---:|---:|
| 1 | Concat | 100.00% | 100.00% | 100.00% |
| 4 | Concat | 68.55% | 68.36% | 64.53% |
| 4 | Heads | 100.00% | 48.75% | 100.00% |
| 16 | Concat | 58.96% | 58.85% | 49.53% |
| 16 | Heads | 98.77% | 49.82% | 96.41% |
| 16 | Independent | 99.99% | 49.89% | 99.84% |

The population-optimal **query-blind** reader predicts the majority of the underlying independent fair binary attributes. Its expected accuracy is

`sum_{j=0}^U binom(U,j) * max(j,U-j) / (U*2^U)`.

For U=4 this is 68.75%; for U=16 it is 59.82%. The concat results are very close to these ceilings, and shifting the query barely changes them. The explicit heads, by contrast, depend strongly on which query is selected. These two checks suggest the concat reader largely ignores the query under the tested training conditions. They do not prove that it cannot learn a better solution.

## What survived

1. Holding the map and per-query training exposure fixed, distinct readout demands can produce a large access deficit for a shared query-conditioned reader.
2. The deficit is not caused by the mere number of query labels: the same reader gets 100% when sixteen labels ask one relationship.
3. Explicit routing nearly eliminates the deficit in this constructed world. More training also helps the wider heads (91.64% at 256 to 98.77% at 512 examples per label).
4. Independent linear heads are the strongest baseline. A complicated shared reader has not earned an efficiency claim.

## What has not been established

- No universal developmental law. The map is deliberately easy and linearly decodable, not learned from a corpus.
- No comparison of brains or species, and no claim about dendritic anatomy or consciousness.
- The shared concat network's failure could be optimization, inductive bias, initialization, or insufficient width/training. The architectures have different active computation and parameter counts; do not call this a pure capacity experiment.
- In Gate 2, duplicated relationships receive more gradient presentations of the same training objects. Unique objects, total updates, and per-label exposure are matched, but per-relationship updates are not.
- Five seeds and a single synthetic generator are a pilot. No preregistered confirmatory test or external validation.

## Reproduce

Node.js 18+, no dependencies, no GPU. From the repository root:

```bat
node reader_complexity.js > runs\reader_complexity.json
node reader_control.js > runs\reader_control.json
```

The committed JSON files contain all seeds, checkpoints and the reported summaries. Gate 1 stores per-query accuracy as well.

- [Gate 1 source](reader_complexity.js) · [raw results](runs/reader_complexity.json)
- [Gate 2 source](reader_control.js) · [raw results](runs/reader_control.json)

## Next discriminators

A stronger test would use the same number of distinct questions and vary overlap/correlation of their attributes; match active parameter and compute budgets; sweep optimization on separate development seeds; and test an unseen set of questions rather than just unseen objects. In the actual Pythia line, measure category margins and real/random residue curves across checkpoints, with directional sensitivity and prompt alternatives, before attributing the native-answer gap to a particular circuit. A permanent yes/no-interface gap does not mean a model lacks all native category access.

The modest engineering hypothesis: give a learned map a separate, explicitly query-conditioned read path, and test whether that path can generalize to new questions without sacrificing the map or its other distinctions. That is a testable successor to the Sihti/GraphNN line, not a new theory of intelligence.
