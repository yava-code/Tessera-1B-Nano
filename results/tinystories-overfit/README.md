# TinyStories architecture gate

This run checks whether the complete NCP path can be optimized before spending on matched
FineWeb-Edu continued pretraining. It is deliberately an overfit experiment, not a language
model quality result.

## Run

| Item | Value |
| --- | --- |
| Backbone | `HuggingFaceTB/SmolLM2-135M` |
| Hardware | Modal L4 |
| Packed train subset | 1,048,576 tokens, repeated 16 times |
| Packed validation set | 1,048,576 tokens |
| Tokens seen | 16,777,216 |
| Optimizer steps | 4,096 |
| Wall time | 3,455 s |
| Configured cost estimate | $0.883 |

The cost estimate uses the configured L4 rate and 15% overhead. It is not a Modal invoice.

## Optimization result

Means below compare the first and last ten logged train windows.

| Loss | First 10 | Last 10 | Change |
| --- | ---: | ---: | ---: |
| NTP | 1.937 | 1.107 | -42.9% |
| NCP | 1.332 | 0.788 | -40.9% |
| VQ | 1.334 | 0.784 | -41.3% |
| Total | 4.603 | 2.678 | -41.8% |

All three objectives fall without NaNs, OOM recovery, or a mid-run configuration change.
This passes the architecture gate.

## Held-out dynamics

| Tokens | NTP | NCP | VQ | Code usage | Code perplexity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,194,304 | 1.672 | 1.125 | 1.126 | 30.7% | 1.62 |
| 8,388,608 | 1.722 | 0.927 | 0.924 | 34.4% | 2.22 |
| 12,582,912 | 1.808 | 0.831 | 0.826 | 31.4% | 2.35 |
| 16,777,216 | 1.864 | 0.800 | 0.794 | 30.9% | 2.39 |

The auxiliary objectives keep improving while held-out NTP degrades after repeated passes
over the fixed train subset. That is expected overfit behavior and is why this run is not used
as evidence of generalization.

## Does the decoder use the concept path?

The same held-out batches were evaluated with predicted concept feedback, zero feedback, and
feedback shuffled across sequences at the same concept offset.

| Checkpoint | Predicted NTP | Zero NTP | Zero delta | Shuffle delta | Code usage | Code perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.39M tokens | 1.724 | 1.744 | +0.019 | -0.00000001 | 41.1% | 2.26 |
| 16.78M tokens | 1.872 | 1.900 | +0.028 | +0.000000004 | 40.6% | 2.43 |

Zeroing the channel hurts NTP, so the decoder uses it. Shuffling across sequences has no
measurable effect, however. Together with low effective codebook perplexity, this suggests
that the small overfit run learns a low-entropy control signal rather than strongly
sequence-specific discrete concepts.

This observation sets up measurements of conditional code entropy across sequences and, if
needed, a minimal diversity regularizer. A direct quantized-target pilot was also attempted,
but its NCP loss was near zero from initialization because transformed codewords were tightly
clustered. That result is recorded in
[`../tinystories-overfit-quantized`](../tinystories-overfit-quantized/README.md). The claim is
not that any of these changes improve language modeling; the matched FineWeb-Edu comparison
must establish that separately.

## Artifacts

- `metrics.jsonl`: complete train and periodic evaluation log;
- `step-00002048-eval.json`: 8.39M-token intervention evaluation;
- `step-00004096-eval.json`: 16.78M-token intervention evaluation;
- `experiment.json` and `data_metadata.json`: pinned run and data provenance;
- `step-*-trainer-state.json`: token, step, cursor, and wall-time records.
