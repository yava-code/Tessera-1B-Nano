# Design note

## Scope

The target is a readable ConceptLM-style implementation small enough to train on rented
hardware. It preserves the method's two-level causal structure and drops scale-specific
machinery that cannot be evaluated credibly under the budget.

## Forward path

Let `h_t` be the token state after the first backbone block and `k = 4`. For each complete
chunk, the encoder uses mean pooling:

```text
c_i = mean(h_4i, ..., h_4i+3)
```

`c_i` is split along the hidden dimension. Each segment is matched to its nearest entry in a
SimVQ-style codebook: fixed random anchors pass through a learned two-layer transform. With
SmolLM2-360M, the 960-dimensional state is divided into 15 segments of width 64, each with
64 codewords.

Two causal decoder blocks process the concept sequence. A segmented classifier predicts the
next product code, and its soft expected codeword produces the continuous feedback vector.
That vector is repeated across the next token chunk and added before backbone block 2.

## Objectives

The training objective is:

```text
L = L_ntp + alpha L_ncp + beta L_vq
```

- `L_ntp` is standard next-token cross entropy.
- `L_ncp` is mean squared error between the predicted code embedding and the detached next
  pooled concept.
- `L_vq` moves transformed codewords toward detached pooled concepts.

The default experiment uses `alpha = beta = 1`. Keeping the concept target detached prevents
the auxiliary objectives from finding a trivial solution by moving the token representation
toward the prediction.

The paper writes the NCP target as the continuous pooled concept, while the released Llama
implementation targets its selected codeword. The main config follows the paper. The
`ncp_target: quantized` switch keeps the released-code form available as a controlled ablation.

## What is intentionally smaller

This implementation follows the simple ConceptLM path: product VQ, a thin causal concept
module, and residual concept injection. It does not claim the NCP-ArchPreview recipe.
Specifically, it omits the 8.9B scale, eight-layer concept stack, iterative residual coding,
cross-scale residual connections, Muon setup, and large-corpus schedule.

Those omissions are structural choices, not hidden caveats. They isolate the central NCP
objective at a scale where a token-matched control is affordable.

The predicted feedback is detached before token injection, matching the released Llama path.
The NCP objective still updates the concept module and propagates into the encoder; the NTP
objective trains the ordinary token path without turning the latent code into an unconstrained
side channel.

## Checkpoint format

NCP checkpoints use the Hugging Face `PreTrainedModel` format and contain the complete token
backbone, concept layers, product codebook, and head. Optimizer state and deterministic data
cursor are stored beside the weights for preemption-safe resume.

The code comparison in this note used upstream revision
`a0ab281286f5c0337c35de3181cc992c562eacaa`.
