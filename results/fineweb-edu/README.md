# FineWeb-Edu comparison

Both matched arms are complete. Each consumed 999,948,288 tokens of the same packed
corpus, in the same order, from the same initialization, under the same optimizer and
schedule. The H2 contract of docs/experiments.md is satisfied.

| Split | Tokens | SHA256 |
| --- | ---: | --- |
| Train | 1,000,000,000 | `c42f4bffe9e225eca05ee5fafa9bfc42875d014125f18f47919be611f775a413` |
| Validation | 10,000,000 | `61ecc77c5d586daa10c5938a6719463380e8e64347c0194e09635651561d4330` |

The data revision, tokenizer revision, packing length, and dtype are recorded in
`data_metadata.json` (train arm copy: `../fineweb-edu-ntp/data_metadata.json`).

## Matched result (H2)

| | NTP-only | NTP+NCP |
| --- | ---: | ---: |
| Tokens | 999,948,288 | 999,948,288 |
| Wall time | 9.08 h | 11.05 h |
| Tracked cost estimate | $24.85 | $30.27 |
| Final held-out NTP loss | **2.5135** | **2.5142** |
| Final held-out perplexity | 12.35 | 12.36 |
| Held-out loss @600M | 2.5113 | 2.5115 |

Final held-out NTP loss differs by **+0.0007 (+0.03%) in favor of the NTP-only arm** —
neutral within run noise. Full records: `../fineweb-edu-ntp/README.md` and
`../fineweb-edu-ncp/README.md`.

## Concept-path findings (H3/H6)

Evaluated on identical batches at the final NCP checkpoint:

- **Zero-feedback delta +0.1050** (2.5142 -> 2.6192): the decoder makes material use of
  predicted concept feedback;
- **Shuffle-feedback delta +0.0008**: feedback from another sequence in the same batch is
  as good as its own — the use is sequence-generic, not sequence-specific;
- **Codebook perplexity 7.55, usage 85.6%**, grew monotonically from 2.93/74.8% at 100M
  tokens — no low-entropy codebook shortcut at this scale.

This combination — a neutral token-loss comparison with a richly used but
sequence-generic latent channel — is the matched-scale analogue of the TinyStories gate
finding and sharpens the open question: the concept path carries information the decoder
consumes, but nothing a pure token model misses at 1B tokens with this recipe. The
candidate next probes (scale, schedule, target geometry, conditional code entropy) remain
in docs/experiments.md.
