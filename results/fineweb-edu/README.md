# FineWeb-Edu cache

The shared packed corpus for the matched continued-pretraining runs is prepared in the Modal
`ncp-smol` volume. Both arms will read these exact files:

| Split | Tokens | SHA256 |
| --- | ---: | --- |
| Train | 1,000,000,000 | `c42f4bffe9e225eca05ee5fafa9bfc42875d014125f18f47919be611f775a413` |
| Validation | 10,000,000 | `61ecc77c5d586daa10c5938a6719463380e8e64347c0194e09635651561d4330` |

The data revision, tokenizer revision, packing length, and dtype are recorded in
`data_metadata.json`. This is corpus provenance, not a training result; neither FineWeb-Edu
arm has started yet.
