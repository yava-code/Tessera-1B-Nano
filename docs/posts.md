# Post drafts

Three variants, one hook - the same one the Space and the cards lead with: *we taught a
small LLM to predict its next thought; it reads those thoughts constantly, but cannot
tell whose they are.* Story first, numbers for the record. Every number below is from the
committed artifacts, not paraphrased. Everything referenced is live: the
[Space](https://huggingface.co/spaces/yava-code/tessera-comparison) (with the demo video
pinned at the top of the model cards and the README), the weights
([Tessera-1B-Nano](https://huggingface.co/yava-code/Tessera-1B-Nano),
[Tessera-1B-Nano-Base](https://huggingface.co/yava-code/Tessera-1B-Nano-Base)), and the
study repo. Full tracked GPU spend so far: ~$140 ($55 the 1024 comparison arms, $79 the
seq-4096 probe with its resume, the rest on the 135M gate, the h5 pilots, evals, and data
prep).

## 1. X / Twitter thread (EN)

**1/** We taught a small LLM to predict its next thought - and it reads those thoughts
constantly. But it can't tell whose they are. 🧵

We took an off-the-shelf 360M model, bolted on a tiny module that compresses every 4
tokens into one concept code, guesses the next code, and feeds the guess back into the
decoder. Then we trained two copies of it on the same billion tokens and watched.

**2/** Watch the demo first (60s, it's pinned at the top): left column writes, middle
column writes, right column is the concept model with its thought-feedback silenced. The
loss line shows how much the model minds. It minds.
hf.co/spaces/yava-code/tessera-comparison

**3/** The setup is the cheapest honest test of ConceptLM (Next Concept Prediction) we
could design: SmolLM2-360M, two arms, 1B tokens each, same data in the same order, same
schedule. One arm pure next-token prediction, one arm with the concept path. ~$55 of GPU
for both. Everything preregistered, everything open.

**4/** Now the strange part, measured on held-out data. Zero the concept feedback: loss
jumps +0.105 nats. The decoder leans on the channel. Causal intervention, not
correlation.

**5/** Replace the concept feedback with concepts predicted from a *different* sentence
in the batch: +0.0008, nothing. From a topically similar sentence: +0.0008. From
Wikipedia articles: +0.0016. From Python source code: +0.0011. Any partner works. The
model reads the channel at every step - and can't tell whose thoughts it's reading.

**6/** So what is it reading? We don't know yet, and that's the point. It's not sentence
identity, not topic, not domain. Whatever the concept path carries, the token path wants
it - and a pure token model trained identically does not miss it.

**7/** We pushed context 4x (seq 4096, same token budget): the pattern held, and the
zero-feedback penalty *grew* (+0.1373). The channel isn't a short-context artifact.
Also a cost finding: the concept path pays a quadratic price at long context, 1.8x the
baseline's wall time per token.

**8/** And yes: token loss is neutral at 1B tokens (2.5142 vs 2.5135). Same structure as
the original paper, where only the full triple wins at 8-300B tokens. Neutral with a
mystery is a finding, not a failure.

**9/** Everything is open: code, 21-test suite incl. causality tests, SHA-pinned caches,
full metric logs, both checkpoints, the 60s demo, preregistered next probes.
github.com/yava-code/Tessera-1B-Nano
hf.co/yava-code/Tessera-1B-Nano (+ -Base). Play with the Space first.

## 2. Reddit / Hacker News longread (EN)

**Title:** We taught a 360M LLM to predict its next thought; it reads them constantly but can't tell whose they are ($140, everything open)

**Body:**

We independently reimplemented ConceptLM's Next Concept Prediction on SmolLM2-360M:
every 4 tokens are pooled into one concept code, a small causal module predicts the next
code, and the prediction is fed back into the decoder while it writes. Two arms, same
billion tokens in the same order, same initialization - one with the concept path, one
without. The concept checkpoint is released as Tessera-1B-Nano with a live side-by-side
demo Space (and a 60-second demo video pinned on the model cards).

If you open the Space, this is what you'll see, and it's the honest summary of the whole
study: the decoder uses the concept channel constantly - silencing the feedback costs
+0.105 nats of held-out loss - but feed it concepts predicted from a *different* sentence
and nothing changes (+0.0008). We pushed on that gap with everything we had:

1. **Similar sentences**: swap in feedback from the most topically similar sequence:
   +0.0008. Even topic identity is absent.

2. **Foreign domains**: feedback computed by the same model on Wikipedia articles
   (+0.0016) or on Python source code (+0.0011) works as well as its own. Not domain
   either.

3. **Longer context**: re-run both arms at sequence length 4096, same token budget. The
   gap to the baseline stayed at +0.0006, the zero-feedback penalty *grew* to +0.1373,
   and the shuffle controls stayed at noise. The channel isn't a short-context artifact;
   its use scales with the number of in-window chunk decisions.

4. **The codebook is real**: effective perplexity 7.55, 85.6% of codes in use,
   monotonically up from 2.93. Our earlier 135M TinyStories gate fell into a ~2.4-ppl
   low-entropy shortcut; the gate run caught it cheaply and the 1B run left it behind.

5. **Token loss stays neutral** (2.5135 vs 2.5142 at 1B tokens), which mirrors the
   original paper's own ablation: NTP plus a single auxiliary is *worse* than pure NTP
   there, and their reported gains arrive at 8-300B tokens with the full loss triple.

So: a model that constantly consumes a signal that carries ... what? Not sequence, not
topic, not domain. That question - with preregistered predictions (three of four landed
at 4096; the fourth missed by 0.33 ppl and is recorded as a miss) - is the study.

Methodology notes people asked about last time: causality is enforced by construction and
regression-tested (changing a suffix never changes earlier logits); held-out comparisons
are batch-exact by fixed seed; we aborted a quantized-target pilot when its auxiliary
loss sat at ~1e-6 and wrote down the prerequisite instead of reporting a vacuous success
(with normalized codewords the repeat came out token-neutral); the seq-4096 NCP arm hit
its $40 budget guard at 789M tokens because the concept path pays a quadratic cost at
long context (15.0k vs 26.9k tok/s), and we resumed it to the full budget with the stop
documented.

Total tracked spend so far: ~$140 - $55 for the two 1024 arms, $79 for the seq-4096
probe (including the budget stop and resume), the rest on the 135M gate, the quantized
target pilots, evaluations, and data preparation. Everything is open: training code,
21-test suite, SHA-pinned caches, full metric logs, eval JSONs with all intervention
modes, both checkpoints, the demo.

## 3. Короткий пост (RU)

Мы научили маленькую LLM предсказывать следующую «мысль» - и она читает эти мысли
постоянно, но не может понять, чьи они.

Как устроено: каждые 4 токена сжимаются в один концепт-код, маленький каузальный модуль
угадывает следующий код, и эта догадка подаётся декодеру, пока он пишет. Два клона
SmolLM2-360M, один миллиард токенов, одинаковый порядок данных - с концепт-путём и без.
Это независимая репликация ConceptLM (Next Concept Prediction), всё открыто: ~$140 GPU за
всё исследование, чекпойнты, демо Space.

Что показало сравнение:

- Занулили концепт-фидбек на инференсе: +0.105 натов лосса. Декодер реально опирается на
  канал - каузальное вмешательство, не корреляция.

- Подсунули фидбек от *чужого* предложения в батче: +0.0008, ничего. От топически
  похожего: +0.0008. Из Википедии: +0.0016. Из питон-кода: +0.0011. Модель читает канал
  на каждом шаге - и не отличает свои мысли от чужих.

- Удлинили контекст в 4 раза (seq 4096, тот же бюджет): паттерн сохранился, а штраф за
  зануление вырос до +0.1373. Шорткат короткого контекста исключён.

- Сам токен-лосс при этом нейтрален (2.5135 vs 2.5142). В оригинальной статье выигрыш
  появляется только на 8-300B токенов с полной тройкой лоссов - наша нейтральность на 1B
  повторяет эту структуру, а не противоречит ей.

Что именно несёт канал, если не идентичность, не топик и не домен - открытый вопрос, ради
которого всё и затевалось. Следующие пробы сформулированы заранее, до прогона.

Демо (одинаковый seed = бит-в-бит те же генерации):
hf.co/spaces/yava-code/tessera-comparison
Веса: hf.co/yava-code (Tessera-1B-Nano, Tessera-1B-Nano-Base)
Код, whitepaper, логи: github.com/yava-code/Tessera-1B-Nano
