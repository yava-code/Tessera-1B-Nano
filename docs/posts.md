# Post drafts for the completed comparison

Three variants, same facts. Every number below is from the committed artifacts, not
paraphrased. The released checkpoint is Tessera-1B-Nano by Paragon Intelligence Labs.
Post nothing wider until the public checkpoint exists (Modal `huggingface`
secret -> `modal_publish.py`).

## 1. X / Twitter thread (EN)

**1/** We ran the cheapest controlled test of ConceptLM (Next Concept Prediction) we could
design: SmolLM2-360M, two arms, 1B tokens each, one pure NTP and one with the concept path
we release as Tessera-1B-Nano. Same data, same order, same everything. Total cost: ~$55.
Here's what happened. 🧵

**2/** First, the boring number: held-out loss 2.5135 (NTP) vs 2.5142 (NCP). +0.03% in
favor of plain next-token prediction. Neutral. If this were the whole story, we wouldn't
be posting a thread.

**3/** The whole story starts with the interventions. Take the trained NCP model and zero
its concept feedback at inference: loss jumps +0.105 nats (+4.2% ppl). The decoder
*demonstrably uses* the latent channel. That's a causal statement, not a correlation.

**4/** Is the codebook a shortcut? No. Effective perplexity 7.55, 85.6% of codes in use,
monotonically up from 2.93 at 100M tokens. Our earlier TinyStories gate fell into a
~2.4-ppl low-entropy shortcut; at 1B tokens that failure mode is gone.

**5/** Now the twist. Shuffle concept feedback between sequences in the batch: cost is
+0.0008, nothing. Feedback from *another sequence* works as well as your own. The decoder
consumes the channel, but what it extracts isn't sequence-specific.

**6/** We saw the same zero-hurts/shuffle-doesn't split at 135M on TinyStories. Two scales,
two data regimes, same structure. This split is the interesting object, not the 0.03%.

**7/** Context: in the original ConceptLM paper's own ablation, NTP+one auxiliary is
*worse* than pure NTP; only the full triple wins, and their gains appear at 8 to 300B
tokens. Our neutral result at 1B mirrors that structure one scale down.

**8/** Everything is open: code, causal leakage tests, SHA-pinned data caches, full metric
logs, both checkpoints' eval JSONs. Two matched 1B-token runs, zero restarts, zero NaNs.
Next probes are stated in advance. Code and whitepaper:
github.com/yava-code/Tessera-1B-Nano. Weights: hf.co/yava-code (Tessera-1B-Nano,
Tessera-1B-Nano-Base).

## 2. Reddit / Hacker News longread (EN)

**Title:** Tessera-1B-Nano: we reproduced Next Concept Prediction at 360M for ~$55, the token loss stayed neutral, but the decoder demonstrably uses the concept channel (and shuffling it costs nothing)

**Body:**

We independently reimplemented ConceptLM's Next Concept Prediction recipe on
SmolLM2-360M and ran the most controlled comparison we could afford: two arms, pure
next-token prediction vs NTP+concepts, each consuming exactly 999,948,288 tokens of the
same packed FineWeb-Edu cache, in the same order, from the same initialization. No
restarts, no NaNs, tracked compute about $55 for both runs combined. The checkpoint from
the NTP+concepts arm is released as Tessera-1B-Nano (Paragon Intelligence Labs).

The headline is neutral: held-out loss 2.5135 vs 2.5142 (+0.03% for the baseline). If you
stop reading there, you'll miss the parts we actually find interesting:

1. **The concept channel is causally used.** Zero the predicted concept feedback at
   inference and held-out loss rises +0.105 nats (+4.2% perplexity). The decoder
   measurably depends on what the concept path feeds it.

2. **The learned vocabulary is rich.** Codebook effective perplexity 7.55 with 85.6%
   usage, grown monotonically over the run (2.93 -> 7.55). Our earlier TinyStories gate
   hit a low-entropy shortcut (~2.4 ppl); at 1B tokens that failure mode is gone.

3. **But the use is sequence-generic.** Replace each sequence's concept feedback with
   another sequence's from the same batch: cost +0.0008, i.e. nothing. We then replaced
   it with the most *similar* sequence's feedback: +0.0008 again. The decoder consumes
   something, but neither sequence nor even topic identity within a batch. The same
   split appeared at our 135M TinyStories gate, replicated across two scales.

For calibration: the original paper's own ablation shows NTP+either auxiliary *alone* is
worse than pure NTP, and only the full loss triple wins at 8–300B-token scales. A neutral
result at 1B tokens with sharply falling auxiliary losses mirrors that structure rather
than contradicting it. Mechanistically, concept layers run on T/k = 256 positions per
sequence here, a coarse summary channel, which is consistent with what the shuffle
control measures.

Methodology notes that mattered to us: causality is enforced by construction and tested
(prefix changes never change earlier logits); held-out comparisons are batch-exact (fixed
seed), not just dataset-exact; we aborted a quantized-target pilot when its auxiliary loss
sat at ~1e-6 and wrote down the prerequisite for retrying it (normalized codewords) rather
than reporting a vacuous "training works"; next probes are stated in advance.

Everything is open: training code, 21-test suite, SHA-pinned data caches, full metric
logs, eval JSONs with all three intervention modes, and both arms' records.

## 3. Короткий пост (RU)

Воспроизвели Next Concept Prediction (ConceptLM) на SmolLM2-360M и прогнали самое
честное сравнение, которое могли себе позволить: два arm'а, 1B токенов каждый,
одинаковые данные в одинаковом порядке, с концепт-путём и без. Чекпойнт с концепт-путём
выпускаем как Tessera-1B-Nano (Paragon Intelligence Labs). ~$55 за оба прогона.

Заголовок скучный: лосс 2.5135 vs 2.5142 (+0.03% в пользу обычного NTP). Нейтрально.

А теперь интересное:

- Зануляем концепт-фидбек на инференсе: +0.105 натов (+4.2% ppl). Декодер *реально
потребляет* канал: это каузальное вмешательство, не корреляция.

- Кодбук живой: эффективная ppl 7.55, использование 85.6%, монотонный рост с 2.93.
Low-entropy шортката, в который падал наш TinyStories-гейт, на 1B токенов нет.

- Но! Шафлим фидбек между последовательностями: +0.0008, то есть ничего. Фидбек чужой
последовательности почти так же хорош, как свой. Канал несёт топику, не идентичность
текста. Тот же сплит мы видели на 135M, воспроизведено на двух масштабах.

В оригинальной статье NTP+одна вспомогательная тоже хуже чистого NTP, выигрывает только
полная тройка, и на 8–300B токенов. Наш нейтральный результат на 1B повторяет эту
структуру на масштаб ниже, а не противоречит ей.

Всё открыто: код, каузальные тесты, хэши датасетов, полные логи метрик. Следующие пробы
сформулированы заранее.
