# ai-self-learning-code

A single-file logistic-regression demo on a 6-point 2-D toy dataset.
"Self-learning" is a stretch — it's a linear classifier trained with
SGD — but it's the smallest thing that shows a model actually fitting
data and reporting an accuracy.

## Files

- `logistic_regression.py` — current, TF2/Keras. Run this.
- `doom` — the original TF1 snippet. Kept for history; **does not run
  under modern TensorFlow.** See "What was broken" below.

## Usage

```sh
python3 -m pip install -r requirements.txt
python3 logistic_regression.py
```

Expected output:

```
Hypothesis (probability of class 1):
  [1.0, 2.0]: 0.02..
  [2.0, 3.0]: 0.05..
  [3.0, 1.0]: 0.15..
  [4.0, 3.0]: 0.86..
  [5.0, 3.0]: 0.96..
  [6.0, 2.0]: 0.99..

Hard predictions: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
Accuracy: 1.0
```

Numbers will vary by initialization but the accuracy should be 1.0 once
training converges.

## What was broken in `doom`

The original file targeted TensorFlow 1.x:

- `tf.placeholder`, `tf.Session`, `tf.global_variables_initializer`,
  `tf.random_normal`, `tf.train.GradientDescentOptimizer` — all
  removed or moved under `tf.compat.v1` in TF 2.
- The identifier `tf.reduce_mean` was split across two lines
  (`tf.reduce_\n\nmean(...)`) — a `SyntaxError`.
- The prose sentence `"Let's add an additional optimizer..."` was
  pasted inline as Python — another `SyntaxError`.
- The bottom `for` loop had unindented body lines — yet another
  `SyntaxError`.

The TF2/Keras rewrite in `logistic_regression.py` is ~40 lines
(comments included) and captures the same intent: sigmoid activation,
binary cross-entropy, SGD.

## License

MIT — see [LICENSE](LICENSE).
