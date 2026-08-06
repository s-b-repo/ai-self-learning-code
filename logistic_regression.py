#!/usr/bin/env python3
"""Tiny logistic-regression classifier on a 6-point 2-D toy dataset.

This is the TF2/Keras equivalent of the original TF1 snippet in `doom`.
The `doom` file is preserved for historical reference — it can't run
under modern TF because it uses the TF1 session API (tf.placeholder,
tf.Session, GradientDescentOptimizer) and contains a few literal syntax
errors on top of that.
"""

import numpy as np
import tensorflow as tf


def build_model():
    """One dense layer, sigmoid output — the entire logistic regression."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # 6 points, 2 features, binary label. The first three are class 0,
    # the last three are class 1.
    x = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], dtype=np.float32)
    y = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

    model = build_model()
    model.fit(x, y, epochs=1000, verbose=0)

    preds = model.predict(x, verbose=0)
    hard_preds = (preds > 0.5).astype(np.float32)
    accuracy = float((hard_preds == y).mean())

    print("Hypothesis (probability of class 1):")
    for pt, prob in zip(x, preds.flatten()):
        print(f"  {pt.tolist()}: {prob:.4f}")
    print("\nHard predictions:", hard_preds.flatten().tolist())
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
