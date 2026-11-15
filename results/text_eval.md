# Text model evaluation (Dreaddit test set)

Model: the owner's fine-tuned BERT in `artifacts/text/`. Produced by `scripts/eval_text.py`.
Test set: 715 posts.

The model's original training split is unknown, so some of these posts may have been seen in training.

| metric | value | 95% interval |
|---|---:|---:|
| accuracy at 0.5 | 0.9077 | +/- 0.0217 |
| F1 | 0.9115 | +/- 0.0215 |
| AUC | 0.9669 | +/- 0.0115 |
| always-guess-majority accuracy | 0.5161 | |

## Probe sentences (negated, calm, and neutral)

These are not from Dreaddit; they test the model's weak spots by hand.

| score | sentence |
|---:|---|
| 0.6023 | I am not stressed at all, life is great. |
| 0.4133 | Today was calm, no stress at all. |
| 0.3148 | The weather is nice. |
| 0.7161 | I feel relaxed and rested after a good night's sleep. |
| 0.3963 | Everything is under control and going smoothly. |
| 0.3700 | I had a quiet, peaceful afternoon reading a book. |
| 0.1935 | No worries here, I am doing perfectly fine. |
| 0.1270 | I love my job and my family. |
| 0.3746 | We went for a gentle walk in the park. |
| 0.3715 | I am content and at ease today. |
| 0.3966 | I went to the store and bought milk. |
| 0.4548 | The train arrives at nine in the morning. |
| 0.2765 | There are twelve months in a year. |
| 0.2134 | She painted the fence a light shade of blue. |
| 0.1371 | The recipe needs two cups of flour. |
| 0.1372 | Our meeting is scheduled for Tuesday. |
| 0.3806 | He parked the car in the usual spot. |
| 0.1725 | The library closes at eight tonight. |
| 0.4879 | Water boils at one hundred degrees Celsius. |
| 0.4830 | The cat slept on the windowsill all day. |
