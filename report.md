# Spam Detection Model Evaluation

## 1. Model Configuration

- Best parameters: `{'model__C': 2.0, 'model__class_weight': None, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 1), 'tfidf__sublinear_tf': True}`
- Cross-validation F1 score: 0.9631
- Final model: `Pipeline(steps=[('tfidf', TfidfVectorizer(sublinear_tf=True)), ('model', LogisticRegression(C=2.0, max_iter=2000, random_state=42))])`

---

## 2. Training Evaluation

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.9846 | 0.9560 | 0.9701 | 1000 |
| 1 | 0.9572 | 0.9850 | 0.9709 | 1000 |
| Accuracy | - | - | 0.9705 | 2000 |
| Macro Avg | 0.9709 | 0.9705 | 0.9705 | 2000 |
| Weighted Avg | 0.9709 | 0.9705 | 0.9705 | 2000 |

### Confusion Matrix

| Actual \ Predicted | 0 | 1 |
| --- | ---: | ---: |
| 0 | 956 | 44 |
| 1 | 15 | 985 |

### Performance Summary

- Accuracy: 0.9705
- Precision: 0.9572
- Recall: 0.9850
- F1-score: 0.9709

---

## 3. Testing Evaluation

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.9722 | 0.9589 | 0.9655 | 146 |
| 1 | 0.9737 | 0.9823 | 0.9780 | 226 |
| Accuracy | - | - | 0.9731 | 372 |
| Macro Avg | 0.9730 | 0.9706 | 0.9717 | 372 |
| Weighted Avg | 0.9731 | 0.9731 | 0.9731 | 372 |

### Confusion Matrix

| Actual \ Predicted | 0 | 1 |
| --- | ---: | ---: |
| 0 | 140 | 6 |
| 1 | 4 | 222 |

### Performance Summary

- Accuracy: 0.9731
- Precision: 0.9737
- Recall: 0.9823
- F1-score: 0.9780

---

## 4. Evaluation Conclusion

These are very good scores for a spam detector.

- Training F1 score: 0.9709
- Testing F1 score: 0.9780
- Testing accuracy: 97.31%
- Spam recall: 98.23% means the model catches most spam messages.
- False positives are low: only 6 in the test set.

The model is strong and production-ready for a basic spam filtering use case. If the goal is to minimize false positives even further, the decision threshold could be tuned, but the current results are already excellent for a spam classifier.