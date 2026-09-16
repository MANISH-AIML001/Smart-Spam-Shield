# Training Evaluations 
Best parameters: {'model__C': 2.0, 'model__class_weight': None, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 1), 'tfidf__sublinear_tf': True}
Cross-validation F1: 0.9631
==================================================
Prediction model  Pipeline(steps=[('tfidf', TfidfVectorizer(sublinear_tf=True)),
                ('model',
                 LogisticRegression(C=2.0, max_iter=2000, random_state=42))])
==================================================
Classification report:
               precision    recall  f1-score   support

           0     0.9846    0.9560    0.9701      1000
           1     0.9572    0.9850    0.9709      1000

    accuracy                         0.9705      2000
   macro avg     0.9709    0.9705    0.9705      2000
weighted avg     0.9709    0.9705    0.9705      2000

Confusion Matrix: 
 [[956  44]
 [ 15 985]]
Performance Matrix
Accuracy: 0.9705
Precision: 0.9572
Recall: 0.9850
F1: 0.9709

# Testing Evaluation 
==================================================
Prediction model  Pipeline(steps=[('tfidf', TfidfVectorizer(sublinear_tf=True)),
                ('model',
                 LogisticRegression(C=2.0, max_iter=2000, random_state=42))])
==================================================
Classification report:
               precision    recall  f1-score   support

           0     0.9722    0.9589    0.9655       146
           1     0.9737    0.9823    0.9780       226

    accuracy                         0.9731       372
   macro avg     0.9730    0.9706    0.9717       372
weighted avg     0.9731    0.9731    0.9731       372

Confusion Matrix: 
 [[140   6]
 [  4 222]]
Performance Matrix
Accuracy: 0.9731
Precision: 0.9737
Recall: 0.9823
F1: 0.9780