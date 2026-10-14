# NoCatch Match Lab

**Live demo:** https://pinkal018.github.io/nocatch-match-lab/

Match Lab is the prototype from my project *NoCatch: A Machine-Learning-Based Matchmaking System*. NoCatch is an Indian companion-booking startup where users book "Mates" for social meetups. Today it recommends Mates mostly by age gap. This project tests whether a machine-learning ranking that uses the rest of the onboarding answers can do better, while staying fair and explainable.

You fill in the details NoCatch asks for when a user makes a booking, choose which models to compare, and the page shows the top 3 Mates each model picks and the reasons behind each pick. Everything runs in your browser, so nothing you type is sent anywhere.

All data is **synthetic** (`NoCatch_Synthetic_Dataset_v3.xlsx`). No real users are included.

## What the models look at

Each user and Mate pair is turned into 10 features:

| Feature | Meaning |
|---|---|
| `shared_interests` | number of interests they share |
| `shared_conversation_preferences` | conversation styles they share |
| `gender_match` | the Mate matches the user's preferred gender |
| `shared_languages` | languages they share |
| `current_state_match` | similar current life stage |
| `vibe_match` | same preferred vibe |
| `age_gap` | age difference in years |
| `age_score` | the age-based score NoCatch uses now |
| `mate_experience_level_ordinal` | Mate is New (0), Intermediate (1) or Experienced (2) |
| `shared_languages_missing` | flag for pairs with missing language data |

NoCatch's existing rule score is **not** used by the ML models here, because it can't be worked out for a brand-new profile. (The main analysis also tested an 11-feature version that includes it.)

## Models and results

All ML models are trained on a staged match-quality target (booking, then completion, rating and repeat booking) instead of "booking = success". Results are on the same held-out test users (293 of the 300 had a booking to rank against).

| Model | Type | Hit@3 | NDCG@5 | MRR |
|---|---|---|---|---|
| Logistic Regression | ML | 0.587 | 0.358 | 0.483 |
| Decision Tree (depth 5) | ML | 0.597 | 0.350 | 0.471 |
| Random Forest (300 trees) | ML | 0.601 | 0.358 | 0.477 |
| **SVM (RBF kernel)** | ML | 0.594 | **0.362** | **0.486** |
| k-Nearest Neighbours (k = 151) | ML | 0.601 | 0.356 | 0.474 |
| k-means (6 clusters) | ML, unsupervised | 0.567 | 0.336 | 0.456 |
| Current rule | Baseline | 0.573 | 0.332 | 0.461 |
| Age only | Baseline | 0.539 | 0.323 | 0.433 |

Every ML model ranks Mates better than the two baselines. A simple content-similarity method (not in the demo) reached NDCG@5 of 0.353, so the profile answers themselves carry most of the signal. The SVM is best on NDCG@5, about 12% above age only. The gains are modest and come from synthetic data, so they need to be confirmed on real data before NoCatch changes its live ranking.

SVM, k-NN and k-means were added after my supervisor suggested testing more algorithms. k for k-NN was picked by 5-fold cross-validation on training users, and the number of k-means clusters by silhouette score. Neither choice looked at the test users.

## How each model explains its picks

- **Logistic Regression:** each feature's push up or down (`coefficient × (value − training average)`).
- **Decision Tree and Random Forest:** exact TreeSHAP values, plus the yes/no path the tree followed.
- **SVM:** exact Shapley values against an average training pair.
- **k-NN:** the most similar past pairings and how they turned out.
- **k-means:** what makes the pair's cluster different from an average pairing.
- **Current rule:** the page uses a linear copy of the rule score built from the same 10 features (R² ≈ 0.82 on test data). The results table uses the original rule score.

## Checking it works

The browser versions of the models are tested against the original Python libraries (scikit-learn and shap):

```
node test_tree_shap.js
node test_extra_models.js
```

## Files

| File | What it is |
|---|---|
| `index.html` | The app |
| `tree_shap.js`, `extra_models.js` | TreeSHAP, SVM, k-NN and k-means running in the browser |
| `mates_data.json` | The candidate Mates |
| `demo_model.json`, `dt_model.json`, `rf_model.json`, `svm_model.json`, `knn_model.json`, `kmeans_model.json`, `rule_model.json` | The exported models |
| `comparison_summary.json` | Test results shown on the page |
| `shap_reference.json`, `extra_reference.json` | Python reference values for the tests |

The model files are exported by three scripts in the project's analysis code (`Code_and_Analysis/Match_Lab_Scripts/`), run in this order: `export_demo_model.py`, `export_comparison_models.py`, `export_extra_models.py`.

This is a research prototype, not NoCatch's live system.

AI acknowledgement: Claude helped write code and documents in this repo. The project design, analysis and explanations are mine, and AI use is acknowledged in my SCI701 report.

Pinkal Makwana (1211769), University of the Sunshine Coast, SCI701, Trimester 2 2026

