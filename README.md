# NoCatch Match Lab

A live, in-browser demo of NoCatch's compatibility models: fill in a profile (the same onboarding questions NoCatch asks a new user), tick which models to compare, and see the 3 Mates each one ranks highest, with the reasoning behind each pick.

Part of the SCI701 dissertation project "NoCatch: A Machine-Learning-Based Matchmaking System."

## Models

| Model | Type | Scores by | "Why" explanation |
|---|---|---|---|
| Logistic Regression | ML | compatibility index (logit) | closed-form linear SHAP: `coefficient x (value - training mean)` |
| Decision Tree (depth 5) | ML | P(high-quality match) | exact TreeSHAP + the rule path the tree followed |
| Random Forest (300 trees) | ML | P(high-quality match) | exact TreeSHAP, averaged over trees |
| SVM (RBF kernel) | ML | SVM margin (signed distance from the boundary) | exact baseline Shapley values vs. the average training pair (all 2^10 coalitions) |
| k-Nearest Neighbours (k=151) | ML | share of the k most similar training pairs that were high-quality | the nearest past pairings themselves |
| k-means clustering (6 clusters) | ML, unsupervised | high-quality rate of the pair's cluster | what sets the cluster apart (centroid vs. average, in SDs) |
| Age only | Baseline | smallest age gap | n/a (age is the only input) |
| Current rule (reconstructed) | Baseline | linear reconstruction of `existing_rule_score` | linear contributions |

- All six ML models are trained on the project's staged/composite outcome target, **without** `existing_rule_score` (Ablation A variant), because that score can't be computed for a brand-new profile.
- SVM, k-NN and k-means use standardised features. k for k-NN was chosen by 5-fold GroupKFold CV on training users (NDCG@5); the number of k-means clusters by silhouette score on training rows. Neither choice looked at the test split. k-NN ties at equal distance are broken by training-row order so Python and the browser pick identical neighbours.
- "Current rule" is a least-squares reconstruction of the synthetic `existing_rule_score` from the same 10 features (held-out R² ≈ 0.82). Its ranking numbers in the page are for the original rule score.
- Held-out ranking metrics (user-level test split, booking outcome) are in `comparison_summary.json`.
- TreeSHAP runs in the browser (`tree_shap.js`). `node test_tree_shap.js` checks it against the Python `shap` library (`shap_reference.json`).
- SVM / k-NN / k-means run in the browser (`extra_models.js`). `node test_extra_models.js` checks them against sklearn and `shap.KernelExplainer` (`extra_reference.json`).

## Files

- `index.html` – the app (open via any static server, e.g. GitHub Pages)
- `mates_data.json` – candidate Mates
- `demo_model.json`, `dt_model.json`, `rf_model.json`, `rule_model.json`, `svm_model.json`, `knn_model.json`, `kmeans_model.json` – exported models (from `nocatch_ml/phase7_prototype/export_demo_model.py`, `export_comparison_models.py` and `export_extra_models.py`, run in that order)

**Data:** `NoCatch_Synthetic_Dataset_v3.xlsx` — synthetic data only, no real users.

This is an illustrative research prototype, not a live production system.
