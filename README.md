# NoCatch Match Lab

A live, in-browser demo of NoCatch's compatibility model: fill in a profile (the same onboarding questions NoCatch asks a new user), and see which 3 Mates a trained model ranks highest, with the reasoning behind each pick.

Part of the SCI701 dissertation project "NoCatch: A Machine-Learning-Based Matchmaking System."

- **Model:** Logistic Regression, trained on the project's staged/composite outcome target, without `existing_rule_score` (an Ablation A variant).
- **Data:** `NoCatch_Synthetic_Dataset_v3.xlsx` — synthetic data only, no real users.
- **Explanations:** the linear-model closed-form SHAP decomposition (`coefficient x (value - training mean)`) for each candidate pairing — exact, not approximated.

This is an illustrative research prototype, not a live production system.
