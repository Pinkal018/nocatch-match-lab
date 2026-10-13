// Checks tree_shap.js against SHAP values from the real `shap` library.
// Run: node test_tree_shap.js
const fs = require("fs");
const TreeShap = require("./tree_shap.js");

const ref = JSON.parse(fs.readFileSync("shap_reference.json", "utf8"));
let worst = 0;
for (const [key, file] of [["decision_tree", "dt_model.json"], ["random_forest", "rf_model.json"]]) {
  const model = JSON.parse(fs.readFileSync(file, "utf8"));
  ref.rows.forEach((x, r) => {
    const out = TreeShap.explainEnsemble(model.trees, x, x.length);
    const diffs = [
      Math.abs(out.prediction - ref[key].prediction[r]),
      Math.abs(out.baseValue - ref[key].expected_value),
      ...out.phi.map((v, i) => Math.abs(v - ref[key].shap_values[r][i])),
    ];
    worst = Math.max(worst, ...diffs);
  });
  console.log(`${key}: checked ${ref.rows.length} rows`);
}
console.log(`max abs difference vs shap library: ${worst.toExponential(2)}`);
process.exit(worst < 1e-4 ? 0 : 1);
