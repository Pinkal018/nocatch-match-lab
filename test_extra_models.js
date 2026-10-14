// Checks extra_models.js against values from sklearn / the `shap` library
// (extra_reference.json, written by export_extra_models.py).
// Run: node test_extra_models.js
const fs = require("fs");
const EM = require("./extra_models.js");

const ref = JSON.parse(fs.readFileSync("extra_reference.json", "utf8"));
const svm = JSON.parse(fs.readFileSync("svm_model.json", "utf8"));
const knn = JSON.parse(fs.readFileSync("knn_model.json", "utf8"));
const km = JSON.parse(fs.readFileSync("kmeans_model.json", "utf8"));
const means = svm.feature_order.map(f => svm.feature_training_means[f]);

let svmWorst = 0, knnMismatch = 0, kmMismatch = 0;
ref.rows.forEach((x, r) => {
  const out = EM.baselineShapley(v => EM.svmDecision(svm, v), x, means);
  svmWorst = Math.max(svmWorst,
    Math.abs(out.prediction - ref.svm.decision[r]),
    Math.abs(out.baseValue - ref.svm.expected_value),
    ...out.phi.map((v, i) => Math.abs(v - ref.svm.shap_values[r][i])));

  const nb = EM.knnNeighbours(knn, x);
  if (Math.abs(nb.score - ref.knn.score[r]) > 1e-12 || nb.neighbours.some((v, i) => v !== ref.knn.neighbours[r][i])) knnMismatch++;

  if (EM.kmeansAssign(km, x).cluster !== ref.kmeans.cluster[r]) kmMismatch++;
});
console.log(`svm: max abs difference vs sklearn + shap.KernelExplainer: ${svmWorst.toExponential(2)}`);
console.log(`knn: ${ref.rows.length - knnMismatch}/${ref.rows.length} rows with identical neighbours and score`);
console.log(`kmeans: ${ref.rows.length - kmMismatch}/${ref.rows.length} rows assigned to the same cluster`);
process.exit(svmWorst < 1e-3 && knnMismatch === 0 && kmMismatch === 0 ? 0 : 1);
