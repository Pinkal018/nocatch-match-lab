// SVM (RBF), k-Nearest Neighbours and k-means scoring for the models exported by
// nocatch_ml/phase7_prototype/export_extra_models.py, plus exact baseline Shapley
// values for any scoring function. Checked against Python in test_extra_models.js.
(function (root) {
  "use strict";

  function standardize(x, scaler) {
    return x.map((v, j) => (v - scaler.mean[j]) / scaler.scale[j]);
  }

  // decision(z) = sum_i dual_coef_i * exp(-gamma * |z - sv_i|^2) + intercept
  function svmDecision(m, x) {
    const z = standardize(x, m.scaler);
    let s = m.intercept;
    for (let i = 0; i < m.support_vectors.length; i++) {
      const sv = m.support_vectors[i];
      let d2 = 0;
      for (let j = 0; j < z.length; j++) { const d = z[j] - sv[j]; d2 += d * d; }
      s += m.dual_coef[i] * Math.exp(-m.gamma * d2);
    }
    return s;
  }

  // Exact Shapley values of f at x against one reference point: features in the
  // coalition take x's value, the rest take the reference value. All 2^M
  // coalitions are enumerated (M = 10 here -> 1024 evaluations).
  // Returns { baseValue: f(reference), prediction: f(x), phi[] }, sum(phi) = prediction - baseValue.
  function baselineShapley(f, x, reference) {
    const M = x.length;
    const n = 1 << M;
    const vals = new Float64Array(n);
    for (let mask = 0; mask < n; mask++) {
      const v = reference.slice();
      for (let j = 0; j < M; j++) if (mask & (1 << j)) v[j] = x[j];
      vals[mask] = f(v);
    }
    const fact = [1];
    for (let i = 1; i <= M; i++) fact[i] = fact[i - 1] * i;
    const phi = new Array(M).fill(0);
    for (let mask = 0; mask < n; mask++) {
      let size = 0;
      for (let j = 0; j < M; j++) if (mask & (1 << j)) size++;
      const w = fact[size] * fact[M - size - 1] / fact[M];
      for (let j = 0; j < M; j++) {
        if (mask & (1 << j)) continue;
        phi[j] += w * (vals[mask | (1 << j)] - vals[mask]);
      }
    }
    return { baseValue: vals[0], prediction: vals[n - 1], phi };
  }

  // k nearest training rows. Same arithmetic as knn_scores() in the Python
  // export: standardised squared distance summed feature by feature, rounded to
  // 1e-9, ties broken by training-row order.
  function knnNeighbours(m, x) {
    const X = m.train_X, scale = m.scaler.scale;
    const d2 = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) {
      const r = X[i];
      let s = 0;
      for (let j = 0; j < x.length; j++) { const d = (x[j] - r[j]) / scale[j]; s += d * d; }
      d2[i] = Math.round(s * 1e9) / 1e9;
    }
    // Equivalent to a stable sort by distance, without sorting every row:
    // find the k-th smallest distance, keep everything closer, then fill up
    // with rows at exactly that distance in training-row order.
    const kth = Float64Array.from(d2).sort()[m.k - 1];
    const closer = [], atKth = [];
    for (let i = 0; i < d2.length; i++) {
      if (d2[i] < kth) closer.push(i);
      else if (d2[i] === kth) atKth.push(i);
    }
    closer.sort((a, b) => (d2[a] - d2[b]) || (a - b));
    const top = closer.concat(atKth.slice(0, m.k - closer.length));
    let pos = 0;
    for (const i of top) pos += m.train_y[i];
    return { score: pos / m.k, neighbours: top, distances: top.map(i => Math.sqrt(d2[i])) };
  }

  function kmeansAssign(m, x) {
    const z = standardize(x, m.scaler);
    let best = 0, bestD = Infinity;
    m.centroids_scaled.forEach((c, k) => {
      let d2 = 0;
      for (let j = 0; j < z.length; j++) { const d = z[j] - c[j]; d2 += d * d; }
      if (d2 < bestD) { bestD = d2; best = k; }
    });
    return { cluster: best, distance: Math.sqrt(bestD), score: m.cluster_high_quality_rate[best] };
  }

  const api = { standardize, svmDecision, baselineShapley, knnNeighbours, kmeansAssign };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ExtraModels = api;
})(typeof self !== "undefined" ? self : this);
