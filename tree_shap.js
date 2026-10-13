// Tree prediction + exact path-dependent TreeSHAP (Lundberg et al. 2018,
// Algorithm 2) for the flat-array trees exported by
// nocatch_ml/phase7_prototype/export_comparison_models.py.
// Same algorithm shap.TreeExplainer uses; checked against it in test_tree_shap.js.
(function (root) {
  "use strict";

  // Follow the decision path for x; returns { value, path: [nodeIds] }.
  function predictTree(tree, x) {
    let node = 0;
    const path = [0];
    while (tree.left[node] !== -1) {
      node = x[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
      path.push(node);
    }
    return { value: tree.value[node], path };
  }

  function extendPath(m, pz, po, pi) {
    const l = m.length;
    m.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 });
    for (let i = l - 1; i >= 0; i--) {
      m[i + 1].w += po * m[i].w * (i + 1) / (l + 1);
      m[i].w = pz * m[i].w * (l - i) / (l + 1);
    }
  }

  function unwindPath(m, idx) {
    const l = m.length - 1;
    const o = m[idx].o, z = m[idx].z;
    let next = m[l].w;
    for (let j = l - 1; j >= 0; j--) {
      if (o !== 0) {
        const tmp = m[j].w;
        m[j].w = next * (l + 1) / ((j + 1) * o);
        next = tmp - m[j].w * z * (l - j) / (l + 1);
      } else {
        m[j].w = m[j].w * (l + 1) / (z * (l - j));
      }
    }
    for (let j = idx; j < l; j++) {
      m[j].d = m[j + 1].d; m[j].z = m[j + 1].z; m[j].o = m[j + 1].o;
    }
    m.pop();
  }

  function unwoundSum(m, idx) {
    const l = m.length - 1;
    const o = m[idx].o, z = m[idx].z;
    let total = 0;
    if (o !== 0) {
      let next = m[l].w;
      for (let j = l - 1; j >= 0; j--) {
        const tmp = next * (l + 1) / ((j + 1) * o);
        total += tmp;
        next = m[j].w - tmp * z * (l - j) / (l + 1);
      }
    } else {
      for (let j = l - 1; j >= 0; j--) total += m[j].w * (l + 1) / (z * (l - j));
    }
    return total;
  }

  // Adds this tree's SHAP values for x into phi (array indexed by feature).
  function treeShap(tree, x, phi) {
    function recurse(node, path, pz, po, pi) {
      const m = path.map(e => ({ d: e.d, z: e.z, o: e.o, w: e.w }));
      extendPath(m, pz, po, pi);
      const left = tree.left[node];
      if (left === -1) {
        for (let i = 1; i < m.length; i++) {
          phi[m[i].d] += unwoundSum(m, i) * (m[i].o - m[i].z) * tree.value[node];
        }
        return;
      }
      const f = tree.feature[node];
      const right = tree.right[node];
      const goLeft = x[f] <= tree.threshold[node];
      const hot = goLeft ? left : right;
      const cold = goLeft ? right : left;
      let iz = 1, io = 1;
      for (let k = 1; k < m.length; k++) {
        if (m[k].d === f) { iz = m[k].z; io = m[k].o; unwindPath(m, k); break; }
      }
      const cover = tree.cover[node];
      recurse(hot, m, iz * tree.cover[hot] / cover, io, f);
      recurse(cold, m, iz * tree.cover[cold] / cover, 0, f);
    }
    recurse(0, [], 1, 1, -1);
  }

  // Ensemble average: { prediction, baseValue, phi[] } with baseValue + sum(phi) = prediction.
  function explainEnsemble(trees, x, nFeatures) {
    const phi = new Array(nFeatures).fill(0);
    let prediction = 0, baseValue = 0;
    for (const t of trees) {
      prediction += predictTree(t, x).value;
      baseValue += t.value[0];
      treeShap(t, x, phi);
    }
    const n = trees.length;
    return { prediction: prediction / n, baseValue: baseValue / n, phi: phi.map(v => v / n) };
  }

  function predictEnsemble(trees, x) {
    let s = 0;
    for (const t of trees) s += predictTree(t, x).value;
    return s / trees.length;
  }

  const api = { predictTree, treeShap, explainEnsemble, predictEnsemble };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TreeShap = api;
})(typeof self !== "undefined" ? self : this);
