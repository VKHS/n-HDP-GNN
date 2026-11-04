# Method specification

This document fixes every quantity the implementation computes, maps the
article's symbols to code, and lists the implementation choices the article
leaves open.

## Pipeline (article Sect. 10)

| Stage | What happens | Code |
|---|---|---|
| A | One-time edge cut into `num_parts` parts (METIS, community packing, BFS or random); nested Louvain inside each part, so no community crosses a part. | `partition.partition_nodes`, `communities.nested_communities` |
| B | Nested-HDP inference per part; only global statistics are summed across parts and ranks. Output: top-k responsibilities π. | `nhdp.fit_nhdp`, `pipeline.run_stages` |
| C | Training on each part's induced subgraph with the π-gated operator and community/global attention; DistributedDataParallel across ranks. | `pipeline.build_batches`, `model.NHDPGNN`, `train.run`, `distributed.worker` |

## Symbols

| Article | Meaning | Code |
|---|---|---|
| γ | concentration of the top-level DP | `NHDPConfig.gamma` |
| α₁ | concentration of the intermediate (super-community) DPs | `NHDPConfig.alpha1` |
| α₀ | concentration of the community DPs | `NHDPConfig.alpha0` |
| G₀, β | global measure and its stick weights | `q(β'_k) = Beta(u_k, v_k)` in `nhdp._Global` |
| G₀ₖ | super-community measure | sticks `Shard.ps_a/ps_b`, atom map `Shard.phi` |
| Gⱼ, Gⱼₖ | community measures | sticks `Shard.pj_a/pj_b`, atom map `Shard.psi` |
| zⱼᵢ | assignment of node i | `Shard.zeta` |
| θ | atom parameters | Normal-Gamma posterior `m, kappa, a, b` |
| π_ic | responsibility of node i for cluster c | `R` (top-k, rows sum to 1) |
| P = Σ_c Diag(π_c) Ã_c | gated propagation operator | `gating.gated_operator`, edge values `gating.gate_weights` |
| ρ = \|∂E\|/\|E\| | cross-partition edge ratio | `gating.cross_partition_ratio` |
| r_E = 1 − \|E_s\|/\|E\| | edge reduction of the super-node graph | `gating.super_graph`, `gating.edge_reduction` |
| MADGap of Eq. (28) | inter-class centroid distance minus intra-class spread | `metrics.centroid_gap` |

## Stage B: the nested HDP

Groups are nested Louvain communities: the finest Louvain level gives
communities j, the coarsest level groups them into super-communities s(j).

```
global      β'_k ~ Beta(1, γ),        θ_k = (μ_k, λ_k) ~ NormalGamma(m0, κ0, a0, b0)     k = 1..K
super s     π'_st ~ Beta(1, α₁),      c_st ~ Cat(β)                                       t = 1..T1
community j π'_ju ~ Beta(1, α₀),      d_ju ~ Cat(π_s(j))                                  u = 1..T2
node i ∈ j  z_i ~ Cat(π_j),           x_i ~ N(μ_k, diag(1/λ_k)),  k = c_{s(j), d_{j, z_i}}
```

`levels = 2` removes the super-community level (each community draws atoms
directly from β): the two-level HDP of the article's Table 1 at matched truncation.

**Variational family.** q(β'_k) = Beta(u_k, v_k), q(π'_st) = Beta, q(π'_ju) = Beta,
q(c_st) = Cat(φ_st), q(d_ju) = Cat(ψ_ju), q(z_i) = Cat(ζ_i), q(θ_k) Normal-Gamma.
With ℓ_ik = E_q[log N(x_i | θ_k)], the coordinate updates are

```
ζ_iu  ∝ exp( E[log π_ju] + Σ_t ψ_jut Σ_k φ_stk ℓ_ik )
ψ_jut ∝ exp( E[log π_st] + Σ_{i∈j} ζ_iu Σ_k φ_stk ℓ_ik )
φ_stk ∝ exp( E[log β_k]  + Σ_{j∈s} Σ_u ψ_jut Σ_{i∈j} ζ_iu ℓ_ik )
```

plus conjugate Beta updates of the three stick levels from expected counts and the
conjugate Normal-Gamma update from the responsibilities

```
r_ik = Σ_u ζ_iu Σ_t ψ_jut φ_stk .
```

Each update is the exact maximiser of the evidence lower bound in its block, so
the bound is non-decreasing; it is computed exactly (`fit_nhdp(...).elbo_trace`),
and `tests/test_nhdp.py` checks monotonicity, recovery of planted clusters, and that
sharded and simulated multi-rank inference reproduce the single-process result.

**Relation to the article's Sect. 7.** The article describes optimising the bound
with reparameterised gradients. All conditionals here are conjugate, so the same
bound is maximised by closed-form coordinate ascent: no Monte-Carlo gradient is
needed, and the Beta stick variables (which have no location-scale
reparameterisation) are handled exactly. The divergence term of the bound is
KL(q ‖ p).

**Distributed inference.** The model factorises over super-communities. Each part
updates its own ζ, ψ, φ and sticks; the global updates need only
Σ φ (stick counts, K values) and the Normal-Gamma moments Σ r, Σ r x, Σ r x²
(K × (2d + 1) values), which are all-reduced once per iteration.

**Likelihood features.** Node features are projected to 32 dimensions by
TruncatedSVD and standardised (`feature_dim`). `feature_hops > 0` smooths them with
the normalised adjacency first; the default 0 models the node features themselves,
as the article does.

## Gated propagation

With Ã_c = Â Diag(π_c) (columns of the normalised adjacency masked by π_c),

```
P = Σ_c Diag(π_c) Â Diag(π_c),     P_ij = Â_ij ⟨π_i, π_j⟩ .
```

* One-hot π keeps exactly the intra-community edges (P is block diagonal); soft π
  keeps a cross-community edge with weight ⟨π_i, π_j⟩ < 1, and removes it when the
  top-k supports of the two endpoints are disjoint.
* 0 ≤ ⟨π_i, π_j⟩ ≤ 1, so |P| ≤ Â entrywise and the spectral radius of P is at most 1.
* π is inferred once (Stage B), so the operator is the same at every layer; the
  article's layer index on π is not used.
* In the attention layer the gate multiplies the attention weights inside the
  softmax normalisation: α_ij = g_ij exp(e_ij) / Σ_k g_ik exp(e_ik). With all gates
  equal this is GAT.

### What the gate does at large depth

Let G_g be the graph of edges with g_ij > 0 (self-loops included) and C_1..C_m its
connected components.

**Proposition.** Let Ã = (A + I) ⊙ G be the gated adjacency and D_g its row sums.
(a) For P_rw = D_g⁻¹ Ã, P_rwᴸ H converges as L → ∞ to a matrix whose rows are
constant on every C_r. For the GCN-normalised gated operator P = Â ⊙ G of the gated
GCN layer, the rows of PᴸH on each C_r become collinear (P is entrywise below Â, so
it is also contracting: ρ(P) ≤ 1). Either way the limiting representation has rank
at most m. (b) Without skip connections, a classifier applied to the P_rw limit
predicts one class per component, so its accuracy is at most the purity
Σ_r max_y |C_r ∩ {y_i = y}| / N.

*Proof.* Both operators are block diagonal over the components. Each block of P_rw
is the transition matrix of an irreducible chain with self-loops, hence aperiodic,
so its powers converge to 1 μ_rᵀ with μ_r its stationary distribution: every row of
the block of P_rwᴸ H tends to μ_rᵀ H_{C_r}. Each block of P is symmetric, irreducible
and non-negative, so by Perron-Frobenius its powers are dominated by λ_rᴸ v_r v_rᵀ:
the rows on C_r align with v_rᵀ H_{C_r}. A classifier that is a function of the row
is then constant on each component, and the best constant prediction on C_r is its
majority class. ∎

The attention layer is row-stochastic on the gated support with layer-dependent
weights. Products of such matrices behave as in (a) when the weights are bounded
below uniformly across layers, which holds whenever the attention logits are bounded.

Consequences:

1. The gate does not remove over-smoothing. It changes the limit from one consensus
   per connected component of the graph to one per gate-connected component, and
   lowers the convergence rate through the spectral gap of each block.
2. With hard Louvain gating (`assign = "louvain"`) the bound is the community purity.
   `scripts/regime_report.py` prints it per dataset; when it lies below the accuracy
   of shallow models, accuracy retained at depth 20-30 must come from the residual,
   initial-feature, community or global paths. The ablations in
   `configs/table1_ablation.toml` switch these off one at a time.
3. Information cannot cross between gate components through the node-level operator
   at any depth. Long-range exchange between communities runs only through the
   community- and global-level attention.

## Community and global levels

* Super-node states: H_c = Σ_i π_ic h_i / Σ_i π_ic (pooling by πᵀ).
* Community level (Eq. 16): multi-head attention among super-nodes on the
  super-node graph A_S = πᵀ A π (off-diagonal non-zeros, self-loops added).
* Global level (Eq. 17): attention read-out over super-nodes (weights ∝ exp(score) ×
  super-node mass), projected and added to every node.
* Unpooling: node i receives Σ_c π_ic H'_c.
* Layer update: u = Dropout(ELU(Norm(m_node + m_community + m_global))), then
  `res`: h + u, `initial`: (1 − a) u + a h₀, or `none`: u.

## Systems metrics under DistributedDataParallel

DDP exchanges gradients (bucketed all-reduce). Bytes per step therefore depend on the
number of parameters and how often gradients are synchronised, not on the graph.
In partition-parallel training the cut edges are not propagated, so ρ measures how
much of the graph the cut removes, which is an accuracy cost rather than a
communication cost. ρ_g, the share of *gated* propagation mass on cut edges,
measures how much of what the model would propagate is lost. The distributed
trainer reports time per epoch, edge-updates per second, all-reduce wall time
(launch to completion, overlapping backward), Comms%, payload and ring-wire bytes,
ρ and r_E. ρ_g needs the responsibilities of both endpoints of every cut edge, so it
is computed offline by `scripts/regime_report.py` on the same partition.

## Choices the article leaves open

| Item | Choice in this code |
|---|---|
| n-HDP likelihood | diagonal Gaussian, Normal-Gamma prior (m0 = 0, κ0 = 0.1, a0 = b0 = 1), on 32-d SVD-reduced standardised features |
| nesting | finest Louvain level = communities; coarsest = super-communities |
| truncations, concentrations | K = 64, T1 = 16, T2 = 8; γ = α₁ = α₀ = 1 |
| sparsification | top-k with k = 4, renormalised |
| self-loop gate | g_ii = ‖π_i‖² (from the operator; only relative gates matter in attention) |
| auxiliary Louvain loss | weight 0.01 (value of the first-release prototype); off in DDP, where community ids are local to a part |
| semi-supervised regime | public Planetoid split; 20 labels per class on Amazon Photo |
| label-scarce regime | 5% of nodes, stratified, drawn from the full-regime training pool; validation and test as in the full regime |
| MADGap | both the article's Eq. (28) (`centroid_gap`) and the topological MADGap of Chen et al., AAAI 2020 (`madgap`: neighbours ≤ 3 hops, remote ≥ 8 hops, cosine) |
| clustering protocol | k-means, k = #classes, 10 initialisations, on the L2-normalised logits (the representation every compared model has) |
| embedding for MADGap and Eq. (28) | the logits of every model; the per-layer centroid gap profiles the hidden layers where a model has them |

## Ablations (configs/table1_ablation.toml)

| Variant | Isolates |
|---|---|
| hdp-gnn vs nhdp-gnn | the super-community level of the prior (2- vs 3-level HDP at matched truncation) |
| +res, +gat1, +gat2 | the article's progression of residual and attention heads |
| full | gate + community + global attention |
| full-gate-off | the gate itself (same S for the community/global levels, all gates 1) |
| full-louvain-hard | soft Bayesian responsibilities vs hard Louvain assignment |
| full-random-partition | community structure vs a random partition with the same community sizes |
| gat+res-no-hierarchy | everything n-HDP-specific removed |
