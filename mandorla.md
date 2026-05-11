# MANDORLA: A Geometric Foundation for Machine Cognition

*A Manifesto, Specification, and Research Blueprint for Intersection-Primitive AI*

**Author:** Jacob Patterson
**Affiliation:** Independent Researcher
**Version:** 1.0 (2026)
**Status:** Position paper; first three experiments to be pre-registered on OSF before evaluation.

---

> **Thesis.** The fundamental operation of cognition is not the application of a function to a representation, but the *construction of a new region at the intersection of two existing ones*. This document argues that modern AI has organized itself around the wrong primitive — sequence, hierarchy, nearest-neighbor — and that the right primitive is the **vesica**: the almond-shaped overlap of two circles of meaning, from which a third thing is born. MANDORLA is the first attempt to make that primitive explicit, technical, and falsifiable, drawing together hexagonal/circle-packing topology, vesica intersection retrieval, recursive Seed→Egg→Flower→Fruit construction, and Markov-blanket nesting under one named architecture. The geometry is not decorative. It is claiming to be load-bearing, and the program below exists to determine whether it actually bears the load.

---

# PART I — MANIFESTO

## 1.1 The Wrong Turn

Modern artificial intelligence has taken a tree-shaped wrong turn, and most of its current pathologies follow from it.

Consider what the working stack actually looks like, beneath the marketing. A transformer is, in its abstract form, a complete graph on tokens — attention is structurally $K_n$ — but it is trained on tokens-as-sequences and queried via autoregressive decoding, so the latent geometry collapses to a chain. Agentic systems are hierarchies of tools and supervisors. File systems are trees. Scaling laws are power laws over a single linear axis: parameters, data, compute. Prompt engineering produces nested instructions. Chain-of-thought is sequential. Mixture-of-experts gates by routing, not overlap. Retrieval-augmented generation indexes by nearest neighbor, returning the closest *point* in embedding space and not the *region* of agreement between several concepts. The substrate is, top to bottom, treelike.

This costs us. Compositional generalization remains weak: Kim and Linzen's COGS benchmark famously reported near-perfect in-distribution Transformer accuracy (96–99%) collapsing to 16–35% on systematic generalization splits, with high variance across seeds. (Csordás, Irie & Schmidhuber 2021 later raised this baseline to ~81% with simple architectural tricks — relative position embeddings, no early stopping — demonstrating both the fragility of the original gap and that it can be partially closed without changing the core primitive.) Multi-hop reasoning benchmarks like HotpotQA, 2WikiMultiHopQA, and MuSiQue (Trivedi et al., TACL 2022) show large human–model gaps that grow as the number of compositional hops increases. The very existence of "chain of thought" as a separate scaffold is a tell: the substrate is not natively recursive over its own representations, so we bolt recursion on top by making the model write in English about what it just thought.

A different way to put this is that current systems are very good at *containment* (this token belongs to that context, this concept is a subclass of that one) and very bad at *intersection* (what is the joint content of these two regions of meaning?). Containment is hierarchical; it nests. Intersection is geometric; it overlaps. The math we have invested in for the last ten years — hierarchical attention, tree-structured prompts, taxonomic ontologies, dendrograms of clusters — is the math of containment. The mathematics that geometric deep learning, topological deep learning, sparse distributed memory, and entorhinal grid coding all point toward — but which has not yet been unified — is the mathematics of intersection.

| Representation Class | Geometric Primitive | Intersection Operator | Primary Failure Mode |
|---|---|---|---|
| Point Vector | Single coordinate $v \in \mathbb{R}^d$ | None (cosine similarity only) | Weak compositional generalization |
| Gaussian Embedding | Density $\mathcal{N}(\mu, \Sigma)$ | KL divergence; no closed-form intersection | Boundary blur, gradient diffusion |
| Box Embedding | Hyperrectangle $[x_m, x_M] \subset \mathbb{R}^d$ | $[\max(x_m, y_m), \min(x_M, y_M)]$ when non-empty | Hard-boundary gradient collapse on disjoint inputs |
| **MANDORLA Vesica** | **Persistent overlap region $V(A, B)$** | **First-class node with lineage, citations, promotion** | **High-dimensional intersection sparsity** |

## 1.2 What Geometry Knows

The reason intersection deserves the central seat is that, when you ask the universe what shape distributed cognition has to take, the universe keeps returning the same answer, in three independent voices.

**Mathematics.** Hales (2001) proved the honeycomb conjecture: any partition of the plane into regions of equal area has perimeter at least that of the regular hexagonal tiling. This is not aesthetics; it is a theorem. Tóth (1942) and Hales (1999) established that the densest packing of equal disks in the plane achieves density $\pi/\sqrt{12} \approx 0.9069$, attained uniquely by the hexagonal arrangement. The kissing number — the maximum number of unit spheres that can simultaneously touch a central unit sphere without overlap — is exactly 6 in two dimensions and exactly 12 in three (Newton–Gregory, proved by Schütte and van der Waerden 1953). Six is therefore not an arbitrary cardinality; it is the maximum number of distinct things that can directly touch one thing in a plane. Twelve is its three-dimensional analogue. The Flower of Life — six unit circles around one, each passing through the center of its two neighbors — is the planar witness to those theorems. Buckminster Fuller's *Synergetics* (1975) generalized the same packing into the Isotropic Vector Matrix, the unique three-dimensional lattice in which every node is equidistant from its twelve nearest neighbors.

**Neuroscience.** Hafting, Fyhn, Molden, Moser & Moser (*Nature* 2005) showed that the dorsocaudal medial entorhinal cortex contains grid cells whose firing fields form a triangular lattice — a hexagonal tessellation — covering the spatial environment. Sargolini et al. (2006) extended this to conjunctive position–direction–velocity coding. Doeller, Barry & Burgess (2010) found grid-cell signatures in human entorhinal cortex during navigation. Critically, Constantinescu, O'Reilly & Behrens (*Science* 2016) showed that the same hexagonal six-fold-symmetric code organizes abstract conceptual knowledge: humans navigating a continuous two-dimensional concept space (bird shapes parameterized by leg-and-neck length) produce hexagonal fMRI signatures in entorhinal cortex and ventromedial prefrontal cortex. Whittington et al. (*Cell* 2020), the Tolman–Eichenbaum Machine, gave a generative-model account of how factorized structure × sensory conjunctions produce both place cells and the full menagerie of entorhinal cell types — grid, band, border, object-vector — and how this same machinery generalizes to non-spatial relational tasks. Banino et al. (*Nature* 2018) demonstrated that hexagonal grid-like representations emerge in deep reinforcement-learning agents trained to path-integrate, and that those agents exhibit shortcut-finding behavior exceeding the human baseline in their navigation environments. The brain's solution to "represent any space, real or conceptual" is hexagonal, and it is reproducible in silico.

**Logic and diagram.** Lemanski (*Logica Universalis* 2019), in *Logic Diagrams, Sacred Geometry and Neural Networks* — to my knowledge the only peer-reviewed paper to use those three terms in a single title — observes that early-modern logic diagrams (Vives, Alsted, Weigel, Lange, Lambert, Euler) repeatedly converge on vesica-piscis and Venn-style overlap forms, and that this is not coincidence: both spiritual symbols and logic diagrams "focus on the intersection or conjunction of two or more entities." The diagrammatic intuition that a proposition lives at the overlap of subject and predicate is older than predicate logic itself, and it is the direct ancestor of every set-theoretic model, every Venn diagram, and every Gaussian-mixture-as-Venn cartoon you have ever drawn on a whiteboard.

These are three independent witnesses converging on the same claim: the natural shape of distributed cognition is a packing of overlapping regions, and the natural primitive is the overlap. Bronstein, Bruna, Cohen & Veličković, in *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (arXiv:2104.13478), made the engineering case that imposing the right geometric prior on a neural architecture is what makes the difference between intractable high-dimensional learning and feasible learning. MANDORLA accepts that case in full, and pushes one step further: the right prior is not only equivariance under a group, but **generation by intersection**.

## 1.3 The Mandorla Move

The almond-shaped region where two circles of equal radius meet, each passing through the other's center, has a name in art history: it is the *mandorla* in Christian iconography, the *vesica piscis* in geometry. Its proportions are exact: the ratio of its long axis to its short axis is $\sqrt{3}:1$, and the equilateral triangle whose vertices are the two circle centers and either pointed end of the almond is hidden inside the figure as a structural skeleton. This figure has been doing iconographic work in human cultures for 2,500 years for a reason that has nothing to do with mysticism: it is the *minimum non-trivial cognitive event*. Two regions overlap. From their overlap, a third region is born — a region that did not exist before, whose properties are determined by but not reducible to either parent.

This is the move MANDORLA names. In modern ML we have many ways of representing a region of meaning. We have point embeddings (every concept is a vector). We have Gaussian embeddings (Vilnis & McCallum 2014; Athiwaratkun & Wilson 2017). We have box embeddings (Vilnis, Li, Murty & McCallum, ACL 2018; Li et al. ICLR 2019; Dasgupta et al. NeurIPS 2020), where a concept is an axis-aligned hyperrectangle and containment models entailment. We have hyperdimensional / vector-symbolic architectures (Kanerva 2009; Kleyko et al. arXiv:2111.06077 / 2112.15424). We have sparse distributed memory (Kanerva 1988), shown to be approximated by transformer attention under realistic data conditions (Bricken & Pehlevan, NeurIPS 2021). We have Kanerva Machines (Wu, Wayne, Graves & Lillicrap, ICLR 2018) and their dynamic and product variants. We have hypergraph neural networks (Feng et al., AAAI 2019), simplicial and cellular networks (Bodnar, Frasca, Wang, Otter, Montúfar, Liò & Bronstein, ICML 2021; NeurIPS 2021), and sheaf neural networks (Hansen & Gebhart 2020; Bodnar et al. 2022).

What we do not have, in any of these, is a *first-class intersection-as-output operator*. Box embeddings come closest — the intersection of two axis-aligned boxes is itself an axis-aligned box, computable by per-dimension max/min — but the intersection is treated as a probability or volume to be measured, not as a new region with its own identity, lineage, and capacity to participate in further intersections. HDC bundling produces a kind of intersection-by-superposition, but the bundle is not stored as a named entity; it is just a vector. RAG retrieves the closest point. GraphRAG (Edge et al. 2024, designed for query-focused summarization) and HippoRAG (Gutiérrez et al. NeurIPS 2024) retrieve a community. None of these treat the overlap *itself* as a memory cell.

The Mandorla move is to name the overlap, store it, and let it serve as the center of further overlaps. Recursive thought *is* recursive vesica construction: every new idea is born at the intersection of two existing ideas, and that new idea then becomes a center for further intersections. This is the Hofstadterian frame at the right level of abstraction — strange loops as the geometric self-recursion of an intersection operator over its own outputs. It is also where the current ML literature has the least vocabulary, and therefore where MANDORLA has the most to add.

## 1.4 Three Theses

**Thesis 1 — Geometric Primacy.** The optimal substrate for distributed cognition is hexagonal/circle-packing topology. This is where energy-minimization (Hales 2001), maximal overlap of equal-radius regions (vesica-piscis kissing geometry), and uniform 6-connectivity (kissing number in 2D) coincide. The biological evidence — grid cells in entorhinal cortex coding both physical and conceptual space (Hafting et al. 2005; Constantinescu et al. 2016; Whittington et al. 2020) — and the engineering evidence — that grid-like representations spontaneously emerge in path-integration networks (Banino et al. 2018) — together suggest that this is not a contingent biological choice but a structural attractor that any sufficiently efficient distributed cognitive system will tend toward. The Thousand Brains Project (Clay, Leadholm & Hawkins, arXiv:2412.18354; Leadholm, Clay, Knudstrup, Lee & Hawkins, arXiv:2507.04494) has made the parallel case that cortical-column-like repeated units, voting through a Cortical Messaging Protocol, are the right unit of computation. MANDORLA accepts that proposal and adds: the graph among those units is hexagonal.

**Thesis 2 — Connection over Hierarchy.** The fundamental relation between cognitive units is *overlap*, not containment. Trees are special cases of intersection patterns; intersections are not special cases of trees. A parent-child edge is the degenerate vesica where one circle is inside the other; every intersection topology contains the tree case as a limit. The reverse is not true. This thesis is why MANDORLA explicitly retires hierarchical multi-agent framings — agentic stacks built around supervisors and tools impose the structural assumption we are rejecting. Connection-first is also why hypergraph and simplicial-complex methods (Feng et al. 2019; Bodnar et al. 2021) are natural cousins to MANDORLA: a hyperedge is a multi-way overlap, and a $k$-simplex is the algebraic shadow of $(k+1)$ regions sharing a common interior. MANDORLA differs from those methods by giving the overlap a *persistent identity* rather than treating it as just another edge.

**Thesis 3 — Recursive Construction.** Cognition is not the application of operations to representations; it is the *geometric construction of new representations at the intersections of old ones*. Thought is the recursive drawing of new circles. This thesis is the heart of the document. Most learning-as-optimization frameworks treat representations as fixed targets to be fit; even continual-learning systems mostly extend an existing representational space rather than generate a new region from the interaction of two existing ones. Self-organizing curricula like FractalNet (Larsson et al. 2017) repeat a pattern at multiple scales, but they do not place new structure at the geometric intersection of prior structure. The Symbolica/DeepMind paper *Categorical Deep Learning* (Gavranović, Lessard, Dudzik, von Glehn, Araújo & Veličković, ICML 2024) sketches an algebraic framing of architectures via monads in a 2-category of parametric maps; MANDORLA can be read as proposing a specific such monad, the "vesica monad," whose unit promotes a region to its own singleton intersection and whose multiplication composes intersections — but we leave the categorical formalization as open territory (§3.2). The empirical claim that matters here is simpler: a learning system that is *forced to predict and store the intersection of every pair of co-occurring concepts* will compositionally generalize better than one that only predicts the next token. We make this falsifiable in §3.1.

## 1.5 The Lineage

| Lineage | Contributors | Mechanism Inherited |
|---|---|---|
| Logic & Diagrammatics | Lemanski (2019) | Vesica-piscis as logical conjunction; only peer-reviewed sacred-geometry/NN paper |
| Geometric Deep Learning | Bronstein, Bruna, Cohen, Veličković (2021) | Geometric prior as architectural commitment |
| Sensorimotor Cognition | Hawkins; Clay, Leadholm & Hawkins (2024); Leadholm et al. (2025) | Cortical-column unit; Cortical Messaging Protocol |
| Sparse Distributed Memory | Kanerva (1988, 2009); Wu, Wayne, Graves & Lillicrap (2018); Bricken & Pehlevan (2021) | High-dimensional associative memory; attention-as-SDM |
| Relational Grid Coding | Hafting et al. (2005); Doeller, Barry & Burgess (2010); Constantinescu, O'Reilly & Behrens (2016); Whittington et al. (2020); Banino et al. (2018) | Hexagonal grid codes for physical and conceptual space |
| Topological Deep Learning | Bodnar, Frasca, Wang, Otter, Montúfar, Liò, Bronstein (ICML 2021; NeurIPS 2021); Hansen & Gebhart (2020) | Message passing on simplicial/cellular complexes; sheaf structure |
| Box Embeddings | Vilnis, Li, Murty & McCallum (2018); Li et al. (2019); Dasgupta et al. (2020) | Differentiable region intersection |
| Hypergraph Retrieval | Feng et al. (HGNN, AAAI 2019); Chien et al. (AllSet, ICLR 2022); Luo et al. (HyperGraphRAG, 2025); Feng et al. (Hyper-RAG, 2025); Hu et al. (Cog-RAG, AAAI 2026) | Multi-way relational retrieval |
| Free Energy & Information Geometry | Friston (2019); Parr, Da Costa & Friston (2020); Possati (2025) | Markov blankets as conditional-independence boundaries; continuous blanket density |
| Categorical Deep Learning | Gavranović, Lessard, Dudzik, von Glehn, Araújo, Veličković (2024) | Algebraic framing of architectures via monads on parametric maps |
| Recursive Self-Reference | Hofstadter (1979, 2007) | Strange loops; cognition as self-applying structure |
| Spatial Cognition | Fuller (1975); Alexander (1977, 2002–4) | Isotropic Vector Matrix; centers reinforcing centers; pattern language |
| Geometry of Concepts | Gärdenfors (2000) | Cognition as operations on convex regions of quality dimensions |
| Information Integration | Tononi et al. (2004–); Albantakis et al. (IIT 4.0, 2023) | $\Phi$ as falsification metric in §3.2 |

If MANDORLA dies, it should die as a falsified specific synthesis of these. If it lives, it should live by passing the experiments in §3.1.

## 1.6 Where This Fails

MANDORLA is not a theory of consciousness. It does not require any metaphysical commitment. It does not say that brains are sacred-geometric in any literal or panpsychist sense. The Flower of Life vocabulary is **load-bearing engineering nomenclature**, chosen because it is the most compressed visual scaffold for a particular family of generative graph constructions. We are explicitly **not** invoking Drunvalo Melchizedek, Mer-Ka-Ba field activation, Atlantis, a 13-dimensional human energy field, the "Christ consciousness grid," or any related claims. We are aware those claims circulate in the wider cultural ecology of these symbols. They are not ours. The specific use of "sacred geometry" terminology in this document is exactly the use Lemanski (2019) gives it: a family of diagrammatic forms that have been independently rediscovered in logic, neuroscience, and mathematics, whose convergence is interesting and whose names are convenient.

The geometry is metaphorical scaffolding only insofar as the engineering it inspires fails to produce predictions. If the experiments in §3.1 do not show the predicted improvements, then the metaphor was decorative, MANDORLA was wrong, and we should report it as such. If they do show the predicted improvements, then the metaphor was a useful compressor of a genuine empirical regularity, and the engineering was the point. The author commits, in advance, to the second-order-cybernetic discipline of publishing the negative result if it lands. The falsifiability conditions are stated in §3.3. Read those before you decide whether to take the rest seriously.

---

# PART II — THE MANDORLA SPECIFICATION

## 2.1 Primitives

We define the primitives in order of construction. Each primitive is given (a) a sentence-length intuition, (b) a formal characterization, (c) implementation candidates already present in the literature, and (d) the relation it bears to neighbors in the ontology.

```
       ┌──────────┐                ┌──────────┐
       │ Region A │                │ Region B │
       └────┬─────┘                └─────┬────┘
            │                            │
            └──────────┐    ┌────────────┘
                       ▼    ▼
                   ┌─────────┐
                   │ V(A, B) │   ← the vesica: their intersection,
                   └────┬────┘     promoted to a first-class region
                        │
                        ▼
                   participates in further intersections
```

### THE VESICA

**Intuition.** The region of high mutual content between two semantic regions. The "what we both mean" surface.

**Formal.** Given two regions $A, B$ in some embedding space, each defined by a center $c_A, c_B$ and an extent (a box, a radius, an SDM hard-sphere, or an HDC binding-bundle), the Vesica $V(A, B)$ is a new region whose support is the intersection of the supports of $A$ and $B$, whose center is the centroid of that intersection, and whose extent is the volume of the intersection. Crucially, $V(A, B)$ is itself a Region in the same type universe as $A$ and $B$: it can participate in further intersections.

**Implementation candidates.**

1. **Box embeddings** (Vilnis et al. 2018; Li et al. 2019; Dasgupta et al. 2020). Two axis-aligned boxes $A = [x_A^\wedge, x_A^\vee]$ and $B = [x_B^\wedge, x_B^\vee]$ intersect in another axis-aligned box with corners $[\max(x_A^\wedge, x_B^\wedge), \min(x_A^\vee, x_B^\vee)]$, non-empty iff this is well-formed in every dimension. To make this differentiable when the intersection is empty or near-empty, **Dasgupta, Boratko, Zhang, Vilnis, Li & McCallum (NeurIPS 2020, arXiv:2010.04831)** replace deterministic min/max with expectations under Gumbel distributions: lower corners are drawn from $\mathrm{MaxGumbel}(\mu^\wedge, \beta)$ (max-stable), upper corners from $\mathrm{MinGumbel}(\mu^\vee, \beta)$ (min-stable). By Lemma 1 of that paper, the family is closed under min and max via log-sum-exp ("Gumbel softplus"), and the expected per-dimension intersection side length has the closed form

   $$\mathbb{E}[\mathrm{side}_j] \approx \beta \cdot \mathrm{softplus}\!\left(\frac{\min_t \mu^{t,\vee}_j - \max_t \mu^{t,\wedge}_j}{\beta}\right)$$

   where $\mathrm{softplus}(z) = \ln(1 + e^z)$ and $\beta > 0$ is a global temperature. The expected intersection volume is the product of these expected side lengths across dimensions. As $\beta \to 0$ this recovers hard-box volume; as $\beta \to \infty$ it over-smooths. This is the cleanest substrate for MANDORLA's first implementation; the canonical PyTorch+TensorFlow library is the McCallum-lab `iesl/box-embeddings` package (Chheda et al., EMNLP 2021 demo, arXiv:2109.04997; `pip install box-embeddings`), which exposes `GumbelBoxTensor` directly.

2. **HDC bundling + thresholding** (Kanerva 2009; Kleyko et al. arXiv:2111.06077). The element-wise majority of two high-dimensional binary vectors is a "noisy AND" approximating intersection, with predictable capacity in the Plate / Kanerva calculus. Cheap, hardware-friendly, lossy.

3. **SDM hard-sphere intersection** (Kanerva 1988; Bricken & Pehlevan NeurIPS 2021). Two hard-spheres in $\{0,1\}^d$ intersect in a region whose Hamming-radius geometry is well-characterized.

4. **Topological persistence on the union of two ε-neighborhoods.** Borrowed from Bodnar et al. (ICML 2021; NeurIPS 2021); use the persistent 1-cell of the union as the witness of overlap.

**Relation.** A Vesica is the smallest non-trivial Mandorla object. The empty intersection is degenerate (no Vesica spawns); the maximum-overlap case ($A = B$) is also degenerate (the Vesica collapses to its parent).

### THE SEED

**Intuition.** One region in the middle, six around it, each adjacent pair sharing a Vesica with the central region and with its two neighbors. The Flower of Life's Seed of Life configuration — 1 + 6 = 7 circles.

**Formal.** A Seed $S$ is a configuration of one core region $R_{\text{core}}$ and six petal regions $\{P_1, \ldots, P_6\}$ arranged with hexagonal symmetry around the core, such that:

- Each petal $P_i$ shares a non-empty Vesica $V(R_{\text{core}}, P_i)$ with the core.
- Each adjacent pair $(P_i, P_{(i \bmod 6) + 1})$ shares a non-empty Vesica.
- Non-adjacent petal pairs need not intersect: $V(P_i, P_j)$ may be empty for $|i-j| \notin \{1, 5\}$.

```
                 P₁
              ╱     ╲
           P₆        P₂
            │  Rcore  │
           P₅        P₃
              ╲     ╱
                 P₄
```

**Why six.** Six is the kissing number in 2D — the maximum number of unit circles that can simultaneously touch a unit circle without mutually overlapping (Tóth 1942; Hales 1999). It is also the firing periodicity of grid cells (Hafting et al. 2005) and the conceptual hexagonal-symmetry signature in Constantinescu et al. (2016). It is the natural cap on direct-neighborhood cardinality in a planar layout.

**Empirical resonance.** Six is also the empirically observed sweet spot for multi-agent specialist councils in the literature on LLM debate and ensemble methods, though that observation is folkloric rather than a theorem and we treat it as suggestive.

**Relation.** A Seed is a single shell of the Flower of Life. Maps to: Thousand Brains cortical-column voting circle (Clay et al. 2024), Banino's hexagonal grid response, a 7-agent multi-agent debate arrangement.

### THE EGG

**Intuition.** The Seed plus its six inter-petal Vesicas, promoted to first-class regions. The configuration where the overlaps between adjacent Seed petals are no longer ephemeral but stored as named nodes with their own lineage and capacity.

**Formal.** An Egg $E$ is a Seed $S$ together with the six promoted inter-petal Vesicas. Cardinality: 1 core + 6 petals + 6 promoted Vesicas = **13 nodes**.

**Why 13.** This is exactly the cardinality of the Fruit of Life, the 13 circles obtained by selecting the central circle and the six "outer" circles of the next ring of the Flower of Life. It is the same 13 vertices that Metatron's Cube draws as a complete graph (78 edges, $\binom{13}{2}$). The fact that the Egg construction lands on 13 is not coincidence; it is forced by the geometry. We make this load-bearing in the next two definitions.

### THE FLOWER

**Intuition.** The Egg recursed once: every newly born Vesica becomes the center of its own Seed.

**Formal.** A Flower $F$ is the union of the Egg with a new Seed grown around each of its promoted Vesicas. In the canonical 2D embedding the visible cardinality is 19 nodes (the standard 19-circle Flower of Life diagram); in the actual computational graph the Flower extends to whatever recursion depth is supported, and each Seed brings its own potential Egg. The recursion is the generative rule.

**Implementation note.** The Flower is best stored not as a fixed graph but as a *generative process* over a graph of Regions: the recursion depth is a parameter, the spawning of new Seeds is conditional on Vesica density and persistence (see §2.4 step 4), and consolidation (step 7) prunes branches that fail to earn their cost.

### THE FRUIT

**Intuition.** Council mode. The 13 vertices of the Egg, taken with full pairwise connectivity.

**Formal.** A Fruit $F^*$ is a 13-region complete-graph configuration. Total edges: $\binom{13}{2} = 78$. This corresponds to Metatron's Cube, in which all 13 circle-centers of the Fruit of Life are joined by all 78 line segments. Drawn this way, the figure contains projections of all five Platonic solids — a fact that is mathematically genuine and was the original motivation for Renaissance interest in the figure.

**Use.** When a query is high-stakes enough to warrant full mesh deliberation, MANDORLA invokes the Fruit and runs Metatron's Operator over it.

### METATRON'S OPERATOR

**Intuition.** Full-mesh attention/voting over the 13 nodes of a Fruit.

**Formal.** Given a Fruit $F^* = \{V_1, \ldots, V_{13}\}$, Metatron's Operator is

$$M(F^*) = \mathrm{Aggregate}\!\left(\{\mathrm{Attend}(V_i, V_j) : i \neq j\}\right)$$

where Attend is a standard scaled-dot-product attention over Region representations,

$$\mathrm{Attend}(V_i, V_j) = \mathrm{softmax}\!\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right) V_j^{\mathrm{val}}$$

and Aggregate is a votes-weighted aggregation over the 13 outputs. This is exactly $K_{13}$ attention. Cost: $13 \times 12 = 156$ directed attention pairs, or 78 undirected. Cheap. Used selectively; the per-cycle default is Seed-mode (§2.4).

## 2.2 The Mandorla Memory Architecture

A MANDORLA memory is a directed multigraph whose nodes are Regions and whose edges record lineage and co-retrieval. The interface below is given in Python, using Pydantic models for runtime-validated I/O and dataclasses where mutability matters. This is the **reference interface** that Experiment 1 (§3.4) implements first.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, TypeAlias
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# ─── Aliases ────────────────────────────────────────────────────────────────

ULID: TypeAlias = str                       # sortable opaque id
Vec:  TypeAlias = NDArray[np.float32]       # d-dim embedding (e.g. d=768 or 4096)

# ─── Extents (the geometric body of a Region) ───────────────────────────────

class BoxExtent(BaseModel):
    """Axis-aligned hyperrectangle. Vesica = per-dimension max/min of corners."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: Literal["box"] = "box"
    min_corner: Vec
    max_corner: Vec

class BallExtent(BaseModel):
    """L2 ball. Vesica = lens of two balls; centroid + effective radius."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: Literal["ball"] = "ball"
    radius: float = Field(gt=0)

class SDMExtent(BaseModel):
    """Sparse-distributed-memory hard-sphere in {0,1}^d. Vesica = intersection
    of two Hamming-radius hard spheres."""
    kind: Literal["sdm"] = "sdm"
    hard_radius: int = Field(ge=0)

Extent: TypeAlias = BoxExtent | BallExtent | SDMExtent

# ─── Lineage and metadata ───────────────────────────────────────────────────

@dataclass
class Lineage:
    parents: Optional[tuple[ULID, ULID]] = None   # None for primordial regions
    depth: int = 0                                # 0 for primordial; +1 per vesica

@dataclass
class RegionMeta:
    overlap_history: list[float] = field(default_factory=list)
    co_retrieval_count: int = 0
    promotion_score: float = 0.0                  # see §2.4 step 6

# ─── Region (a node in the memory graph) ────────────────────────────────────

@dataclass
class Region:
    id: ULID
    center: Vec
    extent: Extent
    birth_time: float                             # monotonic clock
    lineage: Lineage = field(default_factory=Lineage)
    citations: set[ULID] = field(default_factory=set)
    meta: RegionMeta = field(default_factory=RegionMeta)

    def is_primordial(self) -> bool:
        return self.lineage.parents is None

@dataclass
class Vesica(Region):
    """A Region whose parents are guaranteed non-null. Type-narrowed Region."""
    def parent_ids(self) -> tuple[ULID, ULID]:
        assert self.lineage.parents is not None, "Vesica must have two parents"
        return self.lineage.parents

# ─── Compositional configurations (Seed → Egg → Fruit) ──────────────────────

@dataclass
class Seed:
    core: Region
    petals: tuple[Region, Region, Region, Region, Region, Region]   # exactly 6

    def __post_init__(self) -> None:
        if len(self.petals) != 6:
            raise ValueError(f"Seed requires exactly 6 petals; got {len(self.petals)}")

@dataclass
class Egg:
    seed: Seed
    vesicas: tuple[Vesica, Vesica, Vesica, Vesica, Vesica, Vesica]  # 6 inter-petal vesicas

    def __post_init__(self) -> None:
        if len(self.vesicas) != 6:
            raise ValueError(f"Egg requires exactly 6 promoted vesicas; got {len(self.vesicas)}")

    def all_nodes(self) -> tuple[Region, ...]:
        return (self.seed.core, *self.seed.petals, *self.vesicas)   # 1+6+6 = 13

@dataclass
class Fruit:
    vertices: tuple[Region, ...]   # length 13; full K_13 connectivity is implicit

    def __post_init__(self) -> None:
        if len(self.vertices) != 13:
            raise ValueError(f"Fruit requires exactly 13 vertices; got {len(self.vertices)}")

    @property
    def edge_count(self) -> int:
        return 78   # binomial(13, 2)
```

The store exposes five operations. Their signatures are part of the spec; the implementation choice (box / ball / SDM / hybrid) is decided by the experiment in §3.1.

```python
class MandorlaStore(Protocol):
    def write(self, region: Region) -> ULID:
        """Allocate an id, insert into the index, return the id. Idempotent on
        identical centers/extents up to ε."""

    def intersect(self, a: Region, b: Region) -> Optional[Vesica]:
        """Compute geometric intersection of a.extent and b.extent. If empty
        (or, under GumbelBox, with expected volume below threshold), return None.
        Otherwise produce a Vesica with:
          center  = centroid of the intersection
          extent  = the intersection itself
          lineage = Lineage(parents=(a.id, b.id), depth=max(a.depth, b.depth)+1)
        The returned Vesica is *not yet persisted*; call promote() to persist."""

    def promote(self, vesica: Vesica) -> ULID:
        """Persist a Vesica as a first-class Region: write() it and register
        lineage pointers in both parents' citations. Promotion is the
        *learning event* — Hebbian for regions of meaning."""

    def walk(self, query: Vec, depth: int) -> "Subgraph":
        """From the regions whose centers are nearest to `query`, recursively
        follow lineage edges and Vesica links for `depth` steps; return the
        induced subgraph of regions and their Vesicas.

        1-hop  → regions containing the point
        2-hop  → Vesicas of those regions
        3-hop  → Vesicas of Vesicas (i.e. reasoning chains)"""

    def vote(self, council: Fruit) -> "Distribution":
        """Run K_13 attention via Metatron's Operator over the 13 region
        representations; return a softmax distribution over candidate outputs."""
```

The architecture is **interface-specified, not implementation-specified**. The substrate could be box embeddings, HDC, SDM, or a hybrid; the experimental program in Part III will choose.

## 2.3 The Mandorla Agent Topology

The communication graph of a MANDORLA multi-agent system grows by Flower-of-Life construction, which is the agent-level analogue of the memory-level Seed→Egg→Flower recursion in §2.1.

```
            Stage 2 — Seed (7 agents)              Stage 3 — Egg (13 agents)
                                                   Vesicas promoted to agents
                                                          ───────
                  P₁                                    V₆₁     V₁₂
               ╱     ╲                                ╱            ╲
            P₆        P₂                            P₆ ── P₁ ── P₂
             │  CORE  │                              │   CORE   │
            P₅        P₃                            P₅ ── P₄ ── P₃
               ╲     ╱                                ╲            ╱
                  P₄                                    V₅₆     V₂₃
                                                          ───────

           Stage 4 — Fruit / Council
        13 agents, full K₁₃ deliberation
```

- **Stage 0 — Bootstrap.** One agent. The seed-of-self. This is the smallest viable system; it has no Vesicas yet.
- **Stage 1 — Vesica.** Two agents. They share an explicit Vesica subject — a named semantic region they both write to and read from. This is the smallest configuration with a non-trivial overlap surface, and it is where most existing "two-LLM critic" systems live.
- **Stage 2 — Seed.** One coordinator + six specialists in hexagonal arrangement; each adjacent specialist-pair shares a Vesica subject; the coordinator shares a Vesica with each specialist. Total: 7 agents, 6 + 6 = 12 Vesica subjects.
- **Stage 3 — Egg.** The six inter-specialist Vesicas are *promoted* to first-class agents. Each promoted Vesica-agent's job is to maintain the intersection between two specialists: to summarize what they agree on, to flag where they diverge, to keep a running representation of their overlap that either specialist can read. Total: 13 agents.
- **Stage 4 — Fruit / Council.** When the system enters council mode, the 13 agents temporarily form a $K_{13}$ deliberation graph: every agent attends to every other (Metatron's Operator). Used for high-stakes decisions only; expensive.
- **Stage 5 — Flower.** The recursive case. Each Vesica-agent becomes the core of its own Seed: the system spawns six new specialist-agents around it, and the construction repeats. The resulting communication graph is, at any depth $d$, the Flower of Life expanded $d$ times.

**Message-passing semantics.** Each agent has:

- **Own subject** (publish + subscribe; durable work queue).
- **Vesica subjects** with each adjacent neighbor (publish + subscribe; fan-out). These are the persistent overlap surfaces.
- **Coordinator subject** (subscribe; fan-out). For receiving directives from the local Seed's coordinator.

Messages adhere to a **Cortical Messaging Protocol (CMP)** compatible with the Thousand Brains Project framing (Clay et al. 2024; public spec at https://thousandbrainsproject.readme.io/docs/cortical-messaging-protocol). Each message carries not just content but a coordinate-grounded reference frame:

```python
from typing import Any, Optional
from pydantic import BaseModel, Field

class Pose(BaseModel):
    location: list[float]                   # coordinate in shared reference frame
    rotation: list[float]                   # orientation vector

class CMPMessage(BaseModel):
    sender_id: ULID
    confidence: float = Field(ge=0.0, le=1.0)
    pose: Pose
    object_id: Optional[str] = None         # high-level conceptual identifier
    features: Optional[dict[str, Any]] = None
    vesica_context: Optional[ULID] = None   # the Vesica subject this message participates in
```

The decisive structural commitment: **agents communicate via persistent Vesica subjects, not via direct unicast.** This is what makes the topology connection-first rather than hierarchy-first. There is no parent-child edge anywhere in the graph that is not also someone's Vesica.

Substrate is implementation-agnostic. NATS-style hierarchical subjects with wildcards (`mandorla.flower.depth2.seed3.vesica.specialist1.specialist4`) are one natural fit, because the subject hierarchy directly mirrors the geometric one. Celery / Redis Streams / RabbitMQ are alternatives. The Thousand Brains CMP itself is the closest pre-existing protocol designed for repeated-unit voting, and MANDORLA can be read as proposing a CMP-compatible *graph topology* for the units it connects.

## 2.4 The Mandorla Cognitive Cycle

The per-thought cycle of a MANDORLA system, abstracted away from any particular implementation:

```
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ 1. INVOCATION│───▶│  2. DESCENT  │───▶│ 3. VESICA    │
   │              │    │ nearest-k    │    │   SEARCH     │
   └──────────────┘    └──────────────┘    └──────┬───────┘
                                                   │
                                                   ▼
                                           ┌──────────────┐
                                           │ 4. SEED      │
                                           │  FORMATION   │
                                           └──────┬───────┘
                                                   │
                            ┌──────────────────────┴───┐
                            ▼ (default)                ▼ (high-stakes)
                     ┌──────────────┐          ┌──────────────┐
                     │ 6. PROMOTION │          │ 5. COUNCIL   │
                     │   & decay    │◀─────────┤ Metatron K₁₃ │
                     └──────┬───────┘          └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ 7. CONSOLI-  │  (offline / "sleep")
                     │  DATION      │
                     └──────────────┘
```

1. **INVOCATION.** A query enters at some point in the Flower. It is encoded into the same embedding space as the Regions.
2. **DESCENT.** The query is routed to its top-$k$ nearest Regions by center distance. This is vanilla nearest-neighbor retrieval and serves as the lower bound of behavior — if MANDORLA does no better than this, it has no reason to exist.
3. **VESICA SEARCH.** For each pair $(A, B)$ among the top-$k$ retrieved regions, `intersect(A, B)` is computed. The resulting candidate Vesicas are scored by overlap volume × query-alignment. Candidate Vesicas already `promote`-d are looked up; novel ones are computed on the fly.
4. **SEED FORMATION.** If a candidate Vesica is dense enough (overlap above threshold $\tau_v$) and persistent enough (productively co-retrieved more than $\tau_p$ times in recent history) it spawns a Seed: a 6-petal configuration around it, where the petals are drawn from the Region neighborhood. The system reasons over the Seed.
5. **COUNCIL (optional).** For high-stakes queries (signaled by user, by uncertainty, or by a meta-heuristic), the active configuration is expanded to a Fruit and Metatron's Operator runs $K_{13}$ attention. Output: a votes-weighted answer plus an explicit disagreement spectrum.
6. **PROMOTION.** Vesicas that were productively used in this cycle — i.e., contributed to the answer with high attention weight — get their `promotion_score` incremented. When the score crosses $\theta_{\text{promote}}$, `promote()` is called, and the Vesica becomes a first-class Region available to future cycles. Promotion is *learning*. It is geometric, not gradient-descent. It is the MANDORLA analogue of a Hebbian synapse, but operating at the level of named regions of meaning rather than synaptic weights.
7. **CONSOLIDATION.** Periodically — analogous to sleep — the Vesica graph is pruned and reorganized. Rarely-traversed Vesicas decay (`promotion_score` decays geometrically; below a floor, the Region is GC'd unless cited). Densely-traversed Vesicas get re-centered (their center is updated to the centroid of their actual recent uses). Clusters of Vesicas with high mutual overlap are candidate-merged. This is closely related to the consolidation step in Wu et al.'s Kanerva Machine and to the offline replay mechanisms in Whittington et al. (2020) TEM.

The cycle is intentionally simple. The work is in the primitives.

## 2.5 The Markov-Blanket Identity

This section makes a load-bearing structural claim. We state it cleanly, and we mark it as — to my knowledge — novel, therefore the highest-priority target for adversarial mathematical review.

**Claim.** *The Vesica $V(A, B)$ between two regions $A$ and $B$ is, structurally, a Markov blanket between them.* The internal states of $A$ and the internal states of $B$ are conditionally independent given the contents of $V(A, B)$.

In Friston's formulation (Friston 2019; Parr, Da Costa & Friston, *Phil. Trans. A* 2020; Hipólito et al. 2021), a Markov blanket is the partition of states (sensory + active = blanket) that renders the internal states of a "particular" conditionally independent of the external states:

$$p(I, E \mid B) = p(I \mid B) \cdot p(E \mid B)$$

A Vesica appears to be exactly such a partition for two Regions: anything $A$ "knows" about $B$ that is decision-relevant is, by construction, in the overlap of their supports — that is, in the Vesica. Anything outside the Vesica is, for purposes of $A \leftrightarrow B$ exchange, conditionally independent.

If this identification holds, three consequences follow:

1. **MANDORLA Regions are *particles* in Friston's sense.** Each has internal, blanket, and external states. The Vesicas are the blankets.
2. **The Flower-of-Life construction is the spatial recursion of nested Markov blankets.** In the FEP literature, "Markov blankets of Markov blankets" is the standard formulation for hierarchical self-organization (Friston 2019; Parr et al. 2020). MANDORLA's Seed→Egg→Flower→Fruit recursion is, structurally, that nesting drawn explicitly. To my knowledge, no one has previously identified the Flower-of-Life construction as a literal diagram of Markov-blanket recursion. We make the claim cleanly and ask the FEP community to falsify it.
3. **Continuous-density extension.** Possati (*Markov Blanket Density and Free Energy Minimization*, arXiv:2506.05794, 2025) reframes Markov-blanket density as a continuous scalar field $\rho(x) \in [0, 1]$ over space, where $\rho = 1$ indicates perfect conditional independence and $\rho = 0$ indicates maximal coupling. We adopt this proposal as a *working hypothesis*, not as established formalism — Possati 2025 is a recent (June 2025) preprint with limited independent citations as of this writing, and we use the symbol $\rho$ to match Possati's notation. Under that framing, MANDORLA's cognitive cycle (§2.4) becomes implementable as gradient flow on a Possati-style scalar field over Region densities, where $-\log \rho$ plays the role of variational free energy.

The Markov-blanket identity is therefore not a metaphor — it is a structural claim with a specific failure mode (§3.3 F4): if d-separation properties of an actual Markov blanket are mathematically incompatible with vesica-piscis intersection geometry under realistic distributions, this thesis breaks and the FEP–MANDORLA bridge collapses. It is one of the highest-priority items in the open roadmap (§3.2).

## 2.6 Hex Working Memory

We propose, more briefly, that the working memory / context window of a MANDORLA system is laid out not as a sequence (current transformer practice) but as a **hex grid** — a 2D triangular lattice with 6-connectivity.

```
   Cartesian (8-connected, anisotropic)     Hexagonal (6-connected, isotropic)

        o─────o─────o                              o       o
        │ \   │   / │                              │ \   / │
        │   \ │ /   │                              │   x   │
        o─────x─────o                              o   │   o
        │   / │ \   │                              │   │   │
        │ /   │   \ │                              │ /   \ │
        o─────o─────o                              o       o
   (diagonal neighbors √2× further)        (all neighbors equidistant)
```

Concretely: the KV-cache cells are placed at hex grid positions (axial or cube coordinates with $a + b + c = 0$). Adjacent cells share Vesicas; the global topology has uniform 6-connectivity; the cost of moving attention between any two cells is proportional to hex-grid distance ($O(\sqrt{n})$), not to sequential token distance. The model can attend to a "nearby" piece of working memory without first traversing every token in between, the way it has to under positional encoding of a 1D sequence.

The hex-grid layout is the natural one for two reasons. First, by Hales' honeycomb theorem, it minimizes per-cell perimeter for equal areas, which means it minimizes the total surface area of inter-cell Vesicas while keeping uniform connectivity — the right trade-off if the cost of a Vesica edge is non-trivial. Second, it is the topology that grid cells use to encode both physical and conceptual space (Constantinescu et al. 2016), suggesting that if you want a context window that supports the kind of relational generalization grid cells enable, you should give it the same shape.

Implementation candidates exist. HexagDLy (Steppa & Holch, *SoftwareX* 2019; arXiv:1903.01814) extends PyTorch with native hexagonal convolutions and pooling. HexCNN (Zhao, Ke, Korn, Qi & Zhang, ICDM 2020; arXiv:2101.10897) provides a more memory-efficient native hex-CNN framework. Reported gains versus rectangular-grid-with-padding baselines: ~25% memory reduction at data loading, ~42% memory reduction during convolution, ~42% reduction in training time (Zhao et al. 2020 — these numbers are domain-specific to *convolution* and should be re-validated for the attention-layer setting MANDORLA uses, which is a different operator family). MANDORLA's Hex Working Memory layer would be a transformer attention variant whose positional encoding is hex-grid axial coordinates and whose attention bias is hex distance. This is a concrete falsifiable architectural choice and we list it as Year-2 territory in §3.2.

---

# PART III — THE RESEARCH BLUEPRINT

## 3.1 The First Three Experiments

Each is named, scoped, falsifiable, runnable in 4–12 weeks by a small team, and tied to one of the three theses in §1.4.

### EXPERIMENT 1 — VESICA-RAG vs. Vanilla RAG

**Tests Thesis 2 (Connection over Hierarchy).**

**Hypothesis.** A retrieval system that explicitly indexes and retrieves *intersections* (promoted Vesicas) outperforms vanilla nearest-neighbor RAG on tasks requiring multi-concept reasoning — queries that name two or more distinct concepts whose answer lives at their intersection.

**Setup.**

- Build a corpus where ground-truth answers are known to lie at the intersection of $k \geq 2$ semantic regions. The natural fit is multi-hop QA: HotpotQA (Yang et al. 2018), MuSiQue / MuSiQue-Ans (Trivedi et al., TACL 2022, arXiv:2108.00573), 2WikiMultiHopQA (Ho et al., COLING 2020). A 2-hop or higher question is, definitionally, asking for the intersection of two or more facts.
- Implement **Vesica-RAG**: index chunks of Wikipedia using GumbelBox embeddings (Dasgupta et al. 2020, via `iesl/box-embeddings`) on top of a strong dense retriever (ColBERTv2 or contriever). On each retrieval, compute pairwise box intersections among the top-$k$, score each Vesica by expected overlap volume × cosine to query, retrieve from the union of points and (promoted) Vesicas. Maintain a Vesica store with promotion / decay per §2.4.
- Compare against:
  - Sparse: BM25.
  - Dense: ColBERTv2 (Santhanam et al., NAACL 2022, arXiv:2112.01488), contriever (Izacard et al., TMLR 2022, arXiv:2112.09118).
  - Hybrid: BM25 + dense reranking.
  - Generative augmentation: HyDE (Gao et al. 2023).
  - Hierarchical: RAPTOR (Sarthi et al. ICLR 2024, arXiv:2401.18059).
  - Graph: HippoRAG (Gutiérrez et al., NeurIPS 2024, arXiv:2405.14831); HippoRAG 2 (Gutiérrez et al. 2025, arXiv:2502.14802).
  - Hypergraph: HyperGraphRAG (Luo et al., arXiv:2503.21322), Hyper-RAG (Feng et al., arXiv:2504.08758), Cog-RAG (Hu et al., AAAI 2026, arXiv:2511.13201).
  - (Note: Microsoft GraphRAG (Edge et al., arXiv:2404.16130) was designed for query-focused summarization, not multi-hop QA; included for completeness, not as a primary baseline.)

**Falsifiable prediction (pre-registered).** Vesica-RAG shows ≥10% relative improvement in answer F1 / EM on 2+ hop subsets of MuSiQue and 2WikiMultiHop, while not degrading on single-concept (1-hop) queries by more than 2% relative. Both conditions must hold. If either fails, Thesis 2 is undermined and we report it.

**Auxiliary measurement.** A **vesica-coverage metric**: the fraction of multi-hop questions whose ground-truth answer chunk pair was actually identified by the retriever as a single Vesica before the LLM saw it. This metric is intrinsic to MANDORLA and tests the "intersection retrieval" claim independent of LLM strength.

**Timeline.** 6–8 weeks. (See §3.4 for the *smallest defensible slice* sized for solo execution: HotpotQA dev set, contriever baseline, two weeks.)

### EXPERIMENT 2 — HEX-VOTE: A 7-Node Cortical Council

**Tests Thesis 1 (Geometric Primacy).**

**Hypothesis.** A multi-agent reasoning system in **Seed configuration** (1 coordinator + 6 specialists, each adjacent specialist-pair sharing a persistent Vesica subject) outperforms (a) single-agent CoT, (b) flat 7-agent debate, (c) hub-and-spoke 7-agent orchestrator on tasks requiring synthesis across multiple specialty domains.

**Setup.**

- 7 LLM instances (same base model, role-prompted). Coordinator publishes the query; specialists work in their specialty; each adjacent specialist-pair has a Vesica subject they both publish to and read from; coordinator reads the 6 Vesica subjects and the 6 specialist outputs and synthesizes.
- Topology variants:
  - **Hex-Vote (Seed):** as above. Six lateral Vesica subjects.
  - **Flat debate:** 7 agents, all-to-all ($K_7$). AutoGen GroupChat as analogue.
  - **Hub-and-spoke:** 1 orchestrator, 6 leaves, no lateral Vesicas. CrewAI hierarchical mode.
  - **Single-agent CoT.** Same base model, chain-of-thought prompting.
- Benchmarks: MMLU-Pro (Wang et al. 2024); GPQA Diamond (Rein et al. 2024); a custom synthesis benchmark (held out) where each item explicitly requires combining knowledge from ≥3 of the 6 specialty domains.
- **Token budget held constant** across all conditions to avoid the "more tokens always wins" confound.

**Falsifiable prediction (pre-registered).** Seed (Hex-Vote) outperforms hub-and-spoke and flat debate on synthesis-heavy subsets by ≥5% relative accuracy, with token cost within ±3% of the highest-cost baseline. If Seed loses to either alternative on the synthesis subset under matched compute, Thesis 1 is weakened.

**Baselines worth naming.** AutoGen GroupChat (Wu et al. arXiv:2308.08155); CrewAI hierarchical; AgentNet; RUMAD; AMAS.

**Why this is the right test of geometric primacy.** If hex topology is just decoration, performance will be invariant to topology under matched compute. If the geometry is doing something, it should show up here, where the only difference between conditions is the shape of the communication graph.

**Timeline.** 4–6 weeks.

### EXPERIMENT 3 — RECURSIVE CONSTRUCTION: Mandorla Curriculum vs. Standard Training

**Tests Thesis 3 (Recursive Construction).**

**Hypothesis.** A learning curriculum in which the model is forced to *construct new representations at the intersection of pre-existing representations* — i.e., where the loss explicitly rewards predicting the Vesica of two named entities from their parents *and* predicting the parents from the Vesica — produces better compositional generalization than standard MLM/CLM training.

**Setup.**

- Two small transformers (100M–300M params), trained from scratch on the same corpus (a clean subset of C4 or RedPajama).
- **Baseline:** standard CLM, with a span-corruption auxiliary loss à la T5.
- **Mandorla-curriculum:** the same model and corpus, but with two additional self-supervised losses:
  1. **Vesica prediction:** given two co-occurring named entities $E_1$ and $E_2$ in a paragraph, the model must predict a representation of $V(E_1, E_2)$ (operationalized as a contrastive objective against a held-out paragraph containing both $E_1$ and $E_2$); the entity-pair embeddings are trained such that their box-intersection contains the third-entity-pair embedding from the held-out paragraph.
  2. **Parent reconstruction:** given a Vesica representation, the model must predict back its parents (a denoising objective).
- Compositional generalization benchmarks: COGS (Kim & Linzen, EMNLP 2020, arXiv:2010.05465), SCAN (Lake & Baroni, ICML 2018), ReCOGS / ReCOGS_pos (Wu, Manning & Potts, **TACL 2023**, arXiv:2303.13716).
- I.i.d. evaluation (held-out test from the same distribution) to confirm no degradation.

**Falsifiable prediction (pre-registered).** Mandorla-curriculum model shows ≥15% **relative** improvement in accuracy on the systematic generalization splits (COGS Gen split, SCAN length / primitive splits, ReCOGS structural generalization) while losing no more than 2% **relative** on the i.i.d. test accuracy. Both conditions must hold. (Note: *relative*, not *absolute* — 15% absolute on COGS Gen would be enormous; we are committing to relative-to-baseline.)

**Baselines.** Standard CLM; span-corruption (T5); the Csordás et al. 2021 (arXiv:2108.12284) "simple tricks" Transformer that raises COGS to ~81% (this is the harder baseline to beat); Ontañón, Ainslie, Cvicek & Fisher (arXiv:2108.04378); Furrer et al. 2020 (arXiv:2007.08970) pre-training comparison. We note that Tree-of-Thoughts (Yao et al., NeurIPS 2023, arXiv:2305.10601) is an inference-time search method; if we use ToT-style derived trajectories as a training-data augmentation, we cite both the original ToT paper and the specific downstream augmentation pattern used.

**Why this is the right test of recursive construction.** Standard CLM teaches the model to predict the next token given prior tokens. The Mandorla curriculum teaches it to *construct a new region from the geometric overlap of two existing regions, and to construct the parents from the overlap*. If the recursive-construction thesis is right, this should improve exactly the kind of OOD generalization that COGS / SCAN / ReCOGS measure, because those benchmarks are designed to test whether the model can compose representations it has not seen composed before.

**Timeline.** 8–12 weeks (training a 100–300M model from scratch is the longest of the three).

## 3.2 The Open Roadmap (Year 2+)

This is an explicit invitation to other researchers. The questions below are out of scope for the first three experiments but are the natural Year-2 extensions.

1. **Grounding Vesicas in LLM internals.** What is the right way to ground Vesica regions in the parameters of a pretrained LLM? Sparse-autoencoder features (Anthropic — Bricken et al. 2023; Templeton et al. 2024, *Scaling Monosemanticity*) are the obvious candidate — each SAE feature is a candidate Region — but is feature intersection (the co-occurrence of two features in a residual stream) the right operationalization of a Vesica? Attribution-graph nodes (Lindsey et al., Anthropic 2025, *Circuit Tracing* / *On the Biology of a Large Language Model*) are another candidate. This is open territory and probably the highest-value Year-2 question.
2. **Flower-of-Life as generative grammar for NAS.** Can the Seed→Egg→Flower→Fruit construction be turned into a generative grammar for neural architecture search? Each Seed is a sub-network; each promoted Vesica becomes a new sub-network; the growth rule is fixed. Compare to FractalNet (Larsson et al. 2017) and to NAS over modular spaces.
3. **Markov-blanket recursion formalization.** How does §2.5's identification connect formally to Friston's hierarchical free-energy minimization? The geometrized FEP (Possati 2025) is the most plausible bridge — pending wider validation of that paper. Concretely: can the MANDORLA cognitive cycle (§2.4) be derived as gradient descent on a Possati-style scalar field $\rho$ over Region densities?
4. **$\Phi$ of a Seed.** What is the integrated information $\Phi$ (in the Tononi/IIT 4.0 sense, Albantakis et al. 2023) of a Seed configuration vs. an isolated agent vs. a hub-and-spoke configuration? Does MANDORLA generate measurably higher $\Phi$ than baselines? The geometric-$\Phi$ formulation (Oizumi et al. 2016) is the tractable form; the MANDORLA hex topology is a clean test case because of its uniform 6-connectivity.
5. **Emergent grid cells in Hex Working Memory.** Banino et al. (2018) showed that hexagonal grid representations emerge spontaneously in path-integration networks. Does a MANDORLA system with hex working memory and a path-integration auxiliary objective produce measurable grid-cell-like representations in its working-memory cells? If yes, this is a strong convergent-evidence result.
6. **MANDORLA scaling laws.** Hypothesis: MANDORLA's Flower-of-Life growth rule yields communication-edge cost that is *linear* in the number of agents, rather than the quadratic cost of all-pairs communication. At depth $d$, the Flower has $n \sim 6^d$ nodes and $O(6^d) = O(n)$ Vesica edges, vs. $O(n^2)$ for all-pairs. Both grow exponentially in depth, but the *edge-to-node ratio* stays bounded for MANDORLA and grows linearly with $n$ for all-pairs. Does this translate to empirical compute savings while maintaining performance?
7. **Categorical formalization.** Can MANDORLA be expressed as a specific monad in the Symbolica/DeepMind 2-category of parametric maps (Gavranović et al., ICML 2024)? The "vesica monad" should have unit $\eta$ that promotes a Region to its singleton intersection with itself, and multiplication $\mu$ that flattens intersection-of-intersections back to a single Vesica. Spelling this out would give MANDORLA the same kind of clean algebraic foundation that GDL gave to architecture design.
8. **Topology-preserving learning.** When `promote()` creates a new Region, the embedding space's topology shifts. What invariants does the system preserve across this shift? Closely related to sheaf-cohomological invariants in Bodnar et al. (NeurIPS 2022) and Hansen & Gebhart (2020); MANDORLA's promotion event may correspond to a specific cohomological-class-preserving update.

These are not roadblocks. They are the territory we want collaborators to claim.

## 3.3 Falsifiability and Failure Modes

The disciplinary commitment of this document is that MANDORLA is falsifiable *in public*. Below are the conditions under which we will report MANDORLA as falsified or merely decorative:

- **F1 — Retrieval failure.** If Vesica-promoted regions consistently underperform raw nearest-neighbor on Experiment 1's multi-concept queries, Thesis 2 is undermined and the central architectural claim is wrong.
- **F2 — Topology irrelevance.** If Hex-Vote fails to outperform hub-and-spoke and flat debate on Experiment 2's synthesis benchmarks under matched token budget, Thesis 1 is weakened. (A single experiment is not decisive; if F2 fires, we will investigate scaling, base-model choice, and benchmark design before concluding.)
- **F3 — Curriculum failure.** If the Mandorla curriculum produces no compositional-generalization advantage in Experiment 3, Thesis 3 is falsified at small scale; we then either revise the curriculum or report the negative result.
- **F4 — Mathematical incompatibility.** If the Markov-blanket identification in §2.5 is shown to be mathematically broken — i.e. if some structural property of the vesica-piscis intersection is incompatible with the conditional-independence semantics of a Markov blanket under realistic distributions — then §2.5's identity claim must be retracted, even if the rest of the spec survives.
- **F5 — Baseline redundancy.** If MANDORLA does not exhibit any computational advantage over a hypergraph-NN baseline (HGNN, AllSet/AllSetTransformer, ED-HNN, HyperGraphRAG, Cog-RAG) at matched parameters, the differentiator collapses and MANDORLA reduces to a renaming.
- **F6 — Grid anisotropy.** If hex topology shows no advantage over fully-connected attention at any scale where they can be fairly compared, the "geometric primacy" thesis is empirical noise.

**Pre-registration** is the discipline we commit to. The first three experiments will have hypotheses, predictions, and analysis plans deposited publicly (OSF Registries) before the data is collected.

## 3.4 The Build Path

Concrete 12-week first build, sized for a small team (1–2 engineers + 1 researcher):

```
W1–W2:   Implement Vesica primitive + Region store (Python / PyTorch)
         · Use iesl/box-embeddings (pip install box-embeddings) for GumbelBox
         · write / intersect / promote / walk / vote
         · HDC-bundling alternate for ablation

W3–W4:   Build promotion / decay / consolidation
         · Train consolidation loop on a held-out validation stream
         · Verify Vesica graph grows sub-linearly in corpus size

W5–W6:   Run Experiment 1 (Vesica-RAG)
         · Index ~1M Wikipedia chunks (HotpotQA wiki dump for the slice)
         · HotpotQA, MuSiQue, 2WikiMultiHop
         · Pre-register predictions before evaluation (OSF Registries)

W7–W8:   Build Seed agent topology (NATS subjects + CMP messages)
         · Implement four topology variants for Experiment 2
         · Stand up the synthesis benchmark

W9–W10:  Run Experiment 2 (Hex-Vote)
         · MMLU-Pro, GPQA Diamond, custom synthesis bench

W11–W12: Analysis, write-up, open-source release
         · Whether result is positive or negative, publish
```

For a solo first pass, the **smallest defensible slice** is Vesica-RAG on the HotpotQA dev set, against a single dense baseline (contriever), with the vesica-coverage auxiliary metric. Two weeks of focused work. One number on one benchmark. If that number is good, the rest of the program is justified. If it's bad, three months are saved.

Experiment 3 (Mandorla curriculum) is a parallel track on a different timescale, requiring training infrastructure for 100–300M-param models from scratch. Best conducted as a 3–4 month effort by a separate sub-team with access to a small compute cluster.

## 3.5 Closing — The Geometry of the Next Architecture

The point of MANDORLA is not that the Flower of Life is sacred. The point is that the geometric primitives mathematics (Hales' honeycomb theorem, Tóth's disk packing, the Newton–Gregory kissing numbers), neuroscience (entorhinal grid cells coding both physical and conceptual space), and engineering practice (the rediscovery, in geometric and topological deep learning, that the right prior is the right shape) keep converging on are the ones distributed cognition will turn out to need. The Flower of Life is a memorable, generative, falsifiable diagrammatic compression of that convergence. It is — and this is important — *not* a claim that the universe is shaped this way. It is a claim that *cognition under packing-and-overlap constraints* tends to be shaped this way, and that we should design for it on purpose rather than rediscover it accidentally.

The metaphor's job is to make the engineering memorable. The engineering's job is to make the metaphor true. If §3.1 succeeds, the Flower of Life will have earned its place in the technical vocabulary of the field, and we will have a name — Mandorla, Vesica, Seed, Egg, Flower, Fruit, Metatron's Operator — for a family of constructions that the field has been groping toward without one. If §3.1 fails, the names go in the bin and we report the failure cleanly.

The primitive is the intersection. The recursion is the engine. The geometry is on trial.

Build it. Measure it. Tell us what happens.

---

## Caveats

A few caveats explicitly, in the discipline of §3.3:

- The **Markov-blanket identity (§2.5)** is presented as load-bearing but is, at the moment, a structural identification rather than a proven theorem. The probability-theoretic precise form (which family of distributions over Regions makes Vesica-as-blanket exact rather than approximate) is open, and Possati's continuous-density extension is a June 2025 preprint with limited independent citations as of this writing.
- The **GumbelBox formulation in §2.1** uses Lemma 1 and the closed-form softplus expression from Dasgupta et al. NeurIPS 2020 §4.1. Implementers should consult the paper directly and use `iesl/box-embeddings`'s `GumbelBoxTensor` rather than re-deriving.
- The **"kissing number = 6" justification of the Seed** is exact in 2D; in high-dimensional embedding spaces ($d > 2$) kissing numbers grow rapidly (e.g., the kissing number in $\mathbb{R}^{24}$ is exactly 196,560, by Cohn et al. 2017). The defensible claim: the visible/diagrammatic topology has kissing number 6 in 2D, and that 2D shadow is what neuroscience measures and what makes the system human-readable. The latent-space topology may be much richer; Year-2 work should investigate this gap.
- The **empirical "sweet spot of ~6 specialists"** in multi-agent literature is folkloric, not a theorem. We mention it only as suggestive.
- The **Thousand Brains Project comparison** is structural, not endorsing every TBP claim about consciousness or the cortex. We adopt the cortical-column-as-unit framing (well-supported) and the CMP framing, and remain agnostic on the rest. Note that Thousand Brains Project and Numenta are now distinct organizations sharing genealogy.
- The **Lemanski (2019) lineage claim** — that this is the only peer-reviewed paper joining sacred geometry and neural networks — is true to my best literature search at time of writing but is the kind of claim that ages quickly. MANDORLA's novelty does not depend on Lemanski being the only prior; it depends on the synthesis being novel.
- The **HexCNN performance numbers in §2.6** (~25% / ~42% / ~42%) are from the ICDM 2020 paper's *convolution* setting; they should be re-validated in MANDORLA's *attention-layer* setting before being relied on.
- The **"to my knowledge, novel"** claims — particularly the Markov-blanket / Flower-of-Life identification (§2.5) and the unification under one named system — are the author's best survey at time of writing. They are the claims most worth attempting to falsify on prior-art grounds, and corrections are welcome.
- **Numerical thresholds** in the predictions (≥10%, ≥15%, ≤2% degradation) are pre-committed for honesty but are first-pass guesses. They should be tightened by power analysis before formal pre-registration on OSF.

This document is the foundational stone, not the finished cathedral. It is offered in the spirit of Christopher Alexander's *A Pattern Language* (1977): a vocabulary that is meant to be used — extended, contradicted, partially adopted, partially refuted. The names are technical; the claims are falsifiable; the geometry is on trial. If you want to collaborate, the Vesica subject is open.

---

## Citation

```bibtex
@article{patterson2026mandorla,
  title  = {MANDORLA: A Geometric Foundation for Machine Cognition},
  author = {Patterson, Jacob},
  year   = {2026},
  eprint = {arXiv:TBD},
  primaryClass = {cs.AI},
  url    = {https://runascode.com}
}
```
