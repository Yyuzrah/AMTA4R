# Appendix M in detail: empirical checks of the terms of the main bound

This document accompanies Appendix M ("Empirical Checks of the Terms of the Main Bound") of the paper *A Measure-Theoretic Analysis of Reasoning in Transformers: Structural Generalization and Approximation Limits* (NeurIPS 2026) and the code in this repository (https://github.com/Yyuzrah/AMTA4R). Appendix M derives from the terms of Theorem 2 the properties that the coordinates of a projection must have and reports eight experiments. This document gives the material that the appendix leaves out: the kinds of objects in a trajectory, the properties of a coordinate with their sources in the paper, the catalogue of 48 candidate coordinates with its mappings onto Countdown and six other trajectory domains, and the designs and detailed results of the eight experiments. Each experiment section links to the folder of this repository that holds its code and its per-pair, per-model and per-run results, and the `results/README.md` of each folder explains its files and columns.

Numbers of lemmas, theorems, assumptions, definitions, corollaries, figures, tables and appendices (for example Lemma 2, Theorem 2, Figure 4, Table 3, Appendix J) refer to the paper; tables numbered with a dot (Table 1.1 to Table 10.1) are tables of this document. "Section 5 of the paper" names a section of the paper, and "Section 7" alone a section of this document. Licenses of models and datasets are given in parentheses where a model family or dataset is first named. Links are relative to this file. References are listed at the end.

## Contents

| Section | What it contains | Appendix M | Code and results |
|---|---|---|---|
| [0. Notation](#0-notation) | symbols and abbreviations used throughout | all | |
| [1. What the bound requires of a projection](#1-what-the-bound-requires-of-a-projection) | the requirement that each term of Theorem 2 places on a projection | M.1 | |
| [2. The kinds of objects in a trajectory](#2-the-kinds-of-objects-in-a-trajectory) | levels, sources and views; the domain adapter; seven kinds; the feature grammar; the twelve coordinates of the paper in that grammar; six classes of coordinates | M.1 | |
| [3. Properties a coordinate must have](#3-properties-a-coordinate-must-have) | each property, its source in the paper and what fails without it; change of a coordinate under T_k; adding coordinates; two notions of depth; width | M.1 | |
| [4. Catalogue of 48 candidate coordinates](#4-catalogue-of-48-candidate-coordinates) | the catalogue of candidate coordinates for Countdown and six other trajectory domains: adapter primitives by domain; definition and per-domain extraction of every coordinate in the classes T, C, M, N, I, S | M.1, M.1.1, M.2.1 | |
| [5. Domain mappings and conditions per domain](#5-domain-mappings-and-conditions-per-domain) | what is analysed in Countdown and each of the six other domains, recommended projections, which terms of the bound apply, conditions per domain | M.1, M.2 | |
| [6. Building a projection for a new domain](#6-building-a-projection-for-a-new-domain) | the eight-step procedure | M.1 | |
| [7. Experiment 1: from 12 to 34 coordinates on Countdown](#7-experiment-1-from-12-to-34-coordinates-on-countdown) | design and results of Experiment 1 | M.1.1 | [`coordinates_12_to_34/`](../extensions/coordinates_12_to_34/) |
| [8. Agent, embodied and robot trajectories (Experiments 2 and 3)](#8-agent-embodied-and-robot-trajectories-experiments-2-and-3) | Experiment 2: trajectory sources, groups, projections, estimator, results of the 52 pairs; Experiment 3: one model in two agent harnesses | M.2 (M.2.1, M.2.2) | [`structural_audit/`](../extensions/structural_audit/), [`harness_switch/`](../extensions/harness_switch/) |
| [9. Lemma 2 under shifted position indices (Experiments 4 to 6)](#9-lemma-2-under-shifted-position-indices-experiments-4-to-6) | Experiment 4: attention scores of eight pretrained models; Experiment 5: answers of 33 public models under the shifts that serving produces; Experiment 6: two Countdown models of Section 5 of the paper | M.3 (M.3.1, M.3.2, M.3.3) | [`lemma2_pretrained/`](../extensions/lemma2_pretrained/), [`shifted_positions_serving/`](../extensions/shifted_positions_serving/), [`shifted_positions_checkpoints/`](../extensions/shifted_positions_checkpoints/) |
| [10. Splitting deeply nested tasks into separate model calls (Experiments 7 and 8)](#10-splitting-deeply-nested-tasks-into-separate-model-calls-experiments-7-and-8) | Experiment 7: nested plans; Experiment 8: nested tool calls with eleven open-weight language models | M.4 (M.4.1, M.4.2) | [`nested_plans/`](../extensions/nested_plans/), [`nested_tool_calls/`](../extensions/nested_tool_calls/) |
| [11. Summary of the eight experiments](#11-summary-of-the-eight-experiments) | the finding of each experiment, in the order of Table 3 of Appendix M.5, with the results it rests on and how it is used | M.5 | |
| [References](#references) | works cited in this document | all | |

**The eight experiments of Appendix M.**

| Experiment | Appendix M | This document | Code and results |
|---|---|---|---|
| 1. More coordinates on Countdown | M.1.1 | [Section 7](#7-experiment-1-from-12-to-34-coordinates-on-countdown) | [`coordinates_12_to_34/`](../extensions/coordinates_12_to_34/) |
| 2. Structural shift between groups of agent, embodied and robot trajectories | M.2.1 | [Sections 8.1 to 8.6](#81-trajectory-sources) | [`structural_audit/`](../extensions/structural_audit/) |
| 3. One model in two agent harnesses | M.2.2 | [Section 8.7](#87-experiment-3-one-model-in-two-agent-harnesses) | [`harness_switch/`](../extensions/harness_switch/) |
| 4. Attention scores of pretrained models | M.3.1 | [Section 9.1](#91-experiment-4-attention-scores-of-pretrained-models) | [`lemma2_pretrained/`](../extensions/lemma2_pretrained/) |
| 5. Answers under shifted position indices | M.3.2 | [Section 9.2](#92-experiment-5-answers-under-shifted-position-indices) | [`shifted_positions_serving/`](../extensions/shifted_positions_serving/) |
| 6. Two Countdown models of Section 5 of the paper | M.3.3 | [Section 9.3](#93-experiment-6-two-countdown-models-of-section-5-of-the-paper) | [`shifted_positions_checkpoints/`](../extensions/shifted_positions_checkpoints/) |
| 7. Nested plans | M.4.1 | [Section 10.1](#101-experiment-7-nested-plans) | [`nested_plans/`](../extensions/nested_plans/) |
| 8. Nested tool calls | M.4.2 | [Section 10.2](#102-experiment-8-nested-tool-calls-with-open-weight-language-models) | [`nested_tool_calls/`](../extensions/nested_tool_calls/) |

Columns: *Experiment* gives the number and short name used in Appendix M. *Appendix M* is the section of the paper that reports it. *This document* is the section that gives its design and detailed results. *Code and results* is its folder under `extensions/`.

Figures 4, 5 and 6 of Appendix M are drawn by `extensions/structural_audit/scripts/report.py` (`fig_ratios.pdf`, Experiment 2) and by the two scripts in [`extensions/figures_appendix_m/`](../extensions/figures_appendix_m/) (Experiment 5; Experiments 7 and 8). Table 3 of Appendix M.5 summarizes the eight experiments, and Section 11 follows its rows.

## 0. Notation

- z: a trajectory, a sequence of tokens (in Countdown, the search trace). Z: the set of trajectories. x_init^(z): the problem of the trajectory, the only part the model is prompted with.
- Φ: the projection of the paper onto the twelve coordinates χ1, ..., χ12 (Appendix A.1). Φ_D: a projection for a trajectory domain D, with d_Φ coordinates. X: the space of projected points.
- μ_S, μ_T: the source (training) and target (test) distributions of trajectories. ν_S, ν_T: the distributions of Φ_D(z) under them (Appendix A.3).
- W1: the Wasserstein-1 distance between ν_S and ν_T. Ŵ1: its estimate from samples (Appendix G).
- L_f: the structural risk function of a model f, a function on X. K_f: its Lipschitz constant. ε: the sufficiency residual of Assumption 1. The shift part of the bound is K_f·W1 + 2ε (Lemma 1). L\*: the Bayes-optimal structural risk (Appendix D.1).
- T_k: the translation operator of Definition 2. It inserts one structurally inert node of k tokens, so that every later token sits k positions later. δ: how far T_k moves the point Φ_D(z). γ: the jump in risk of Assumption 2.
- T_max: the largest trajectory length in tokens. L: number of layers. m: width. α_circ·L: the depth threshold of Assumption 5 and Lemma 5. χ_nest: the coordinate that carries the depth floor.
- 1{condition}: 1 when the condition holds, 0 otherwise. s.d.: standard deviation. CV: coefficient of variation (s.d. divided by mean).
- In Sections 2 to 5, N is the number of node references in one trajectory. In Sections 7 and 8, N is the number of trajectories in one sample.
- APE: learned absolute position embeddings. RoPE: rotary position embeddings. ALiBi: attention with linear biases. SoS: the Stream of Search trajectory format. BFS, DFS, MIXED: the three search procedures that generate the training distributions μ_BFS, μ_DFS, μ_MIXED of Section 5 of the paper. f1 and f2: the two checked models of Appendix J, Small Deep MIXED RoPE and Large Deep BFS APE.
- Domains: besides Countdown, the six trajectory domains of Sections 2 to 6 are VLA (vision-language-action robot policies), tool-use agent (a language model run inside a harness that calls tools), embodied agent, world-model planner (a planner that predicts outcomes with a learned world model), long chain of thought (long CoT) and code agent.

## 1. What the bound requires of a projection

Each term of Theorem 2 places its own requirement on a projection Φ_D: Z → R^{d_Φ} for a trajectory domain D (Table 1.1). Three consequences organize Sections 2 to 6.

1. **The dimension is free.** Lemmas 1 and 4 and Assumption 4 are formulated on a general space X (Appendix M.1). With the Barron space and Assumption 4 stated on R^{d_Φ}, Theorem 2 keeps its form for any Φ_D that meets the requirements of Table 1.1, provided its assumptions, restated for Φ_D, hold in the domain.
2. **The depth coordinate can be replaced.** The proof of Lemma 5 (Appendix D.2.4) uses χ1 only through the indicator of the event {χ1 > α_circ·L} and through Assumption 5. The same argument applies to any nesting coordinate χ_nest for which that assumption is adopted, with P_implicit = 1{χ_nest > α_circ·L}. Lemma 3 motivates the assumption for push/pop nesting. For another notion of depth, the assumption is a separate modelling choice.
3. **The projection must register the insertion, by a bounded amount.** The proof of Theorem 1 uses K_{f,APE} ≥ γ/δ ≥ γ/C_δ, with C_δ independent of T_max. Every coordinate must therefore move by a bounded amount under T_k, and Φ_D must register the insertion: if δ = 0, then Φ_D(T_k z) = Φ_D(z), and the jump γ of Assumption 2 cannot occur between the two points. Under Definition 2, δ ≥ 1 because χ4 counts the inserted node (Appendix C.1). This separates quantities counted in events from quantities counted in tokens.

**Table 1.1. What each term of Theorem 2 requires of a projection Φ_D.**

| Term of Theorem 2 | Requirement on Φ_D | Source in the paper |
|---|---|---|
| Shift term K_f·W1(ν_S, ν_T) | Φ_D maps Z into a bounded subset of a Euclidean space on which L_f is Lipschitz; the proof holds for every d_Φ | Lemma 1; Appendix B.2 |
| Positional-encoding (Lipschitz) factor K_{f,PE} | T_k moves Φ_D(z) by δ with 0 < δ ≤ C_δ, where C_δ does not depend on T_max (δ ≥ 1 when a count registers the insertion, as χ4 does); in addition, as a property of the task rather than of Φ_D, one wrong routing step rules out success | Definition 2; Appendix C.1; Theorem 1; Assumption 2 |
| Depth floor C_depth·ν_T(χ_nest > α_circ·L) | a nesting coordinate χ_nest for which Assumption 5 is adopted | Lemmas 3 and 5; Assumption 5; Corollary 2 |
| Width term C_width/√m | no coordinate enters it; C_width depends on how well width-m hypotheses can approximate L\* on (X, ν_T), and so on Φ_D as a whole | Lemma 4; Assumption 4 |
| Sufficiency residual ε | under μ_S and μ_T, the expected risk equals, up to ε, the integral of one function L_f of Φ_D(z) | Assumption 1; Appendix J |

Columns: *Term of Theorem 2* names one term of the bound. *Requirement on Φ_D* states what the projection must satisfy for that term to hold. *Source in the paper* gives the statement and proof from which the requirement follows.

## 2. The kinds of objects in a trajectory

### 2.1 Levels, sources and views

**Levels.** A trajectory is described at five levels: token, event, episode (an interval of events), trajectory and measure (a distribution over trajectories). Φ_D maps a trajectory to a point of X, and ν = Φ_D#μ is the distribution of these points (Appendix A.3). Coordinates are defined at the trajectory level. The terms of the bound act at the measure level.

**Sources.** A *task-side* trajectory is produced by a reference solver, an oracle planner, an expert demonstration or an exact simulator. In the paper it is the SoS reference trajectory, and the model is prompted with x_init^(z) only (Appendix A.4). A *model-side* trajectory is the evaluated model's own output.

**Views.** The *log* contains every event that occurred, including branches that a program outside the model (an *outside program*, for example a search program that calls the model at each node of a tree) keeps outside any context. The *context* is the sequence that one forward pass reads. In the paper the two coincide. Where they differ, Φ_D remains one map of the trajectory, so that K_f and W1 refer to the same space: W1 and ε are computed from logs; the insertions that the positional-encoding term concerns must occur in the contexts that the model reads; and the depth coordinate counts nesting that is not written into those contexts (Section 3). Once the rule by which the harness builds contexts is fixed, this depth coordinate is again a function of the log.

### 2.2 The domain adapter and the trajectory event graph

A domain adapter π_D parses a trajectory z into four objects.

- **(a) An event sequence** e_1, ..., e_n with e_t = (κ_t, v_t, a_t, pos_t, len_t). Here κ_t is the kind of the event, one of act (action), obs+ (positive feedback), obs− (negative feedback), obs0 (neutral feedback), ref (node reference), restate (restatement), think (reasoning) and ext (external event); v_t is the entity concerned; a_t is the action type, taken from the action types A_D of the domain; pos_t is the token position; len_t is the token length.
- **(b) The entities V** with a parent map (a search tree or plan tree). The node-reference subsequence v_(1), ..., v_(N) gives the depths D_i = depth(v_(i)).
- **(c) Relations** among events and entities, and **episodes** [s, t] with a type (subgoal, backtracking run, failure streak, phase, verification loop).
- **(d) A push/pop word** w(z), a sequence of opening and closing brackets. It is obtained from push events (expand a child, open a subgoal, take an object, enter a file) and pop events (backtrack, close a subgoal, put an object, return to a file). A pop that returns s levels contributes s closing brackets, as in Appendix D.2.1, where every transition is expanded into parent-child moves. The homomorphism ψ of Lemma 3 is the special case of the Countdown search alphabet.

Positions come in two units: the event index t (structure) and the token position pos_t (realization). An *inert* event changes the state of no entity, introduces no entity that later events use, neither pushes nor pops and carries no feedback; it is either not an action or an identity action. T_k inserts one inert event of k tokens at the current depth, and a volume coordinate counts it. In Countdown this is the structurally inert node of Definition 2, for example an identity operation `x=x` (Appendix C.1): χ4 counts it, χ1 is unchanged, and it contains no operator symbol counted by χ11 and χ12.

The Countdown adapter is exact. `Generated Node #ids` and `Moving to Node #ids` are node references, with depth D equal to the number of ids; `Exploring Operation` is an action whose type is the operator; `unequal: No Solution` is negative feedback and `equal: Goal Reached` positive feedback; `Current State` is a restatement (Appendix K.4).

### 2.3 Seven kinds

The objects that the adapter returns fall into seven kinds (Table 2.1; instances by domain in Table 2.2). Qualities depend on a bearer, so every coordinate is an aggregate of a quality over the bearers of one kind.

**Table 2.1. The seven kinds of objects in a trajectory.**

| Kind | Definition | Constructions | Instances in χ1-χ12 |
|---|---|---|---|
| Entity / state | persists across events and can be referred to again; bears qualities (depth, branching) and enters relations (parent-child, identity) | distinct counts; maximum, mean, s.d. of a quality; graph invariants (revisit ratio, cycle rank); size of the initial state | χ1-χ3: node depth, maximum, mean, s.d. |
| Event / action | a time-stamped occurrence with a kind, a type, a position and a length: action, positive, negative or neutral feedback, node reference, restatement, reasoning step, external event | counts by kind or type; rates (one count over another) | χ4 (node references), χ8 (negative feedback), χ9 (its rate per generated node), χ11, χ12 (operator occurrences) |
| Relation | a tie between entities or events: succession, parent-child, dependency (a later event needs an earlier result or precondition), reference (a return to an earlier entity), matching (a push and its pop), feedback (action to observation), identity (a state recurs) | transition rates by relation type; distances in events between related events; nesting height of the matching relation; invariants of the relation graph | χ5-χ7 (succession by depth change); χ1 − 1 is the nesting height (Lemma 3) |
| Episode / process | a maximal interval of events that forms one unit: a subgoal from opening to closing, a backtracking run, a failure streak, a phase (localization, repair), a verification loop, the sub-trajectory ω of Definition 2 | counts; durations in events; maxima; nesting and interruption of episodes | none |
| Quality / attribute | an attribute that depends on its bearer: depth, branching, action type, feedback polarity, length, position, duration | maximum, mean, s.d., quantiles, entropy, per-type frequencies | depth (χ1-χ3), operator type (χ11, χ12), polarity (χ8) |
| Resource | a quantity consumed by events: events, environment steps, tool calls, model or world-model calls, tokens; counted in events (structure) or in tokens (realization); T_max bounds the token resource | totals; counted in events they can enter Φ_D, counted in tokens they serve as baselines and diagnostics | χ4 (volume in events); token length is the LEN baseline of Appendix J, not part of χ1-χ12 |
| Outcome / verdict | a verdict on a trajectory or a part of it: goal reached, tests passed, final answer; the reference's own outcome is task side; the evaluated model's outcome ℓ(f, z) is the label, never a coordinate | indicator; relative position of the first success (in events); continuation after it | χ10 (outcome of the reference) |

Columns: *Kind* names the kind. *Definition* says what counts as an object of that kind. *Constructions* lists the ways a number is made from objects of the kind. *Instances in χ1-χ12* names the coordinates of the paper built from the kind.

**Table 2.2. Instances of the seven kinds in Countdown and six other trajectory domains.**

| Kind | Countdown | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|---|
| Entity | search node (`#0,1,2`); remaining numbers | subgoals (task, subtask, motion); objects; gripper state | focus (page, API resource, subgoal, tree node); sub-agent | subgoals; rooms, receptacles; held object | imagined state nodes; real states | cases, subproblems, claims, intermediate results | files, functions, tests, patch hunks |
| Event | `Exploring Operation`; `Generated Node`, `Moving to Node`; `No Solution`; `Goal Reached`; `Current State` | subtask issued; action chunk; gripper open/close; re-query; user interjection; success detection | thought; tool call; observation; harness events (compaction, time-out, format error) | go, take, put, open; *Nothing happens*; think lines | world-model call; critic score; committed action; real observation | reasoning step; *wait*, *alternatively*; verification; final answer | view, search, edit, run, submit; linter rejection; test result |
| Relation | parent-child (identifier prefix); succession; backtracking match | preconditions; plan hierarchy; suspend-resume; agreement of reasoning and action | focus-stack push/pop; tool output to later use; delegation | precondition chains; take-put matching; revisits | tree parent-child; mismatch of prediction and observation; state written, then used | case opened-closed; conclusion to later citation; return to a claim | file opened-returned; view to edit; caller-callee |
| Episode | subtree exploration; backtracking run; sub-trajectory ω | subtask segment (dwell); grasp attempt; recovery | turn; sub-agent call; trial of a retry loop; failure streak | subgoal segment; exploration phase | imagined rollout; planning phase of one decision | analysis of a case; verification loop; continuation after the answer | localization; edit-test loop; failed-edit cascade |
| Quality | node depth; operator; polarity | subgoal level; skill type; chunk length | focus depth; tool type; observation length | subgoal depth; action type | imagined depth; action class; pruning reason | case depth; marker type | file depth; command type; error type |
| Resource | node references; tokens | control steps, chunks; reasoning tokens | tool calls; model calls; tokens | environment steps; planner calls | world-model calls; real steps | reasoning steps; tokens | turns; model calls; tokens |
| Outcome | `Goal Reached` of the reference (χ10) | subtask or task success | task completed; submission | goal condition satisfied | goal reached in imagination or reality | final answer correct | tests pass |

Columns: the first column is the kind of Table 2.1; each further column gives its instances in one domain.

### 2.4 Feature grammar

A structural coordinate has the form

- χ(z) = Agg over b in B(z) of q(b), or, for a rate,
- χ(z) = |B(z)| / |B′(z)|, set to 0 if B′(z) is empty.

Here B(z) is a set of bearers of one kind, filtered by kind, type or polarity (all node references, all consecutive pairs of node references, all negative-feedback events, all subgoal episodes); q is a quality of the bearers (depth, a type indicator, a length or a distance in events or tokens); and Agg is a count, maximum, mean, standard deviation, quantile, entropy or indicator. A coordinate is 0 when B(z) is empty, and a rate is 0 when its denominator set B′(z) is empty, as for χ1, ..., χ9 in Appendix A.1 (for χ5-χ7 this agrees with the denominator max(1, N − 1) used there).

Three construction rules follow from the properties of Section 3.

- **Growth.** A quality counted in events is O(T_max). Qualities that are themselves O(T_max), such as distances, are averaged or maximized, not summed. Every coordinate then grows at most linearly in T_max, so that it is bounded.
- **Unit.** Positions and distances are counted in events, so that T_k changes every coordinate by a bounded amount. Quantities counted in tokens serve as baselines and diagnostics.
- **Content.** q reads the kinds and types of events, not content values, so that the coordinate is structural.

Table 2.3 places the twelve coordinates of Φ in this grammar. They use six of the seven kinds: entities, events, the succession relation, qualities, resources (only as events, through χ4) and the outcome. Episodes, the dependency and reference relations, the split of the matching relation into its implicit and written-out parts, and resources other than events do not appear in χ1-χ12; the classes C (C5, C9, C10, C13-C15), M and N of the catalogue (Section 4) fill these gaps. For N ≥ 2, χ5 + χ6 + χ7 = 1, since the three indicators partition the N − 1 consecutive pairs.

**Table 2.3. The twelve coordinates of Φ (Appendix A.1) in the feature grammar: each is an aggregate of one quality over the bearers of one kind.**

| χ | Name | Bearers (kind) | Quality | Aggregate | Class in the catalogue |
|---|---|---|---|---|---|
| χ1 | maximum depth | nodes referred to by node references (entity) | depth | max | T; also N, as χ1 = N1 + 1 |
| χ2 | mean depth | same | depth | mean | T |
| χ3 | depth s.d. | same | depth | s.d. | T |
| χ4 | search volume | node-reference events (event; resource in events) | none | count | T |
| χ5 | deepening rate | consecutive pairs of node references (succession) | 1{depth increases} | rate | C |
| χ6 | backtracking rate | same | 1{depth decreases} | rate | C |
| χ7 | lateral rate | same | 1{depth unchanged} | rate | C |
| χ8 | prune count | negative-feedback events (event) | polarity | count | C |
| χ9 | pruning rate | negative-feedback events per generated node (event) | polarity | rate | C |
| χ10 | solution indicator | the reference trajectory (outcome) | goal reached | indicator | C (outcome) |
| χ11 | multiplicative bias | operator occurrences in actions and their restatements (event; substring counts over z, Appendix A.1) | type in {×, ÷} | count | I |
| χ12 | additive bias | same | type in {+, −} | count | I |

Columns: *χ* is the coordinate. *Name* is its name in Appendix A.1. *Bearers (kind)* is the set of objects it aggregates over, with their kind. *Quality* is what is read from each bearer. *Aggregate* is how the qualities are combined. *Class in the catalogue* is the class of Section 4 to which it belongs.

### 2.5 Why these seven kinds

Each kind answers a requirement of the bound or of its estimation (Table 2.4). A requirement that no kind answered would leave a term of Theorem 2 without a coordinate to act on; a kind that answered none would only add dimensions.

**Table 2.4. Why there are these seven kinds: each answers a requirement of the bound or of its estimation.**

| Requirement | Source in the paper | Kinds it calls for |
|---|---|---|
| coordinates are numbers, bounded, and placed in a metric space | Lemma 1 (Kantorovich duality) | qualities and their aggregation: every kind becomes a number through an aggregated quality |
| the expected risk under each measure is, up to ε, the integral of a function of Φ(z) | Assumption 1; Appendix A.2 (success is largely determined by the routing strategy; ε collects token identities) | events and relations (routing is a sequence of transitions), entities (on which it runs), outcome (whether the reference solves the problem) |
| structure is separated from its token realization, so that T_k is defined | Definition 2 (an inert node shifts later tokens by k) | resources in two units (events, tokens); the inert event as an event kind |
| one coordinate carries the nesting depth | Lemma 3; Assumption 5; Lemma 5 | the matching relation (push/pop) |
| lateral work is distinguished from nesting | Section 3 of the paper (independent lateral states can be processed in parallel; backtracking needs a last-in, first-out stack) | branching (a quality of entities); entities that are live at the same time |
| no coordinate reveals the evaluated outcome | definition of ℓ (Appendix A.4: the scorer's verdict on the model's output); Appendix J (every input computed from the reference trajectory) | outcome as a separate kind, with the reference's outcome kept apart from the label |
| failures, recoveries, phases and verification have duration and recurrence | event counts record how often, not for how long or how many in a row | episodes |

Columns: *Requirement* is what the bound or its estimation needs from the coordinates. *Source in the paper* is where the need arises. *Kinds it calls for* names the kinds that meet it.

### 2.6 Six classes and five meta-attributes

The catalogue of Section 4 groups coordinates into six classes, each built mainly from some of the kinds.

- **T, topology:** entities, their qualities (depth, branching) and the identity relation, aggregated as counts, maxima, means, standard deviations and graph invariants.
- **C, control flow:** events, the succession relation, episodes and outcomes, aggregated as transition rates, run counts, streak lengths and positions.
- **M, dependency and memory:** dependency and reference relations with their distances in events, working sets and lengths of dependency chains.
- **N, nesting:** the matching relation and restatement events, aggregated as Dyck height, its implicit part and the share of nesting moved out of the forward pass.
- **I, interaction:** action and observation events and the feedback relation, aggregated as type frequencies, entropy and feedback rates.
- **S, surface:** resources, as totals; token totals serve only as baselines and diagnostics.

Every coordinate carries five meta-attributes: source (task side or model side), outcome flag (whether it reveals success), unit (events or tokens), view (log or context) and aggregation.

## 3. Properties a coordinate must have

Appendix M.1 derives from the terms of Theorem 2 the properties that the coordinates of Φ_D must have. Whichever term of the bound is used, every coordinate is structural, computed on the task side, bounded and metric. When the positional-encoding term is used, the coordinates change under T_k by a bounded amount and one of them counts the inserted event; when the depth floor is used, one coordinate counts the nesting that one forward pass must resolve. The set as a whole is small and predicts success. For each property, *Requirement* states the check, *Reason* gives its source in the paper and *Without it* says what goes wrong if it fails.

**Structural.**
- *Requirement:* the coordinate depends only on the kinds and types of events, the relations among entities and positions counted in events. Replacing content values (operands, object names, file contents) while keeping the structure leaves it unchanged.
- *Reason:* Appendix A.2: ε collects the token identities, such as the operand values of a Countdown problem.
- *Without it:* the coordinate is one of the finer descriptors of Appendix I, which lower ε at the price of the sample complexity of W1.

**Computed on the task side.**
- *Requirement:* the coordinate is a deterministic function of the reference trajectory z (the trajectory of a reference solver for the same problem), including x_init^(z), and reads neither the model output f(x_init^(z)) nor its score. The outcome of the reference itself (χ10) is allowed.
- *Reason:* the definition of ℓ (Appendix A.4). ℓ is the scorer's verdict on f(x_init^(z)), so a coordinate computed from the model's output can encode the label. Accordingly, Appendix J computes every input from the reference trajectory.
- *Without it:* ν_S and ν_T are not task-side measures, and the sufficiency check measures the scorer. Model-side trajectories, with the coordinates that reveal the outcome removed, serve run-time monitoring, as in Experiment 3 (Section 8.7).

**Bounded, and at most linear in T_max.**
- *Requirement:* the coordinate is finite on every z in Z (empty bearer sets and zero denominators give 0). Counts, depths and distances in events are at most c·T_max; rates, indicators and entropies are bounded by constants.
- *Reason:* Appendices A.1 and A.3 (X is bounded, ν has a finite first moment) and Section 2.2 of the paper (the diameter of X is Θ(T_max), and W1 ≤ O(T_max)).
- *Without it:* sums of distances or products of counts grow as T_max², and statements made in units of T_max, such as the normalized shift W1/T_max of Corollary 1, no longer apply.

**Metric.**
- *Requirement:* differences are meaningful (counts, rates, depths, distances, entropies), so that a unit step along a coordinate means the same everywhere. A categorical attribute enters through per-category counts, as the operators do in χ11 and χ12, never through an arbitrary integer code. The weights of a weighted metric ‖W(u − v)‖₂ and the marker sets of the adapter follow one rule, applied identically to every compared measure, as the joint standardization on the pooled sample of Appendix G is.
- *Reason:* Section 2.2 of the paper and Lemma 1, whose K_f is taken in the metric used.
- *Without it:* K_f depends on an arbitrary numbering, and the Lipschitz statement loses its content. Every function on the finite set Φ_D(Z) is Lipschitz; the metric property makes K_f a statement about structure, since a unit step along each coordinate has a fixed meaning (one event, one level, one percentage point).

**Bounded change under T_k (for the positional-encoding term).**
- *Requirement:* the change |χ(T_k z) − χ(z)| is bounded independently of k and T_max, and Φ_D contains a coordinate that counts the inserted inert event, as χ4 does under Definition 2, so that 1 ≤ δ ≤ C_δ.
- *Reason:* Definition 2, Appendix C.1 and the proof of Theorem 1, which bounds K_{f,APE} from below by γ/δ.
- *Without it:* either δ grows with k (coordinates counted in tokens), so that γ/δ is no longer independent of T_max; or δ vanishes. In the second case Φ_D does not see the insertion, Assumption 2 cannot hold between Φ_D(z) and Φ_D(T_k z), and a risk difference caused by the insertion can only enter ε. Table 3.1 classifies coordinates by their change under T_k.

**A depth coordinate (for the depth floor).**
- *Requirement:* one coordinate χ_nest is the nesting of the push/pop word obtained from the adapter's events through a homomorphism ψ_D (the analogue of Lemma 3), or another notion of depth such as the composition depth M5. Assumption 5 is adopted for it. It counts only nesting that the model resolves within one forward pass, that is, entries that are neither written into its context nor held by an outside program (N2), and it is computed on the context view.
- *Reason:* Lemmas 3 and 5, Assumption 5, Corollary 2, and the implicit routing evaluations "not serialized in the output" of Section 3 of the paper.
- *Without it:* P_implicit is defined on a quantity that Assumption 5 does not concern.

**Few coordinates (the set as a whole).**
- *Requirement:* at most 16 coordinates per domain, a budget chosen close to the d_Φ = 12 of the paper. The within-measure baseline of Appendix G is reported next to every estimated shift, and the one-dimensional W1 of each coordinate is used to attribute the shift to coordinates. The coordinates of one class are aggregated into few coordinates. Of coordinates that are affine functions of one another (χ5 + χ6 + χ7 = 1; T1 and N1 on Countdown), one may be dropped when the budget is tight, since a dependent coordinate adds no information but changes the weight that the distance on X gives to the directions involved.
- *Reason:* Appendix I: the number of samples needed to estimate W1 grows quickly with the dimension (Weed and Bach, 2019).
- *Without it:* finite-sample bias dominates the estimated shift.

**Sufficiency (the set as a whole).**
- *Requirement:* the coordinate set is checked as in Appendix J: out-of-fold ROC-AUC and McFadden pseudo-R² of the same classifier, compared with a length baseline (LEN, the token length of z) and with the problem alone (PROB). Where several trajectories share one task, the folds are also grouped by task.
- *Reason:* Assumption 1. As in Appendix J, the check measures how well the coordinates predict success, which is evidence on the size of ε.
- *Without it:* ε may be large, the shift and depth terms are then loose, and W1 remains a description of the structural shift.

The width term needs no per-coordinate check once Assumption 4 is restated on R^{d_Φ}, as in point 1 of Section 1.

**Table 3.1. Change of a coordinate under T_k.** T_k inserts one structurally inert event of k tokens at the current depth, immediately before a sub-trajectory ω (Definition 2), so that χ1 is unchanged (Appendix C.1).

| Class of change | Typical coordinates | Change | Bounded change under T_k |
|---|---|---|---|
| invariant | maximum depth, nesting depth, negative-feedback counts, outcome indicators, composition depth; counts of types that the inserted event does not have | 0 | yes |
| count +1 | volume χ4, events, environment steps, the count of the type of an inserted identity action; distances in events that span the insertion | at most 1 | yes (marked P in the catalogue if it counts the inserted event) |
| rates and means | χ2, χ3, χ5-χ7, χ9; relative position of the first success | O(1/N) | yes |
| counted in tokens | token length, token positions, distances in tokens, spread of observation lengths | proportional to k | no: baseline or diagnostic only |

Columns: *Class of change* groups coordinates by how they respond to the insertion. *Typical coordinates* lists members. *Change* is the size of |χ(T_k z) − χ(z)|, with N the number of node references. *Bounded change under T_k* says whether the coordinates of the class have this property of Section 3.

### 3.1 Translation in practice

All twelve coordinates are counted in events: under Definition 2, χ4 gains one and the others change by O(1/N) or not at all (Appendix C.1). Quantities counted in tokens describe what a position-dependent kernel sees, namely the same structure at different absolute positions. They measure how often and how far T_k-type insertions occur in a domain (S5) instead of entering Φ_D. A longer tool output or reasoning step shifts later tokens without inserting an event; Φ_D does not register it, and S5 measures such shifts.

Definition 2 and Lemma 2 concern pairs of tokens inside the shifted sub-trajectory ω. Pairs that span the insertion change their relative distance by k under every positional encoding. Examples are attention from an action block to the image tokens at the start of a VLA context, or from a repair step to the task statement at the top of an agent context. Absolute clocks written into the content ("step 17 of 50", a remaining budget) change later tokens when a turn is inserted, and truncation or a sliding window deletes earlier content while re-indexing the rest. None of these is a translation in the sense of Definition 2.

### 3.2 Adding coordinates

Let Φ′ = (Φ, φ) extend Φ by further coordinates φ, and let P be the coordinate projection that drops them, so that ν = P#ν′. Two elementary facts hold.

1. W1(ν_S, ν_T) ≤ W1(ν′_S, ν′_T), because P is 1-Lipschitz and maps couplings of (ν′_S, ν′_T) to couplings of (ν_S, ν_T). With per-coordinate standardization on the pooled sample, this holds for the standardized coordinates as well.
2. If L_f satisfies Assumption 1 on X with residual ε and Lipschitz constant K, then L_f∘P satisfies it on X′ with the same ε and a Lipschitz constant at most K, because ∫ L_f∘P dν′ = ∫ L_f dν and ‖Pu − Pv‖ ≤ ‖u − v‖.

Adding coordinates therefore never raises the attainable ε and never lowers W1. It tightens K_f·W1 + 2ε only when a risk function on X′ lowers 2ε by more than K_f·W1 grows, while W1 becomes harder to estimate. This is the balance stated in Appendix I. The budget of few coordinates and the sufficiency check of Section 3 are its two sides, and Experiment 1 (Section 7) measures both on Countdown.

### 3.3 Two notions of depth, and nesting moved out of the forward pass

*Nesting depth* (N1, N2) is the Dyck height of the push/pop word; Lemma 3 motivates Assumption 5 for it. *Composition depth* (M5) is the number of transitions composed between two written-out complete states, such as a query for the state of an environment after k actions without intermediate states. Writing out intermediate steps lifts the constant-depth limit (Li et al., 2024). Adopting Assumption 5 for M5 is a separate domain assumption, motivated by state tracking rather than by Lemma 3, and shallow shortcut solutions to state tracking (Liu et al., 2023) make the linear threshold α_circ·L a modelling choice to be examined in that domain.

Lemma 5 concerns nesting that can be inferred from the context but must be resolved within one forward pass. When the stack is absent from the input altogether, as for a policy restarted at every step without memory, a failure is due to missing information, which neither depth nor width supplies.

Restated stacks, task lists, sub-agents, written-out intermediate states and outside search programs, such as a tree search that calls the model at each node (Yao et al., 2023), move nesting out of the forward pass, either into the context or outside the model. This lowers N2 and raises N3. With Assumption 5 adopted for N2, the depth floor concerns the nesting that remains. The question that Section 7 of the paper leaves to future theory, how inference-time search reshapes ν_T, can be posed in these terms: a program that builds the model's contexts changes what each forward pass reads, and hence N2, while the log keeps the full tree. Experiments 7 and 8 (Section 10) measure how success changes when subtrees deeper than one call resolves are sent to separate calls.

On Countdown, the `Current State` line writes out the state of every node and the node identifier writes out the current path, so M5 = 1. Which alternatives of the suspended ancestors remain untried is not written out; this is the "most recently suspended, non-adjacent sibling" of Section 3 of the paper. Taking these untried alternatives as the resumption information of a search node, N2 = N1 = χ1 − 1, which is why the depth floor can be stated on χ1.

### 3.4 Width

No coordinate enters C_width/√m. Through Assumption 4(iii), C_width depends on how well width-m hypotheses can approximate L\* on (X, ν_T), and therefore on Φ_D as a whole. The coordinates marked w in the catalogue (T7, C3, M3, M6) measure the lateral work that Section 3 of the paper contrasts with nesting.

## 4. Catalogue of 48 candidate coordinates

This is the catalogue of candidate coordinates for Countdown and six other trajectory domains to which Appendix M refers. Experiment 1 adds its further Countdown coordinates in the order of these classes (Section 7), and the projections of Experiment 2 are assembled from it (Section 8.3). It lists 48 coordinates in the six classes of Section 2.6: T 10, C 15, M 6, N 5, I 7, S 5. Each is defined as a function of the reference trajectory in the notation of Section 2.2, and each is structural, computed on the task side, bounded and metric (Section 3), except the token-counted baselines S4 and S5 (marked B), which depend on the token realization. For each class, a first table gives the definitions and a second the extraction in each of the six other trajectory domains, built from the adapter primitives of Table 4.1. A rate is 0 when its denominator is 0, and every coordinate is 0 when its bearer set is empty. On Countdown, M5 = 1, N2 = N1 and N3 = 0.

**How to read the definition tables.**
- *ID*: the identifier used throughout this document. A ° after the ID marks an outcome of the reference: it is allowed on task-side trajectories and excluded on model-side ones.
- *Coordinate*: its name; the coordinate of the paper it equals, if any, in parentheses.
- *Definition*: the coordinate as a function of the reference trajectory z (D_i: depth of the i-th node reference; N: number of node references; n: number of events).
- *Kind*: written as bearer kind · quality · aggregate, in the terms of Section 2.
- *Terms*: the terms of Theorem 2 the coordinate takes part in. W: the shift term K_f·W1, which every coordinate enters through the distance on X. E: the sufficiency residual ε, which every coordinate can lower. E\*: a coordinate that records a source of difficulty by construction (size, dead ends, recovery, randomness), a first candidate for the sufficiency check. P: counts the inserted inert event and so carries δ of Definition 2. D: a candidate for χ_nest (Assumption 5 to be adopted). w: breadth, in the sense of Section 3.4. B: a token-counted baseline or diagnostic, excluded from Φ_D when the positional-encoding term is used.
- *Change under T_k*: the class of Table 3.1 (0; +1; ≤1, at most one; O(1/N); k, exactly k; proportional to k).

**How to read the extraction tables.** Each row is one coordinate (or a group of coordinates with a shared definition) of the class; each column gives how it is read from trajectories of one domain. "n/a" means that the kind is absent from the domain.

**Table 4.1. Adapter primitives by domain.** Every extraction in Tables 4.2 to 4.13 is built from these primitives.

| Primitive | Countdown | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|---|
| entity v | search node | subgoal (precondition, recovery and interjected subgoals are children of the subgoal they serve); object | focus: page, API resource, subgoal, tree node | subgoal; location, receptacle | imagined state node | case, subproblem | file (focus); function |
| node reference | `Generated Node`, `Moving to Node` | subtask activation | focus of each turn | subgoal activation | each world-model call | case of each step | file of each turn |
| push | expand a child | open a precondition, recovery or interjected subgoal | enter a new focus; open a subgoal; spawn a sub-agent | open a subgoal | extend an imagined branch | open a case (*case 1*, *suppose*, *first we need*) | open a new file; enter a callee |
| pop | move to a shallower node | subgoal completed or abandoned; resume the suspended one | return to a focus on the stack | subgoal completed or abandoned | return to a shallower imagined node | close a case (*so*, *therefore*, *back to*) | return to a file on the stack |
| lateral move | move at the same depth | next subtask at the same level | act within the current focus | next subgoal at the same level | evaluate a sibling action | next step at the same level | operate within the current file |
| action, type a | `Exploring Operation`; operator | primitive or skill (grasp, place, open, close, wipe, gripper) | tool name | go, take, put, open, toggle | proposed or committed action class | none in plain text; tool calls in tool-integrated reasoning | view, search, edit, run, submit |
| negative feedback | `No Solution` | failed grasp; abandoned subtask; detector reports failure | tool error; invalid action; empty result | *Nothing happens*; invalid action; exploration subgoal | branch pruned; mismatch of prediction and observation | abandoned line; re-check finds an error | linter rejection; failed edit; failing test; traceback |
| positive feedback | `Goal Reached` | subtask or task success | goal achieved; submission | goal condition satisfied | goal reached in imagination or reality | final answer | tests pass |
| restatement | `Current State`; path in the node identifier | rewritten plan text | to-do or plan tool output; restated task | think line restating the plan | written-out imagined state | restated goal (*we still need to show*) | restated issue or plan |
| external event | none | user interjection; detector message | compaction; time-out; format error; system reminder | human interjection; environment reset | re-initialization from a real observation | none | time-out; compaction; automatic submission |
| inert event (T_k) | node of k inert tokens, for example `x=x` (Appendix C.1) | wait or no-op step | neutral message; call whose output adds no entity | wait or no-op, observation unchanged | empty step, state unchanged | filler paragraph without a new claim | view or listing call that adds no entity |
| task-side reference | SoS generator | demonstrations with plan or subtask annotation; scripted or oracle-planner episodes | runs of a fixed reference agent; trees of an outside search program | oracle or expert planner (subgoal stack, PDDL expert, gold action sequence) | reference planner on the true simulator | traces of a reference solver or teacher on the same problem | reference run, gold patch, failing tests |

Columns: *Primitive* is one object that the adapter must recognize (Section 2.2); each further column says what that object is in one domain.

### 4.1 Class T: topology (shape of the explored structure)

**Table 4.2. Class T: definitions.**

| ID | Coordinate | Definition | Kind | Terms | Change under T_k |
|---|---|---|---|---|---|
| T1 | Maximum depth (χ1) | max over i of D_i | entity · depth · max | W, E, D | 0 |
| T2 | Mean depth (χ2) | (1/N) Σ_i D_i | entity · depth · mean | W, E | O(1/N) |
| T3 | Depth s.d. (χ3) | standard deviation of the D_i | entity · depth · s.d. | W, E | O(1/N) |
| T4 | Volume (χ4) | N, the number of node-reference events | event · count (resource in events) | W, E, P | +1 |
| T5 | Distinct states | U, the number of distinct entities among v_(1), ..., v_(N) | entity · count | W, E | ≤1 |
| T6 | Revisit ratio | (N − U)/N | identity relation · rate | W, E\* | O(1/N) |
| T7 | Mean branching | mean over internal nodes of the number of visited children | entity · branching · mean | W, E, w | O(1/N) |
| T8 | Cycle rank | number of edges − number of nodes + c of the undirected state-transition graph without multiple edges (c: number of connected components) | relation graph · invariant | W, E\* | ≤1 |
| T9 | Initial-state size | number of entities in x_init^(z) | entity · count (initial state) | W, E\* | 0 |
| T10° | Reference solution size | depth of the goal reached by the reference, that is, its number of solution steps (0 if none) | outcome · quality | W, E\* | 0 |

**Table 4.3. Class T: extraction by domain.**

| ID | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|
| T1-T3 | max, mean, s.d. of subgoal-tree depth | max, mean, s.d. of focus-stack depth | max, mean, s.d. of subgoal-stack depth (precondition chain) | max, mean, s.d. of imagined depth (rollout horizon actually used) | max, mean, s.d. of case or subproblem depth | max, mean, s.d. of file-focus stack depth |
| T4 | subtask segments (subgoal activations) | tool calls | subgoal references (not environment steps) | world-model calls (imagined nodes) | reasoning steps | turns |
| T5 | distinct subtasks | distinct foci | distinct locations, receptacles | distinct imagined states (transpositions merged) | distinct cases or claims | distinct files viewed |
| T6 | share of re-issued subtasks | focus revisit ratio | location revisit ratio | share of transpositions in imagination | share of returns to an earlier claim | file reopen ratio |
| T7 | subtasks per subgoal | child foci per focus | sub-subgoals per subgoal | actions evaluated per imagined node (beam or tree width) | subcases per case | files opened from a file |
| T8 | cycle rank of the subtask graph (open-close-open loops) | cycle rank of the page or resource graph | cycle rank of the location graph | cycle rank of the imagined-state graph and the real-state sequence | cycle rank of the argument-reference graph | cycle rank of the file-transition graph |
| T9 | task-relevant objects | entities given by the task (form fields, to-do items) | task-relevant objects and rooms | entities of the initial state (for example blocks) | given objects and conditions | failing tests |
| T10° | subtasks of the reference plan | length of the reference run's successful path | oracle plan length | optimal plan length | steps of the reference's final derivation (abandoned branches removed) | files or hunks changed by the gold patch |

### 4.2 Class C: control flow (how the walk moves)

**Table 4.4. Class C: definitions.** M = max(1, N − 1) is the number of consecutive pairs of node references.

| ID | Coordinate | Definition | Kind | Terms | Change under T_k |
|---|---|---|---|---|---|
| C1 | Deepening rate (χ5) | (1/M) Σ_i 1{D_{i+1} > D_i} | succession · rate | W, E | O(1/N) |
| C2 | Backtracking rate (χ6) | (1/M) Σ_i 1{D_{i+1} < D_i}; the pops of the push/pop word | succession · rate | W, E | O(1/N) |
| C3 | Lateral rate (χ7) | (1/M) Σ_i 1{D_{i+1} = D_i} | succession · rate | W, E, w | O(1/N) |
| C4 | Mean pop size | mean of D_i − D_{i+1} over pops | matching · quality · mean | W, E | O(1/N) |
| C5 | Backtracking runs | number of maximal runs of consecutive pops | episode · count | W, E | ≤1 |
| C6 | Replans | number of plan-revision events (defined by the adapter) | event · count | W, E\* | 0 |
| C7 | Dead ends (χ8) | number of negative-feedback events that close a branch | event (negative feedback) · count | W, E\* | 0 |
| C8 | Dead-end rate (χ9) | C7 divided by the number of generated nodes or actions | event · rate | W, E\* | O(1/N) |
| C9 | Longest failure streak | maximum number of negative-feedback events with no progress event (new entity or positive feedback) between them | episode · max | W, E\* | 0 |
| C10 | Recovery latency | mean number of events from a negative-feedback event to the next progress event | relation (failure to recovery) · distance · mean | W, E\* | ≤1 |
| C11° | Reference success (χ10) | 1{the reference reaches the goal} | outcome · indicator | W, E | 0 |
| C12° | Position of first success | index of the first positive-feedback event divided by n | outcome · position (events) | W, E | O(1/N) |
| C13° | Continuation after success | number of events after the first positive-feedback event | episode · count | W, E | ≤1 |
| C14 | Verification episodes | number of episodes that re-check an earlier result, including test runs before submission | episode · count | W, E\* | 0 |
| C15 | Exploration phase | number of events before the first state-changing action | episode · length | W, E | ≤1 |

**Table 4.5. Class C: extraction by domain.**

| ID | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|
| C1-C3 | shares of transitions to a deeper subgoal / back to a suspended one / to the next sibling | push / pop / stay on the focus stack | push / pop / next sibling on the subgoal stack | extend a branch / return to a shallower node / evaluate a sibling action | open a case / close it / next step at the same level | open a new file / return to an earlier one / act within the current file |
| C4 | levels returned per completed subtask | stack levels per pop | levels returned per completed subgoal | levels crossed per return in the tree | levels crossed per return to an earlier argument | stack levels per return to an earlier file |
| C5 | runs of returns to higher subgoals | runs of pops | runs of pops | runs of retreats | retreat runs opened by *wait*, *alternatively* | runs of returns to earlier files |
| C6 | changes of the plan text | plan or to-do tool updates | replans after a failure | replans after a mismatch of prediction and observation | changes of approach (*let me reconsider*) | reverted edits followed by a new approach |
| C7 | failed grasps, abandoned subtasks | tool errors, invalid actions, empty results | *Nothing happens*, invalid actions, exploration subgoals | branches pruned as infeasible or of low value | abandoned lines, computations found wrong on re-check | linter rejections, failed edits, failing tests, tracebacks |
| C8 | C7 per subtask | C7 per tool call | C7 per action | C7 per imagined node | C7 per reasoning step | C7 per edit |
| C9 | longest run of failed grasps | consecutive tool errors | consecutive failed actions | consecutive pruned branches | consecutive refuted attempts | longest failed-edit cascade |
| C10 | steps from a failure to renewed progress | calls from an error to the next successful call | steps from a failure to the next subgoal progress | steps from a mismatch to resumed progress | steps from a found error to the next established conclusion | turns from a failed to the next successful edit |
| C11° | demonstration success (dropped when constant) | the reference run completes the task | the oracle solves the instance (dropped when constant) | the reference planner reaches the goal in imagination | the reference answer is correct | the reference run passes the tests |
| C12° | first detected task success | first completion of the goal | first satisfied goal condition | first imagined goal (in world-model calls) | first final answer | first run with all tests passing |
| C13° | steps after success | calls after completion | steps after the goal condition holds | world-model calls after the goal | reasoning steps after the first answer | turns after the tests pass |
| C14 | success-detection queries | read-backs before submission | state checks (look, examine) | re-simulations of the chosen plan | *let me verify*, substitution checks | test runs |
| C15 | steps before the first grasp | calls before the first write | steps before the first object interaction | world-model calls before the first committed action | steps before the first intermediate conclusion | turns before the first edit (localization) |

### 4.3 Class M: dependency and memory (what must be carried across distance)

**Table 4.6. Class M: definitions.** first(u) and last(u) are the indices of the first and last events at entity u.

| ID | Coordinate | Definition | Kind | Terms | Change under T_k |
|---|---|---|---|---|---|
| M1 | Mean return distance | for each resumption of a suspended entity u: index of the resuming event minus index of the previous event at u; mean | reference · distance (events) · mean | W, E | ≤1 |
| M2 | Maximum return distance | as M1, maximum | reference · distance · max | W, E | ≤1 |
| M3 | Live contexts | the largest number, over event indices i, of entities u with first(u) < i < last(u) | entity · count (working set) | W, E, w | ≤1 |
| M4 | Mean recall distance | for each entity produced and used later: index of its first use minus index of its first appearance; mean | dependency · distance · mean | W, E\* | ≤1 |
| M5 | Composition depth | maximum number of state-changing events between two consecutive events that write out the complete current state | dependency chain · max | W, E, D | 0 |
| M6 | Tracked entities | number of distinct entities whose state is changed | entity · count | W, E, w | 0 |

**Table 4.7. Class M: extraction by domain.**

| ID | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|
| M1, M2 | steps from suspending a subtask to resuming it | calls from leaving a focus to returning to it | steps from suspending a subgoal to resuming it | world-model calls from leaving an imagined node to returning to it | steps from suspending a case to returning to it | turns from leaving a file to returning to it |
| M3 | subgoals suspended at the same time | open subgoals or to-do items | suspended subgoals | size of the expansion frontier | open cases | files that will be revisited |
| M4 | steps from first seeing an object to first grasping it | calls from a tool output to the first use of its content | steps from first observing an object to first taking it | calls from writing a state to its use in a later prediction | steps from deriving an intermediate result to citing it | turns from first viewing a file to first editing it |
| M5 | open-loop actions between two re-observations (chunk length) | state-changing calls between two returned observations | actions between two observations | transitions composed between two written-out imagined states | derivation steps between two written-out complete states | edits between two views or runs |
| M6 | distinct objects manipulated | distinct resources modified | objects whose location or state changes | entities that change state (blocks moved) | objects defined or assigned | files edited |

### 4.4 Class N: nesting (the Dyck view)

**Table 4.8. Class N: definitions.** h is the nesting depth (N1) and h_impl the implicit nesting (N2). ξ (N3) is the share of the nesting moved out of the forward pass, that is, written into the context or held outside the model.

| ID | Coordinate | Definition | Kind | Terms | Change under T_k |
|---|---|---|---|---|---|
| N1 | Nesting depth h | h = maximum over prefixes of w(z) of (number of pushes − number of pops) | matching · max | W, E, D | 0 |
| N2 | Implicit nesting h_impl | maximum over prefixes of the number of open entries whose resumption information (defined by the adapter) has not been written out by a restatement since they were suspended | matching + restatement · max | W, E, D | 0 |
| N3 | Share moved out of the forward pass ξ | (h − h_impl)/h | quality · ratio | W, E, D (diagnostic) | 0 |
| N4 | Maximum pop size | maximum of D_i − D_{i+1} over pops | matching · max | W, E | 0 |
| N5 | Interruptions | number of times an open episode is suspended by an unrelated push (interjection, recovery inside a subtask, detour) | relation between episodes · count | W, E\* | 0 |

**Table 4.9. Class N: extraction by domain.**

| ID | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|
| N1 | depth h of the stack of pending subgoals from preconditions, recoveries and interjections (not plan length or horizon) | nesting height of the focus or subgoal stack | nesting height of the subgoal stack (precondition chains) | goal-chain height of the reference solution; nesting of the imagination tree | nesting height of cases | nesting height of the file or call stack |
| N2 | suspended subgoals not written in reasoning or memory (equal to N1 when reasoning is dropped or latent) | suspended items absent from to-do or plan output and task restatements; entries held by an outside program count as moved out | suspended subgoals not restated in thought lines | suspended branches not written out; 0 when an outside search program holds the tree | suspended cases not restated | suspended files or subtasks not written into a plan |
| N3 | from N1, N2 | from N1, N2 | from N1, N2 | from N1, N2 | from N1, N2 | from N1, N2 |
| N4 | most levels returned after a subtask completes | most stack levels per pop | most levels returned after a subgoal completes | most imagination levels per retreat | most levels per return | most stack levels per return to an earlier file |
| N5 | interjections or recovery subtasks suspending the current subtask | detours to an unrelated subtask, including user interjections | subgoal interruptions (recovery, interjection) | real observations interrupting an imagined plan | side lemmas inserted | detours to unrelated errors (environment, dependencies) |

### 4.5 Class I: interaction with an environment

**Table 4.10. Class I: definitions.**

| ID | Coordinate | Definition | Kind | Terms | Change under T_k |
|---|---|---|---|---|---|
| I1 | Action-group counts (χ11, χ12) | for a fixed partition of action types into at most three groups, the number of actions in each group (χ11 and χ12 count operator symbols in actions and their restatements, as substring counts over z, Appendix A.1) | event · type · count | W, E\* | ≤1 (0 on χ1-χ12) |
| I2 | Action entropy | −Σ_a q_a log q_a, where q_a is the frequency of action type a | quality distribution · entropy | W, E | Õ(1/N) |
| I3 | Negative feedback per action | number of negative-feedback events (obs−) divided by number of actions (act) | feedback · rate | W, E\* | O(1/N) |
| I4 | Coupling of action and observation | share of actions followed by an observation before the next action, or mean parallel calls per turn | feedback · rate | W, E | O(1/N) |
| I5 | Environment stochasticity | among repeated (state, action) pairs, the share whose observations differ | feedback + identity · rate | W, E (signals risk beyond structure) | O(1/N) |
| I6 | Reasoning density | share of actions preceded by a new reasoning event since the previous action | event · rate | W, E | O(1/N) |
| I7 | External events | number of external events (interjections, context compaction, time-outs, detector messages) | event · count | W, E, P | ≤1 |

**Table 4.11. Class I: extraction by domain.**

| ID | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|
| I1 | skill groups (pick-place / open-close / wipe) or translation, rotation, gripper primitives | read-type / write-type tool calls | navigation / manipulation actions | action classes (for example stack-unstack / pick-put) | n/a (tool groups in tool-integrated reasoning) | edit / search and view / run |
| I2 | entropy of skill types | entropy of tool types | entropy of action types | entropy of action classes | n/a | entropy of command types |
| I3 | failed grasps per grasp | tool errors per call | failed actions per action | mismatches of prediction and observation per real action | n/a | failed commands per command |
| I4 | executed share of a chunk before re-query | parallel calls per turn | alternation of observation and action | imagined calls per real action | n/a | commands per turn |
| I5 | retried subtasks with different outcomes | identical calls with different results | identical (location, action) pairs with different observations | identical (real state, action) pairs with different observations | n/a | repeated test commands with different results |
| I6 | control steps with fresh reasoning | turns with a thought | actions with a think line | real decisions preceded by imagination | n/a | turns with a thought |
| I7 | user interjections, success-detector messages | context compactions, time-outs, format errors, system reminders | human interjections, environment resets | re-initializations from a real observation | n/a | time-outs, compactions, automatic submissions |

### 4.6 Class S: surface and resources

**Table 4.12. Class S: definitions.**

| ID | Coordinate | Definition | Kind | Terms | Change under T_k |
|---|---|---|---|---|---|
| S1 | Events n | total number of events | resource (events) | W, E, P | +1 |
| S2 | Environment steps H | number of environment (control) steps | resource (steps) | W, E, P | ≤1 |
| S3 | Model calls | total number of model calls, including world-model and planner calls | resource (calls) | W, E | 0 |
| S4 | Token length | length of z in tokens | resource (tokens) | B (LEN baseline) | k |
| S5 | Positional spread | s.d. of the starting token positions of the occurrences of a recurring sub-trajectory type; s.d. of observation lengths in tokens | quality (position, length in tokens) | B (exposure to T_k-type insertions) | proportional to k |

**Table 4.13. Class S: extraction by domain.**

| ID | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|
| S1 | control steps or chunks | events | events | world-model calls plus real steps | reasoning steps | events |
| S2 | control steps | n/a (use S3) | environment steps | committed real steps | n/a | n/a (use S3) |
| S3 | high-level planner queries | model calls, including evaluation calls of an outside search program | planner calls | policy, world-model and critic calls | samples (1 for one generation) | model calls |
| S4 | reasoning and action tokens | tokens | tokens | tokens (token-based world models) | tokens | tokens |
| S5 | start position of the action block in the per-step context when reasoning length varies; start timestep of a sub-behaviour under absolute timestep embeddings | start of a recurring motif (repair after a failed call); observation lengths | start timestep of a sub-behaviour | context position of a local transition segment | start of recurring verification episodes | start of failed-edit motifs; tool-output lengths |

## 5. Domain mappings and conditions per domain

Table 5.1 states, for Countdown and the six other trajectory domains, what the trajectory is, where its task-side reference comes from, what a single forward pass reads and which kinds are absent. Table 5.2 recommends a projection of at most 16 coordinates per domain, and Table 5.3 states which terms of Theorem 2 apply.

The data-side terms (W1 and the sufficiency check) need task-side references and per-instance outcomes of the evaluated system, but not its weights or architecture. The architecture-side terms need an autoregressive Transformer with known L, m and positional encoding; the positional-encoding term further needs Assumption 2 and T_k-type insertions in the domain, and the depth floor concerns nesting resolved implicitly within one forward pass. For closed-weight models only the data side applies.

**Table 5.1. Domain overview.**

| Domain | Trajectory z (log) | Task-side reference | What one forward pass reads (context) | Absent or degenerate |
|---|---|---|---|---|
| Countdown | SoS trajectory | SoS generator (BFS, DFS, MIXED) | problem and generated prefix (log = context) | none; episodes do not enter χ1-χ12 |
| VLA | closed-loop episode: subtask stream, reasoning fields, action chunks, gripper events | demonstrations with plan or subtask annotation; scripted or oracle-planner episodes | one control step (per-step policy) or a history window (policies with history or memory) | no events inside a continuous action chunk; a template regenerated at every step is constant per step, so Φ is defined per episode |
| Tool-use agent | log of thoughts, tool calls, observations and harness events | runs of a fixed reference agent; trees of an outside search program | the context c_t of each call, built by the harness from the log | multi-agent systems: no single context; Φ is defined on the message graph |
| Embodied agent | z_ref, the reference subgoal trace; the agent's z_prefix and z_full are model side | oracle or expert planner on the same instance | the input of each planner or policy call | low-level continuous control has no discrete structure |
| World-model planner | imagination loop (tree of each decision) and reality loop (commit, observe, replan) | reference planner run with the true simulator | each forward pass of the world model or policy | planners without a tree: T, C, N reduce to horizon, iterations, replans; video and continuous-latent models: no discrete events |
| Long CoT | reasoning text split into steps | traces of a reference solver or teacher on the same problem | problem and generated reasoning prefix (log = context) | no environment: class I absent (except tool-integrated reasoning) |
| Code agent | log of view, search, edit, run, submit and their observations | reference run, gold patch, failing tests | the context of each call | none |

Columns: *Domain*. *Trajectory z (log)* is what is analysed. *Task-side reference* is where the reference trajectories, on which every coordinate is computed, come from. *What one forward pass reads* is the context view (Section 2.1). *Absent or degenerate* names the kinds that are missing or constant in the domain.

**Table 5.2. Recommended projections Φ_D (d_Φ ≤ 16), with the coordinate that carries the depth floor.** "I1 ×2" means I1 with two action groups, which gives two coordinates. Each recommendation is a starting point, to be checked in the domain with the within-measure baseline and the sufficiency check of Section 3. The depth floor is a statement on X, so when χ_nest is not among the listed coordinates it replaces one of them (T2 or T3) whenever the depth term is analysed.

| Domain | Φ_D | d_Φ | χ_nest | Notes |
|---|---|---|---|---|
| Countdown (the paper) | T1, T2, T3, T4, C1, C2, C3, C7, C8, C11, I1 ×2 | 12 | T1 (= N1 + 1); N2 = N1 | the coordinates χ1-χ12; I1 with the groups {×, ÷} and {+, −} |
| VLA | N1, T2, T4, C1, C2, C3, C6, C7, C8, T6, T8, N5, I1 ×2, S2 | 15 | N1 (pending subgoals h); N2 when reasoning is partly written | S2 separates horizon from nesting; N5 records interjections; optional 16th coordinate N2 |
| Tool-use agent (harness) | T1, T2, T3, T4, C1, C2, C3, C7, C8, C9, C10, T6, I1 ×2, M1, M3 | 16 | N2 on the context view (replaces T3 when the depth term is analysed) | failure dynamics (C9, C10) and memory (M1, M3); I1 groups read / write |
| Embodied agent | T1, T2, T3, T4, C1, C2, C3, C6, C7, C8, T6, T10, M4, I1 ×2, S2 | 16 | N2 of the subgoal stack (= N1 when the plan is not restated) | T10 and M4 for plan length and object recall; I1 groups navigation / manipulation |
| World-model planner | T1, T2, T3, T4, C1, C2, C3, C6, C7, C8, C11, M5, S2, T8, N1, I1 | 16 | N1 (goal chain) and M5 (composition) | T and C on the imagination tree, aggregated over decisions; S2 real horizon; C11 dropped when constant |
| Long CoT | N1, N2, T2, T4, C1, C2, C3, C5, C7, C8, C12, C13, C14, M4, T6 | 15 | N2 (cases not restated) | C12, C13 on reference traces only; marker sets fixed in the adapter |
| Code agent | T1, T2, T4, T5, T6, C1, C2, C7, C8, C9, C14, C15, M4, M6, I1, T10 | 16 | N2 of the file or call stack (= T1 − 1 when nothing is restated) | C3 omitted (C1 + C2 + C3 = 1); C15 localization, C14 verification; I1 counts edits |

Columns: *Domain*. *Φ_D* lists the recommended coordinates by catalogue ID (Section 4). *d_Φ* is their number. *χ_nest* is the coordinate that carries the depth floor (Section 3). *Notes* explains choices.

**Table 5.3. Which terms of Theorem 2 apply in each domain, and under which conditions.** The data-side terms (W1, ε) need no access to the architecture; the architecture-side terms (positional encoding, depth, width) need an autoregressive Transformer with known L, m and positional encoding.

| Term | Needs | VLA | Tool-use agent | Embodied agent | World-model planner | Long CoT | Code agent |
|---|---|---|---|---|---|---|---|
| W1 shift | task-side references for source and target; coordinates that are structural, computed on the task side, bounded and metric, at most 16 of them | reference episodes | log of reference runs | z_ref | reference-planner traces | reference traces | reference run and gold patch |
| ε | per-instance 0-1 outcomes of the evaluated system; the sufficiency check | episode success of policy rollouts | task completion | goal condition (expected over repeated rollouts if stochastic) | goal reached | answer correct | tests pass |
| Positional encoding / Lipschitz factor | positional encoding in a class of Appendix F; Assumption 2; coordinates with a bounded change under T_k; T_k-type insertions occur | discrete action-token policies; with learned absolute timestep embeddings the argument of Lemma 2 applies with the timestep as position; continuous action heads outside | open-weight backbones, on the context; content clocks and truncation are not T_k | Transformer policies, open-weight planners; reversible environments weaken Assumption 2 | token-based Transformer world models; continuous-latent and diffusion models outside; recurrent models have no positional encoding | open-weight models | as for tool-use agents |
| Depth floor | a depth coordinate (Section 3) that counts the nesting resolved implicitly within one forward pass | N1 (h) when reasoning is dropped or latent; memoryless policies lack information instead | N2 on the context; nesting held by an outside program excluded | subgoal stack not restated | N1 (goal chain), M5 (composition); trees held by an outside search program excluded | N2 | N2 of the file or call stack |
| Width | m known; Assumption 4 restated on R^{d_Φ} | backbone or action decoder | open-weight backbones | open-weight planners | token-based world models | open-weight models | open-weight backbones |

Columns: *Term* is a term of Theorem 2. *Needs* states what the term requires in any domain. Each further column states how the requirement is met, or where it fails, in one domain.

### 5.1 Conditions by domain

- **VLA policies.** The per-step context (image tokens, instruction, reasoning, action tokens) has a fixed structure, so Φ_VLA is defined on the closed-loop episode and computed on reference episodes. The coordinate that plays the role of χ1 is the depth h of the stack of pending subgoals (N1). It is produced by preconditions (open a container to place an object, then close it), recoveries inside a subtask and interjections that suspend the current goal. It differs from plan length and horizon: a sequence of independent pick-and-place subtasks has h = 1 at any length. Coordinates derived from logged signals (proprioception, gripper state, success detection) are preferred to coordinates parsed from text written by a model or an annotator, which depend on wording. For a policy that restarts at every step, the architecture-side terms concern one control step; for policies with history or memory, the history window.
- **Tool-use agents.** A harness maps the log to the sequence of contexts c_1, ..., c_T that the model reads. The focus stack (enter a new page, resource, subgoal or file; return to one on the stack; act within the current one) is the analogue of the search tree. It measures navigation nesting, which need not coincide with the nesting of subgoals. An outside search program replaces one deep context with backtracking by many shallow contexts without it: the log keeps the tree, the contexts lose its backtracking. Multi-agent systems have no single context; Φ is then defined on the message graph, and the architecture-side terms per agent.
- **Embodied agents.** Three objects must be kept apart. z_ref is the trajectory of an oracle or expert planner on the same instance; it is task side and defines ν_S, ν_T, the depth coordinate and the sufficiency check. z_prefix is the agent's first steps; it is model side and serves monitoring only. z_full is the agent's whole episode, whose length, dead ends and loops encode the outcome; it serves diagnosis after the fact only. Waits, no-ops, retries and uninformative feedback are the natural inserted content; history windows of fixed length delete content. Assumption 2 fits settings in which actions are irreversible or evaluated by exact match; where a wrong action can be undone, one routing error need not rule out success.
- **World-model planners.** A trajectory interleaves an imagination loop (propose, predict, score, deepen, branch or abandon within one decision) and a reality loop (commit, observe, continue or replan). Tree coordinates are computed per decision and aggregated over the episode. Φ_WM needs a discrete planner structure, not discrete states: tree search over latent states has a discrete tree, whereas a sampling-based optimizer without a tree reduces T, C and N to horizon, iterations and replans. Three shifts are distinguished: between the planner's queries and the world model's training rollouts, between environments or embodiments, and between planner settings (horizon, beam width, interval at which imagined states are written out, search budget). Under dynamics that do not change over time, the correct prediction does not depend on the absolute frame index, whereas under a learned absolute positional encoding of frame tokens the attention kernel changes when the same transition occurs at another index; this is the setting of Lemma 2. Continuous-latent and diffusion models lie outside Assumption 2, and recurrent models have no positional encoding.
- **Long chains of thought.** The reasoning text is split into steps. The reference is a reference solver's or teacher's trace on the same problem, and the training data are a sample of μ_S; the evaluated model's own reasoning is model side. Log and context coincide, as in the paper. Push, pop, retreat and verification markers, and the step parser, are part of the adapter and are applied identically to every compared measure, as the metric property requires. The depth floor applies to the cases that are not restated (N2). Reasoning that is latent at test time leaves no trajectory, so Φ can only be computed on training supervision.
- **Code agents.** The log consists of view, search, edit, run and submit actions and their observations; the task side consists of a reference run, the gold patch and the failing tests; the outcome is whether the tests pass. The terms apply as for tool-use agents. Events with the same name, such as a failed edit, can be detected by different rules in different harnesses; event definitions are unified before W1 is computed across harnesses, as the metric property requires. A fixed pipeline that runs the same steps on every instance gives a nearly constant Φ.

## 6. Building a projection for a new domain

1. Fix the object of analysis: log or context, the task-side reference, and the per-instance outcome of the evaluated system.
2. Write the adapter primitives of Table 4.1: entity, node reference, push, pop, lateral move, action type, negative and positive feedback, restatement, external event, inert event.
3. List the kinds present in the domain (Table 2.1) and select candidates by kind from the catalogue.
4. Check from the definitions that every coordinate is structural, computed on the task side, bounded and metric, that the coordinates change under T_k by a bounded amount and one of them counts the inserted event, and that one coordinate counts the implicit nesting. For this depth coordinate, state the homomorphism ψ_D, which entries are restated or moved out of the forward pass, and the adoption of Assumption 5.
5. Select at most 16 coordinates; fix weights, marker sets and standardization by one rule for all compared measures; report the within-measure baseline and the per-coordinate attribution next to every shift.
6. Run the sufficiency check of Appendix J against LEN and PROB, grouping folds by task where several trajectories share one.
7. Before using an architecture-side term, verify its conditions: an autoregressive Transformer with known L, m and positional encoding; Assumption 2 plausible in the domain; T_k-type insertions present (S5); a depth coordinate that counts only the implicit part (N2, M5).
8. Record the meta-attributes of every coordinate: source, outcome flag, unit, view and aggregation.

## 7. Experiment 1: from 12 to 34 coordinates on Countdown

Code and results: [`coordinates_12_to_34/`](../extensions/coordinates_12_to_34/). Its [`results/README.md`](../extensions/coordinates_12_to_34/results/README.md) defines all 34 coordinates and every column of the result files; [`results/tables.md`](../extensions/coordinates_12_to_34/results/tables.md) gives ROC-AUC, McFadden pseudo-R² and Ŵ1 for every d, model and pair of measures, and the CSV files of the same folder give them per repetition and per seed.

By Section 3.2, further coordinates can lower the attainable ε, never lower W1, and make W1 harder to estimate. Experiment 1 (Appendix M.1.1) measures both effects on Countdown with the nested projections Φ12 = (χ1, ..., χ12) ⊂ Φ16 ⊂ Φ24 ⊂ Φ34, where the subscript is the number of coordinates. Their 22 further coordinates belong to a Countdown version of the catalogue, are computed on the reference trajectory and enter in the class order of the catalogue code:

- Φ16 adds four coordinates of class T: distinct nodes, revisit ratio, mean and maximum branching.
- Φ24 adds eight: mean pop size, lateral moves into another subtree, backtracking runs, longest failure streak, position of the first goal (as a fraction of the tokens), tokens after the first goal and goals reached (class C), and the mean return distance in tokens (class M).
- Φ34 adds ten: the maximum return distance in tokens and the mean number of open nodes (class M); the maximum pop size, the maximum number of open nodes and the restated states per node reference (class N); operations, operations per state and dead ends per operation (class I); token length and number of lines (class S).

Open nodes are nodes referenced both before and after a node reference; the return distance is the number of tokens since the previous reference to the node returned to. The catalogue code gives the last four of these class-C coordinates, the failure streak and the three goal coordinates, a class of their own (recovery and termination). Four of the further coordinates are counted in tokens (tokens after the first goal, the two return distances and token length). The two measurements concern ε and W1, and the bounded change under T_k enters only the gap of Theorem 1.

**Design.**
- *Prediction of success (the sufficiency check).* The gradient-boosting classifier of Appendix J, which predicts from the coordinates of z whether a trained model fails on the problem of z, with its labels for f1 (50,000 problems) and f2 (3,000), is fitted on each Φ_d with ten repetitions of stratified five-fold cross-validation. The folds of a repetition are shared by all d, and repetition 0 is the split of Appendix J.
- *Estimation of W1.* Ŵ1 is computed as in Appendix G on the three pools of 20,000 training trajectories (one pool per search procedure), z-scored on the pooled sample, with N = 2,000 and ten seeds whose subsamples are shared by all d. Two kinds of distance are computed: the within-measure baseline (between two disjoint samples of N = 2,000 trajectories of one pool) and the three between-measure distances.

**Results.** The prediction of success stays the same: from Φ12 to Φ34 the out-of-fold ROC-AUC (mean over the ten repetitions) goes from 0.927 to 0.929 for f1 and from 0.929 to 0.927 for f2, so it changes by at most 0.003, and the McFadden pseudo-R² changes by at most 0.016. The change in ROC-AUC is below 0.01 in absolute value in every repetition, and the s.d. of the ROC-AUC over repetitions is at most 0.0016. The within-measure baselines grow strictly with d, by a factor of 1.6-2.3 from d = 12 to d = 34 (for BFS from 0.19 to 0.44), and the between-measure distances by a factor of 1.8-2.0 (for BFS-DFS from 2.97 to 5.65), so that the ratio of the mean baseline to the mean between-measure distance stays at 0.13. For every d the baseline falls with the sample size as N^(−α), with α = 0.40-0.48 (N = 250 to 4,000). On Countdown the further coordinates thus roughly double Ŵ1 and its noise floor without improving the prediction of success. By Section 3.2, they would tighten K_f·W1 + 2ε only by lowering 2ε more than K_f·W1 grows. The twelve coordinates are the better choice, within the budget of Section 3. The same two measurements can be run in any domain to decide whether a coordinate is worth adding.

## 8. Agent, embodied and robot trajectories (Experiments 2 and 3)

Experiments 2 and 3 (Appendix M.2) apply projections Φ_D and the estimate Ŵ1 to recorded logs of language agents, code agents and robots. Experiment 2 (Appendix M.2.1, Sections 8.1 to 8.6) compares groups of reference trajectories and asks whether Ŵ1 reports a clear shift between groups whose tasks differ in structure and a small one between groups whose tasks share it. Its task-side projections Φ_D are assembled from the catalogue of Section 4, and Ŵ1 is estimated as in Appendix G. Experiment 3 (Appendix M.2.2, Section 8.7) runs one language model in two agent harnesses.

Code, per-trajectory coordinates and results of Experiment 2: [`structural_audit/`](../extensions/structural_audit/). [`SOURCES.md`](../extensions/structural_audit/SOURCES.md) lists the sources, their licenses and what the released files contain, and [`results/README.md`](../extensions/structural_audit/results/README.md) explains the result files.

### 8.1 Trajectory sources

Every unit is a reference trajectory, namely an expert demonstration, a resolved run or a level solved by the BabyAI bot, and only derived per-trajectory coordinates are used (licenses in parentheses).

- **(a)** Expert trajectories of ETO (Song et al., 2024a; 2024b) (Apache-2.0) in ALFWorld (Shridhar et al., 2021) (3,119), ScienceWorld (Wang et al., 2022) (1,483) and WebShop (Yao et al., 2022) (1,824, of which 905 human and 919 GPT-4 demonstrations).
- **(b)** Resolved runs on GitHub issues of the policy `swe-agent-llama-70b` in SWE-agent (Yang et al., 2024; Nebius, 2024) and of Qwen3-Coder-480B-A35B-Instruct (Qwen Team, 2025) in OpenHands (Wang et al., 2025; Trofimova et al., 2025) on SWE-rebench issues (Badertdinov et al., 2025) (both CC-BY-4.0): 12,467 runs on 835 issues and 32,161 runs on 3,792 issues.
- **(c)** Teleoperated BridgeData V2 demonstrations (Walke et al., 2023) (CC-BY-4.0) with the embodied chain-of-thought features of ECoT (Zawalski et al., 2025; Embodied-CoT, 2024) (MIT), restricted to the 33,252 episodes with a non-empty language instruction.
- **(d)** Trajectories of the BabyAI bot (Chevalier-Boisvert et al., 2019) (BSD-3-Clause) on freshly generated BabyAI-MixedTrainLocal-v0 levels from the Minigrid fork distributed with BALROG (Chevalier-Boisvert et al., 2023; Paglieri et al., 2024) (Apache-2.0), 4,000 solved levels per instruction kind.

Outcome fields (rewards, resolution, exit status, success of the bot) serve only to select resolved runs and solved levels, and every coordinate is computed on the task side. The coordinates are read from actions, observations and logged signals, so that they are structural; the parsers skip the model-written thoughts of ETO and the language plans of ECoT.

### 8.2 Groups and pairs

All pairs among the groups of each family form the 52 pairs.

- **A:** the six ALFWorld task types: pick and place (759), look at an object in light (279), clean (576), heat (371), cool (472), two objects (662); 15 pairs.
- **B:** the three ETO environments; 3 pairs. **B2:** human against GPT-4 demonstrations in WebShop; 1 pair.
- **C:** SWE-agent against OpenHands runs, once on the 554 issues resolved by both and once on all resolved issues (835 and 3,648 issues, the latter covered by a random 20,000 of the OpenHands runs); each sampled issue contributes one of its resolved runs. In C the harness and the policy model change together, so C measures a swap of harness and model.
- **D:** seven skills read from the language instruction: pick and place (23,778 episodes, of which a random 20,000 form the pool), fold (2,780), open (1,493), close (1,355), flip (909), sweep (840), turn (810); 21 pairs. The 1,287 episodes whose instruction matches none of the seven skills belong to no group.
- **E:** five BabyAI instruction kinds: go to, pick up, open, put next to, and sequences of a pick-up followed by a go-to (phrased "pick up ..., then go to ..." or "go to ... after you pick up ..."); 10 pairs.

### 8.3 Projections

Table 8.1 lists the coordinates of each Φ_D, by catalogue ID where one exists, and the adapter that defines its states and depth; d_Φ ranges from 12 to 16, within the budget of Section 3.

**Table 8.1. Projections Φ_D of Section 8.** Every coordinate is computed on the reference trajectory. The surface coordinates, removed in the variant "without S", are S2 in A, B, B2 and E; S2 and the share of inert steps in D; none in C.

| Family | d_Φ | Coordinates | States and depth |
|---|---|---|---|
| A: ALFWorld task types | 14 | S2 actions; T4 location references (go-to actions); T5 distinct locations; T6 location revisit ratio; T8 cycle rank of the location graph; N1 maximum and T2 mean open-context depth; C1 push and C2 pop rate per action; C7 search misses (receptacles checked without finding an object of the target type); C15 actions before the first take; I1 manipulation actions (take, put, open, close) and transformation actions (clean, heat, cool, use, toggle, slice); I2 entropy of the action verbs | open contexts: a held object and the receptacles opened and not yet closed (take and open push, put and close pop) |
| B, B2: ETO environments; WebShop demonstrators | 12 | S2 actions; T5 distinct states; T6 state revisit ratio; T8 cycle rank of the state graph; N1 maximum and T2 mean depth; C1 push and C2 pop rate; C7 negative-feedback observations; I1 navigation and manipulation actions; I2 entropy of the action verbs | one feature map with an adapter per environment: ALFWorld location and open contexts; ScienceWorld room and the number of held, opened and activated or connected objects; WebShop page and page depth (search 0, results 1, item 2, item sub-page 3) |
| C: harness and model | 16 | T1-T3 maximum, mean and s.d. of the file-focus depth; T4 tool calls; T5 distinct files opened; T6 file reopen ratio and share of repeated identical actions; C1 push (new file) and C2 pop (return to a stacked file) rate; C7 failed edits; C9 longest failed-edit cascade; C14 run and test calls; C15 position of the first edit; I1 read calls (view, search) and write calls (edit); I3 error observations per call | file-focus stack of the code agent (Table 4.1), parsed into one step schema for both harnesses; token counts, submission, exit status, resolution and harness-specific tools are excluded |
| D: Bridge skills (VLA) | 16 | S2 control steps and share of inert (stop) steps; T4 primitive segments; T5 distinct primitives; T6 segment revisit ratio; T8 cycle rank of the segment-transition graph; N1 maximum hold depth, number of holds and T2 mean hold depth; translation reversals; C9 longest stall; C15 steps before the first grasp; I1 translation, rotation and gripper steps; I2 entropy of the primitives | per-step movement primitives that ECoT derives from proprioception; closing the gripper pushes a hold, opening it pops |
| E: BabyAI instruction kinds | 13 | T1-T3 maximum, mean and s.d. of the subgoal-stack depth; T4 stack events; C1 deepening and C2 backtracking rate of the per-step stack depth; C7 explore subgoals and C8 their share of pushes; I1 manipulation and navigation subgoals; S2 primitive actions; I1 primitive turns; T9 initial-state size (non-wall objects) | the explicit subgoal stack of the BabyAI bot (push and pop log); the bot's success is constant on solved levels and is not a coordinate |

Columns: *Family* is the group family of Section 8.2. *d_Φ* is the number of coordinates. *Coordinates* lists them, each preceded by its catalogue ID where one exists. *States and depth* says what the adapter takes as states and how it defines depth.

### 8.4 Estimator and separation rule

As in Appendix G, the coordinates are z-scored with the mean and standard deviation of the pooled candidate pools of all groups of a family (a pool holds all units of a group, or a random 20,000 of them). The cost is the Euclidean distance between standardized vectors, and Ŵ1 is the exact transport cost with uniform weights, computed by `ot.emd2` of POT 0.9.6 (Flamary et al., 2021).

For a pair of groups with n_a and n_b units, N = min(2000, ⌊n_a/2⌋, ⌊n_b/2⌋), between 139 and 2,000. Each of 10 seeds draws N units of each group for the between-group estimate and, for the within-group baseline of each group, two disjoint samples of N units of that group. With W̄ the seed mean of the between-group Ŵ1 and B the larger of the two mean baselines at the same N, the ratio is R = W̄/B.

A pair is *separated* if R ≥ 3 and the seed CV of the between-group Ŵ1 is at most 0.10. Because the baseline decreases as N grows, the R of each pair uses the baselines at the N of that pair.

### 8.5 Results

33 of the 52 pairs are separated (Figure 4 of the paper shows R for every pair): 9 of 15 in A, 3 of 3 in B, 1 of 1 in B2, 2 of 2 in C, 9 of 21 in D and 9 of 10 in E. All pairs completed with 10 seeds and without transport warnings, and the largest seed CV is 0.095.

- The three ETO environments are separated from one another with R = 6.26-10.60, nine of the ten pairs of BabyAI instruction kinds with R = 10.94-28.65, the WebShop demonstrators with R = 4.11, and the harness-and-model pair with R = 3.58 on the shared issues and on all issues.
- Among the ALFWorld task types, 9 of 15 pairs are separated, with R up to 4.12.
- The Bridge skills reach 9 of 21 (median R = 2.72). Close is separated from all six other skills, and turn from open, fold and sweep; the ten pairs among pick and place, open, fold, sweep and flip have R = 1.61-2.72. The demonstrations of one skill vary widely, and the large within-skill baselines (1.23-1.99, against at most 1.39 for ALFWorld and 0.61 for ETO) lower R.
- Pairs whose tasks share a sequence of subgoals stay below R = 3: heat against cool in ALFWorld (R = 1.48, the smallest ratio of all pairs; both take an object, use an appliance and place the object), clean against heat and against cool (2.42 and 2.36), and put next to against a pick-up followed by a go-to in BabyAI (2.92).

**Variants.** Without the surface coordinates, 34 of the 52 pairs are separated. With z-scores pooled per pair instead of per family, 30 pairs are separated, and with the mean instead of the larger of the two baselines, 37. In E the initial-state size T9 is constant within each kind; in an additional run without it, the same nine pairs are separated (R = 9.39-28.65).

**Attribution.** One-dimensional Ŵ1 per coordinate attributes the separated Bridge pairs that involve close mainly to the hold coordinates and to the steps before the first grasp (C15), and the harness-and-model pair to the file-focus depth (T1), the tool calls (T4) and the run calls (C14).

The estimate Ŵ1 on Φ_D thus separates 33 of the 52 pairs of groups of reference trajectories, among them most of those that differ in environment, instruction kind or demonstrator, and it estimates the distance W1(ν_S, ν_T) of the shift term.

### 8.6 Results of all 52 pairs

[`results/table_pairs.md`](../extensions/structural_audit/results/table_pairs.md) lists all 52 pairs. [`results/results.json`](../extensions/structural_audit/results/results.json) holds the same values at full precision for the main analysis (`main`) and the two variants (`noS`, without the surface coordinates; `pairz`, with z-scores pooled per pair), together with the one-dimensional Ŵ1 of every coordinate in every pair (`attribution_main`); [`results/babyai_drop.json`](../extensions/structural_audit/results/babyai_drop.json) holds the run of family E without T9, and the `design_*.json` files the coordinates, groups and pool sizes of each family. The columns of `table_pairs.md` are:

- *Family*: the family of Section 8.2; C_H1 is SWE-agent against OpenHands on the issues resolved by both, C_H2 on all resolved issues.
- *Pair*: the two groups compared. Family A: simple = pick and place, look = look at an object in light, clean, heat, cool, two = two objects. Family B: SciWorld = ScienceWorld. Family D: PICKPLACE = pick and place, and the six other skills. Family E: goto = go to, pickup = pick up, open, putnext = put next to, seq = a pick-up followed by a go-to.
- *N*: units per sample.
- *W1 between (sd)*: mean (s.d.) over the 10 seeds of the between-group Ŵ1.
- *baseline a / b*: mean within-group Ŵ1 of the first / second group of the pair at the same N.
- *R = W1/max*: the between-group Ŵ1 divided by the larger baseline.
- *R (noS)*: R without the surface coordinates (family C has none, so its value equals R).
- *R (pair z)*: R with the z-scores computed on the two groups of the pair instead of the family.
- *top coordinates (1-D W1)*: the three coordinates with the largest one-dimensional Ŵ1 between the two groups, with that Ŵ1. The names carry the catalogue IDs of Table 8.1; in family E, chi1_maxdepth to chi12_nav are the subgoal-stack coordinates of Table 8.1 in the order listed there, ref_len is S2 (primitive actions), ref_turns the primitive turns and n_objs is T9.
- *separated*: yes if R ≥ 3 and the seed CV is at most 0.10; the CV condition holds for every pair.

### 8.7 Experiment 3: one model in two agent harnesses

Code, instance list and results: [`harness_switch/`](../extensions/harness_switch/); its [`results/README.md`](../extensions/harness_switch/results/README.md) explains every file and column.

In family C the harness and the policy model change together. Experiment 3 (Appendix M.2.2) holds the model fixed and changes only the harness, the program that builds the prompt, offers the model its tools and runs its commands. One model, Qwen3-8B (Qwen Team, 2025) (Apache-2.0), an open-weight model with eight billion parameters, in bf16 with thinking disabled and served by vLLM (Kwon et al., 2023), was run in mini-SWE-agent 2.4.6 (Yang et al., 2024), which gives the model a single shell tool, and in OpenHands (Wang et al., 2025; software-agent-sdk 1.16.1 with the OpenHands benchmarks of June 2026), which has separate tools for shell commands and for viewing and editing files; both harnesses are under the MIT license. A local proxy gives every request of both harnesses the same sampling settings (temperature 0.7, top-p 0.8, top-k 20, at most 4,096 new tokens). The runs cover 300 instances of SWE-bench Verified (Jimenez et al., 2024; OpenAI, 2024), stratified by repository, with one episode per instance and harness, each in a fresh container of the same prebuilt image of the instance and limited to 100 model calls and 3,000 s.

The units are the agent's own episodes, resolved or not. They are model-side trajectories (Section 2.1), the trajectories that run-time monitors, verifiers and trajectory filters read. Both logs are parsed into the step schema of family C and projected onto its 16 coordinates (Table 8.1).

**Results.** With N = 150 and 10 seeds, Ŵ1 between the harnesses is 3.81 (seed CV 0.024), against within-harness baselines of 1.48 (mini-SWE-agent) and 1.74 (OpenHands), so R = 2.19, below the separation threshold R = 3 of Experiment 2. Without the tool-call count, R = 2.16; with the file-focus rule of family C applied to both harnesses, R = 2.48. One-dimensional Ŵ1 locates the shift: every coordinate moves by more than its own within-harness baseline (ratios 1.95-11.5), most of all the read calls (50.4 per episode under mini-SWE-agent against 17.3 under OpenHands), the file reopen ratio (0.03 against 0.42) and the tool calls (73.8 against 43.5). Failed edits (C7) are more frequent and the file-focus stack (T1) is deeper under OpenHands (standardized mean differences 0.30 and 0.63). The resolve rate changes with the harness as well, from 1.7% to 6.4% (+4.7 points, 95% CI [2.0, 7.7], McNemar p = 0.001, on the 299 instances whose gold patch passes, each harness with its own patch); with the final working tree as the patch in both harnesses it changes from 2.3% to 6.4% (+4.0 points, 95% CI [1.0, 7.0]). With Qwen3-8B held fixed, the change of harness thus moves the trajectories along every coordinate and changes how often the agent succeeds, so a monitor, verifier or trajectory filter calibrated on one harness receives shifted inputs under the other.

The values are in [`results/w1_summary.csv`](../extensions/harness_switch/results/w1_summary.csv) (Ŵ1, baselines and R of the three variants; per seed in `w1_per_seed.csv`), [`results/coordinate_attribution.csv`](../extensions/harness_switch/results/coordinate_attribution.csv) (one-dimensional Ŵ1, its ratio to the baseline, the standardized mean difference and the mean under each harness, for every coordinate), [`results/resolve_rates.csv`](../extensions/harness_switch/results/resolve_rates.csv) (resolve rates, bootstrap intervals and McNemar test; per instance in `resolve_by_instance.csv`) and [`results/trajectory_coordinates.csv`](../extensions/harness_switch/results/trajectory_coordinates.csv) (the 16 coordinates of each of the 600 episodes).

## 9. Lemma 2 under shifted position indices (Experiments 4 to 6)

Lemma 2 states that shifting the position indices of a sub-trajectory changes the attention scores of an APE model almost surely, while under RoPE they stay the same (the identity ΔA^RoPE = 0 of Appendix C.3). These are statements about the attention kernel, so they can be checked on public pretrained models without training. Experiments 4 to 6 (Appendix M.3) shift all position indices without inserting a token, so the tokens and their distances stay the same, and in exact arithmetic every attention score under RoPE and every attention weight under ALiBi is unchanged. Experiment 4 (Section 9.1) checks the attention scores of eight pretrained language models. Experiment 5 (Section 9.2) follows the shift through to the answers of 33 public models, under the mechanisms that produce such shifts in serving. Experiment 6 (Section 9.3) applies the shift to the two Countdown models f1 and f2 of Section 5 of the paper.

### 9.1 Experiment 4: attention scores of pretrained models

Code and results: [`lemma2_pretrained/`](../extensions/lemma2_pretrained/). [`results/table_cells.csv`](../extensions/lemma2_pretrained/results/table_cells.csv) gives every metric for each model, precision (`fp32`, or `fp64diag` for the float64 run), input family and k; [`results/paper_table_models.tex`](../extensions/lemma2_pretrained/results/paper_table_models.tex) gives the ranges per model over k and input family; [`results/summary.json`](../extensions/lemma2_pretrained/results/summary.json) adds the change in every layer and the three controls (`V1`, `V2`, `V3`); and `results/raw/` holds the per-sequence statistics from which the analysis is recomputed. [`results/README.md`](../extensions/lemma2_pretrained/results/README.md) explains every file and column.

**Models.** Four models with learned absolute position embeddings: GPT-2 124M and 355M (Radford et al., 2019) (modified MIT license) and GPT-Neo 125M and 1.3B (Black et al., 2021) (MIT). Four RoPE models (Su et al., 2024): Pythia 160M, 410M and 1.4B (Biderman et al., 2023), which rotate 25% of the dimensions of each head, and SmolLM2 360M (Allal et al., 2025), which rotates all of them (Apache-2.0). In the APE models the position embeddings are added to the token embeddings, and the sum passes through layer normalization before the query and key projections. The kernel therefore depends on absolute positions, as in the position-dependent class of Appendix F, and Experiment 4 checks the conclusion of Lemma 2 on these models.

**Inputs.** For each model, 512 sequences of 512 tokens under its own tokenizer, 128 from each of four families: Countdown reference trajectories of μ_DFS and of μ_BFS (Section 5 of the paper), synthetic Dyck-4 strings of four kinds of brackets nested at most 12 deep, and source files of the Python 3.10 standard library (Python Software Foundation, 2025). The same items are used for every model.

**Shift.** Every position index is shifted by k in {16, 64, 256}, with the tokens unchanged. The translated sub-trajectory ω of Definition 2 is then the whole input, and for RoPE the identity holds for every attended pair in every layer, since by induction over layers the input of each layer is unchanged. The largest position index, 767, lies within the context of every model.

**Numerics.** Each sequence runs in one batch together with its unshifted version, a replicate of the unshifted version and the three shifted versions, in fp32 with TF32 disabled and eager attention (Hugging Face `transformers` 4.57.1). Every sequence is rerun in float64, with the fp32-only casts of the implementations removed. Scores are taken over the attended pairs (causal, and within the 256-token window in the local layers of GPT-Neo).

**Metrics.**
- m1: the change of the pre-softmax scores A relative to their scale, max abs(ΔA) / max abs(A) and mean abs(ΔA) / mean abs(A), over heads, attended pairs and sequences, in the first layer and over all layers.
- m2: the fraction of positions whose top-1 next-token prediction changes.
- m3: the change of the per-token log-likelihood, averaged over tokens.

In `table_cells.csv`, `m1_first_max` and `m1_first_mean` are the two forms of m1 in the first layer and `m1_all_max` and `m1_all_mean` the same maximized over layers (the `_abs` columns give the changes before division by the score scale); `m2` is m2 as a fraction (`m2_count` of `n_positions`); `m3_signed` and `m3_abs` are the mean signed and absolute change of the per-token log-likelihood, in nats, and `base_nll` the per-token negative log-likelihood without the shift; in the float64 rows, `F_base_*` and `F_shift_*` are the deviations of the fp32 scores from float64 without and with the shift, relative to the same scale, which show how large pure rounding error is.

**Results**, over all 96 combinations of model, k and input family (8 models × 3 values of k × 4 input families):
- Under RoPE, the first-layer change is 1.6e-6 to 6.2e-6 of the score scale in fp32, and the all-layer change in float64 is at most 4.4e-12.
- Under APE, the first-layer change is 0.42-1.19 of the score scale, and in every cell it is at least 175 times the largest fp32 change of any RoPE model in any layer at the same k and input family.
- RoPE changes at most 0.038% of the top-1 predictions, all at near-ties (top-1 minus top-2 logit at most 0.0036), and SmolLM2 changes none. APE changes 14.0-87.4% of them, and the shift lowers the log-likelihood in every APE cell, by 0.20-6.77 nats per token.
- In deeper layers the fp32 RoPE change reaches 3.2e-3 (Pythia 160M), the order of the deviation of the unshifted fp32 computation from float64 (up to 1.5e-3), and it falls to at most 4.4e-12 in float64. The RoPE differences are therefore floating-point rounding.
- Three controls pass: explicit position indices reproduce the default forward pass exactly; the replicated unshifted row reproduces the base exactly; and an independent float64 re-implementation of the first layer from the raw weights (GPT-2 124M, GPT-Neo 125M, Pythia 160M, SmolLM2 360M) reproduces the APE changes and gives RoPE changes of at most 1.0e-14.

The contrast of Lemma 2 thus holds in the attention scores and predictions of these eight pretrained models.

### 9.2 Experiment 5: answers under shifted position indices

Code and results: [`shifted_positions_serving/`](../extensions/shifted_positions_serving/). [`results/table_pooled.csv`](../extensions/shifted_positions_serving/results/table_pooled.csv) gives, for every model, condition and shift, the accuracy without the shift, the change Δ with its 95% interval and the share of answers that change; [`results/table_cells.csv`](../extensions/shifted_positions_serving/results/table_cells.csv) gives the same per task, with the changes of the first-token logits; [`results/cards/`](../extensions/shifted_positions_serving/results/cards/) holds one summary per model; and [`results/README.md`](../extensions/shifted_positions_serving/results/README.md) explains the columns and names the rows behind every number below. Figure 5 of the paper is drawn from `table_pooled.csv` by [`extensions/figures_appendix_m/make_fig_shift.py`](../extensions/figures_appendix_m/make_fig_shift.py).

A serving stack can give a request position indices that differ from those of a fresh call with the same tokens, through padding, cached prefixes and sliding windows. Experiment 5 (Appendix M.3.2) asks whether these shifts change the answers of APE models and leave those of RoPE and ALiBi models unchanged. It uses 33 public base models, trained only to predict the next token:
- twelve with learned absolute position embeddings: GPT-2 124M-1.5B (Radford et al., 2019) (modified MIT license), GPT-Neo 125M-2.7B (Black et al., 2021) (MIT) and OPT 125M-6.7B (Zhang et al., 2022) (OPT-175B license agreement, which permits non-commercial research use);
- five with ALiBi (Press et al., 2022): BLOOM 560M-7.1B (BigScience Workshop, 2022) (BigScience BLOOM RAIL License v1.0);
- sixteen with RoPE: Pythia 160M-6.9B (Biderman et al., 2023) and GPT-J 6B (Wang and Komatsuzaki, 2021) (both Apache-2.0), Qwen2.5 0.5B-7B (Qwen Team, 2024) (Apache-2.0; 3B under the Qwen Research License), OLMo-2 1B and 7B (Team OLMo et al., 2024) and SmolLM2 135M-1.7B (Allal et al., 2025) (both Apache-2.0).

ALiBi adds to every score a bias that depends only on the distance j − i between key position j and query position i, and adds no position embedding to the input. Its attention weights, like the scores of RoPE, are therefore unchanged when all position indices are shifted by the same amount; in the BLOOM implementation the scores of a query move by one constant, which leaves the weights unchanged.

**Items.** Each model answers 3,000 items: 500 few-shot exact-match items for each of five tasks, and 500 key-value retrieval items filled to the context of the model (at most 700, 1,700 and 3,000 tokens for contexts of 1,024, 2,048 and at least 4,096 tokens). The five few-shot tasks are: copy a list of 6-24 words; reverse 2-4 words; close the open brackets of a Dyck-4 prefix; add two numbers of 1-2 digits; complete the deterministic remainder of a line of a Countdown search trace of Section 5 of the paper.

**Decoding.** Greedy and KV-cached, in fp32 (TF32 disabled) with eager attention. The answer up to the first newline is scored by exact match.

**Conditions.** With shifts k in {16, 64, 256, 1024}, restricted so that no position exceeds the trained context, each item runs in one batch with its reference and a replicate of the reference under the following conditions (the name of the scenario in the result files in parentheses).
- (i) All position indices of the prompt and of the generation shifted by k (`offset`).
- (ii) Left padding: position indices are assigned by index over k masked pad tokens, so that the real tokens start at position k instead of 0 (`leftpad`).
- (iii) Prefix caching: the prompt before the final query line is prefilled into the KV cache and the rest of the prompt follows; both parts are placed k positions later (`cache`, rows `block<k>`).
- (iv) Sliding window: every layer attends to the 128 most recent cache slots and the first four slots (attention sinks; Xiao et al., 2024), with all positions shifted by k, compared with the same window without the shift (five few-shot tasks only; `window`).
- (v) Control: the cached prefix keeps its positions and the rest of the prompt starts k positions later. This changes the relative distances across the cache boundary (`cache`, rows `gap<k>`).
- (vi) Condition (i) in bf16, compared with bf16 without the shift (`offset_bf16`).

Δ is the change of accuracy from the reference, with a paired bootstrap 95% CI over items (10,000 resamples).

**Results.**
- In fp32, the 21 ALiBi and RoPE models give the same answers under the shifts (i)-(iv) except for 40 of the 948,000 compared answers, never more than one in a cell of 2,500-3,000 items, and no accuracy changes by more than 0.0003.
- The twelve APE models lose accuracy under all four. At k = 256 they lose 0.13-0.52 under (i)-(iii), which give identical changes, and 0.04-0.50 under (iv), and under (i) at k = 256 they change 70-100% of their answers.
- Changing relative distances changes the answers under every positional encoding: under (v), 63 of the 64 RoPE cells and 19 of the 20 ALiBi cells change significantly, by up to 0.44 at k = 64 (up to 0.45 for RoPE and 0.50 for ALiBi at k = 1,024).
- In bf16 (vi) the APE losses remain. The RoPE models except Pythia 160M, whose bf16 accuracy without the shift is 0.012 (0.209 in fp32), stay within 0.022 of their bf16 accuracy without the shift, with significant changes only for Qwen2.5 1.5B at k = 256 and Qwen2.5 7B at k = 64 and 256, whereas the ALiBi models lose 0.007-0.033 at k = 256 and 0.28-0.36 at k = 1,024. The BLOOM implementation adds the ALiBi bias as the slope times the absolute key position, which equals the slope times j − i up to a constant per query, so its bf16 rounding grows with the shift, consistent with this loss.

The shifts of position indices that occur in serving thus change the answers of the APE models, while the ALiBi and RoPE models in fp32 keep theirs as long as the distances between tokens are kept.

### 9.3 Experiment 6: two Countdown models of Section 5 of the paper

Code and results: [`shifted_positions_checkpoints/`](../extensions/shifted_positions_checkpoints/). [`results/tables.md`](../extensions/shifted_positions_checkpoints/results/tables.md) gives the teacher-forced and free-running results by model, site type and shift, the CSV files of the same folder give them per item and per problem, and [`results/README.md`](../extensions/shifted_positions_checkpoints/results/README.md) defines the sites, items and routing tokens.

Experiment 6 (Appendix M.3.3) applies the same shift of all position indices to the two models of Appendix J, f1 = Small Deep MIXED RoPE (GPT-NeoX, L = 12) and f2 = Large Deep BFS APE (GPT-Neo, L = 48), in fp32 with eager attention, and asks whether it changes their routing decisions and their success on Countdown. The routing decisions are the node identifier after `Moving to Node` and the operation after `Exploring Operation` (Appendix K.4). They are read after the first leaf-level operation line of each trajectory that meets the site rule of the results README: a line that explores an operation on a node with two numbers, such as `Exploring Operation: 3+10=13, Resulting Numbers: [13]`, before the end of the search and directly after a dead end at the same node or a return to that node, where the node still has an unexplored operation whose result misses the target. After this line the model must decide where the search goes next.

- **Teacher-forced.** The model reads a reference trajectory, and its routing predictions after this line are compared with and without a shift of all position indices by 16 or 64. On the 2,863 reference trajectories of the three evaluation sets in which such a line, followed by at least one routing decision, lies within the first 896 tokens, the shift changes none of the 94,278 routing predictions of the RoPE model, and 25.7% (shift 16) and 28.3% (shift 64) of those of the APE model.
- **Free-running.** The model generates its own trajectory, generation resumes at this line with all position indices shifted by 16 or 64, and the outcome is scored as in Section 5 of the paper. The test uses the first 1,000 problems of each evaluation set and applies to the 2,128 problems for the RoPE model and the 1,143 for the APE model whose output contains such a line with at least 64 of the 512 tokens of the generation budget left. For the RoPE model the outcome of none of these problems changes (success 0.594). For the APE model, success falls from 0.527 by 0.40 (shift 16) and by 0.39 (shift 64).

The contrast of Lemma 2 thus carries over to the task success of two models of the paper, consistent with the mechanism of Assumption 2, in which changed routing decisions turn into failed searches.

## 10. Splitting deeply nested tasks into separate model calls (Experiments 7 and 8)

Sub-agents, task lists and outside search programs move nesting out of the forward pass into separate contexts (Section 3.3). By Lemma 3, nested subgoals form a Dyck word whose height is their nesting depth, so a program that splits a task into calls of bounded height bounds the nesting that each forward pass resolves. We call such a program a *splitting program*. Lemma 5 bounds the risk from below when the nesting that one forward pass must resolve exceeds α_circ·L, which gives the depth term. Experiments 7 and 8 (Appendix M.4) measure whether splitting restores success when the nesting exceeds what a model resolves in one call, and how deep each call should be: Experiment 7 (Section 10.1) with models trained from scratch on nested plans, Experiment 8 (Section 10.2) with open-weight language models on nested tool calls.

Figure 6 of the paper is drawn by [`extensions/figures_appendix_m/make_fig_split.py`](../extensions/figures_appendix_m/make_fig_split.py) from `results/runs.csv` and `results/per_depth.csv` of the nested plans (Section 10.1) and `results/metrics.json` of the nested tool calls (Section 10.2).

### 10.1 Experiment 7: nested plans

Code and results: [`nested_plans/`](../extensions/nested_plans/). [`results/runs.csv`](../extensions/nested_plans/results/runs.csv) gives d_ext, the call depth and the success with one call and with splitting for each of the 120 runs; [`results/per_depth.csv`](../extensions/nested_plans/results/per_depth.csv) the success per run and depth; [`results/selection_30000_steps.csv`](../extensions/nested_plans/results/selection_30000_steps.csv) the runs of 30,000 steps from which the retrained configurations were chosen; and `results/runs/` the full output of every run. [`results/README.md`](../extensions/nested_plans/results/README.md) explains the columns.

**Task.** Experiment 7 (Appendix M.4.1) trains small models on nested plans and asks whether splitting restores success beyond d_ext, and whether d_ext follows the number of layers, as the threshold α_circ·L suggests, or the depth of the training plans. A plan is a forest of 48 subgoals of 8 types whose maximum nesting depth is exactly d: a spine of d nested subgoals, and 48 − d further subgoals attached at random to nodes of depth below d or to the top level. It is written as a push/pop word, `( g` to open a subgoal of type g and `) g` to close it, in 194 tokens for every d. At every pop the model predicts the type, that is, the type of the matching push. A plan is solved when all 48 predictions are correct. Since all other tokens are given, teacher-forced and free-running success coincide.

**Models and training.** We trained the GPT-NeoX (RoPE) and GPT-Neo (learned APE) architectures of Appendix K from scratch: width 256 with 4 heads and L in {2, 4, 8, 12} layers, and a two-layer model of width 624 with 12 heads whose number of non-embedding parameters is within 1.3% of that of L = 12. Each is trained on fresh plans with d uniform on {1, ..., d_train}, d_train in {2, 4, 6, 8}, for 30,000 steps of 256 plans (AdamW, peak learning rate 1e-3 after 1,000 warm-up steps with cosine decay to a tenth of it, weight decay 0.1, loss on the predicted types), with three seeds: 120 runs. The five configurations in which some run stayed below a single-call success of 0.9 at a depth d ≤ d_train (all APE: the width-624 model at d_train = 2, 4, 6 and 8, and L = 2 at d_train = 4) were trained again, all three seeds, with 60,000 steps; these runs replace the runs of 30,000 steps. After the longer training, three of the five configurations reach a single-call success of at least 0.9 at every depth up to d_train in every run.

**Calibration and splitting.** On a calibration set of 200 plans per depth, d_ext is the largest depth at which a fit of the single-call success that is not allowed to increase with depth is at least 0.5. The splitting program reads only the skeleton of the plan, that is, which subgoal opens and closes where, and never a scored type. With a budget D ≥ 2, each call writes the subgoals down to relative depth D. A subgoal at relative depth D that has children is written as a stub `( g ) g`, whose prediction is scored like the others, and its subtree becomes a new call in a fresh context. A plan deeper than d_ext is split with D = min(d_ext, d_train), so that every call has nesting depth at most d_train; other plans take one call. Test sets have 500 plans per depth up to 4·d_train + 8, disjoint from the calibration set, and a plan counts as solved when every prediction of every call is correct.

**Results.**
- In 38 of the 40 configurations, every run reaches a single-call success of at least 0.9 at every depth up to d_train. In all 114 runs of these configurations, splitting gives a mean success of at least 0.987 over the depths beyond d_ext, and at least 0.91 at each of them, against at most 0.04 for one call (Table 10.1; Figure 6b of the paper shows the 12-layer RoPE model).
- d_ext follows d_train: it is 2, 4-5, 6-9 and 9-12 for d_train = 2, 4, 6 and 8, with no systematic dependence on the number of layers, the width or the positional encoding, and exceeds d_train by at most four levels (Figure 6a of the paper). Deeper training saves calls: beyond d_ext, a plan needs 20.8 calls on average at d_train = 2 and 4.5-4.6 at d_train = 8.
- The two other configurations are the two-layer width-624 APE model at d_train = 2 and 4, whose single-call success stays below 0.9 at some depth up to d_train after 60,000 steps. Five of their six runs have d_ext = 1, too shallow for a split with D ≥ 2, so every plan is solved in one call; in the sixth, splitting raises the success beyond d_ext = 2 from 0.00 to 0.998.

In models of 2 to 12 layers, the depth that one call resolves thus follows the training depth whatever L is, and splitting into calls within the training depths restores success on deeper plans.

**Table 10.1. Experiment 7: the 38 configurations (114 runs) in which every run reaches a single-call success of at least 0.9 at every depth up to d_train, by training depth, computed from `results/runs.csv`.** Each row pools RoPE and APE, width 256 with L in {2, 4, 8, 12} and the two-layer width-624 model, the latter without its APE runs at d_train in {2, 4}. Values are rounded as in Appendix M.4.1; `results/runs.csv` gives them per run at full precision.

| d_train | Test depths | Runs | d_ext, RoPE | d_ext, APE | Single call | Split, mean | Split, min | Calls per plan |
|---|---|---|---|---|---|---|---|---|
| 2 | 1-16 | 27 | 2 | 2 | 0.00 | 1.000 | 1.00 | 20.8 |
| 4 | 1-24 | 27 | 4-5 | 4 | 0.04 | 0.987 | 0.91 | 8.0-8.1 |
| 6 | 1-32 | 30 | 7-8 | 6-9 | 0.04 | 0.997 | 0.98 | 5.5-5.6 |
| 8 | 1-40 | 30 | 9-11 | 9-12 | 0.03 | 0.997 | 0.99 | 4.5-4.6 |

Columns:
- *d_train*: largest nesting depth in training.
- *Test depths*: the depths d of the test set.
- *Runs*: number of runs pooled in the row.
- *d_ext, RoPE* and *d_ext, APE*: range over runs of d_ext, the largest depth at which the fitted single-call success (a fit not allowed to increase with depth) is at least 0.5 on the calibration set.
- The last four columns concern the test depths d > d_ext of each run. *Single call*: the largest, over runs, of the mean single-call success over these depths. *Split, mean*: the smallest, over runs, of the mean success with splitting over these depths. *Split, min*: the smallest success with splitting at any of these depths in any run. *Calls per plan*: mean number of calls per plan with splitting (range over runs).

### 10.2 Experiment 8: nested tool calls with open-weight language models

Code and results: [`nested_tool_calls/`](../extensions/nested_tool_calls/). [`results/test_by_depth.csv`](../extensions/nested_tool_calls/results/test_by_depth.csv) gives the accuracy, the calls and the tokens per item of every model, condition and depth; [`results/dhat.csv`](../extensions/nested_tool_calls/results/dhat.csv) gives d̂ of every model with its bootstrap interval; [`results/calibration_by_depth.csv`](../extensions/nested_tool_calls/results/calibration_by_depth.csv) the calibration accuracies and the fit; [`results/rerun.csv`](../extensions/nested_tool_calls/results/rerun.csv) a repeat of a tenth of the single-call test items in a new vLLM process; and [`results/README.md`](../extensions/nested_tool_calls/results/README.md) explains the columns and lists the values behind every number below.

**Task.** Experiment 8 (Appendix M.4.2) measures d̂ for pretrained language models, whose training depth is unknown, on a task shaped like the tool calls of an agent. An item is a registry of 26 lines (one-line Python functions named by common English nouns, and comment lines) and the question `What does f1(x0) return?`. Every function either returns x + c or calls one other function and adds c, with c in {1, 2, 3} and x0 in {0, ..., 9}. The answer requires following the call chain f1 → ... → fd, among two distractor functions and 24 − d comment lines that keep the length of the prompt constant over d. Four fixed demonstrations precede every prompt, and the first integer of at most eight generated tokens is scored by exact match. The calibration set has 100 and the test set 200 items per depth d in {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24}.

**Models.** Eleven instruction-tuned models: Qwen2.5 7B, 14B and 32B (Qwen Team, 2024), OLMo-2 7B, 13B and 32B (Team OLMo et al., 2024) and Qwen3 8B, 14B and 32B (Qwen Team, 2025) with thinking disabled (all Apache-2.0), Phi-4 14B (Abdin et al., 2024) (MIT) and Mistral-Small-24B-Instruct-2501 (Mistral AI, 2025) (Apache-2.0). All are served by vLLM (Kwon et al., 2023) in bf16 with greedy decoding.

**Conditions.**
- *I, one call*: a single call answers directly.
- *E, one call with the chain listed*: a single call with the open call chain listed in the prompt.
- *A, one level per call*: the splitting program reads only which function calls which, never a constant or a value. It splits the chain bottom-up into segments of at most D levels; each segment is a fresh call in which the definition of the function below the segment is replaced by the value returned by the previous call. Condition A uses D = 1, so it makes d calls.
- *H, calls of d̂ levels*: the same with D = d̂, so it makes ⌈d/d̂⌉ calls. Here d̂ is the largest depth at which a fit of the single-call accuracy on the calibration set that is not allowed to increase with depth is at least 0.8.

When d̂ = 1, H makes the same calls as A; it was run separately and is within 0.015 of A at every depth.

**Results.**
- The accuracy of one call falls quickly as the chain grows, from 0.88-1.00 at d = 1 to at most 0.52 at d = 4 and at most 0.15 at d = 8 (Figure 6c of the paper). d̂ is 3 for Qwen2.5 32B, 2 for Qwen3 14B, Phi-4 14B and Mistral-Small-24B, and 1 for the other seven models. Listing the call chain (E) changes the single-call accuracy by at most 0.09 on average over depths.
- One level per call raises the accuracy of the eight Qwen, Phi and Mistral models at d = 8 from 0.01-0.15 to 0.43-0.99, and Mistral-Small-24B answers 0.70 of the chains correctly at d = 24. For the three OLMo-2 models, one level per call gives 0.04-0.06 at d = 8, against 0.00-0.03 in one call, and 0.37-0.59 at d = 2, against 0.43-0.72.
- The errors of the calls compound along the chain, so the accuracy with one level per call also declines with d.
- For the four models whose d̂ is 2 or 3 (Figure 6d of the paper), calls of d̂ levels need 4 or 3 calls at d = 8 and reach 0.24-0.51, against 0.43-0.99 with one level per call. By the definition of d̂, a call of d̂ levels errs in up to a fifth of the cases, and these errors compound over the calls.

One level per call thus beats calls of d̂ levels at d = 8 for all four models where the two differ, since each of its calls stays at a depth where one call is almost always right. For Qwen2.5 32B, two levels per call (its d̂ at the threshold 0.9) are within 0.04 of one level per call up to d = 5 and at least as accurate from d = 6 on.

## 11. Summary of the eight experiments

Table 3 of Appendix M.5 summarizes the eight experiments. Each concerns one part of Theorem 2: Experiment 1 the balance between the residual ε and the estimation of W1, Experiments 2 and 3 the distance W1 in the shift term, Experiments 4 to 6 the factor K_{f,PE} through Lemma 2, and Experiments 7 and 8 the depth term. This section follows the eight rows of Table 3 and adds, for each experiment, how its result is used.

**Experiment 1, more coordinates on Countdown (Section 7; Appendix M.1.1).** On Countdown, projections of 12, 16, 24 and 34 coordinates predict success equally well (the out-of-fold ROC-AUC changes by at most 0.003), while Ŵ1 between measures grows by a factor of 1.8-2.0 and the within-measure baselines by 1.6-2.3. The twelve coordinates are the better choice. In any domain, a coordinate is worth adding when it improves the prediction of success; a coordinate that leaves the prediction unchanged makes W1 harder to estimate (Section 3.2).

**Experiment 2, structural shift between groups of trajectories (Sections 8.1 to 8.6; Appendix M.2.1).** On public agent, embodied and robot trajectories, Ŵ1 on Φ_D separates 33 of the 52 pairs of groups (34 without the surface coordinates), among them most of those that differ in environment, instruction kind or demonstrator, and tasks that share a sequence of subgoals give the smallest ratios. The one-dimensional Ŵ1 per coordinate attributes the separation to named coordinates, for example the file-focus depth, the tool calls and the run calls for the harness-and-model pair. The same procedure compares any two samples of trajectories for which a domain adapter exists: Φ_D is chosen from the catalogue (Sections 3 and 4), the coordinates are standardized on the pooled sample, and Ŵ1 between reference trajectories for a sample of deployment tasks and the training data is reported next to the within-group baseline at the same N, which sets the noise floor, and next to the one-dimensional Ŵ1 of every coordinate. These quantities state how far a deployment sample lies from the training data in structure, relative to the sampling noise, and along which coordinates. The comparison needs task-side references and no access to model weights (Table 5.3). It estimates the distance W1(ν_S, ν_T) of the shift term, which enters the bound of Lemma 1 multiplied by K_f, together with 2ε.

**Experiment 3, one model in two agent harnesses (Section 8.7; Appendix M.2.2).** With Qwen3-8B held fixed, switching from mini-SWE-agent to OpenHands gives R = 2.19 on the 16 coordinates of family C, every coordinate moves by 1.95-11.5 times its own within-harness baseline, and the resolve rate rises from 1.7% to 6.4%. Monitors, verifiers and trajectory filters calibrated on one harness therefore receive shifted inputs after a switch and should be checked again; the one-dimensional attribution names the coordinates to check first, here the read calls, the file reopen ratio and the tool calls.

**Experiment 4, attention scores of pretrained models (Section 9.1; Appendix M.3.1).** Under a shift of all position indices by 16, 64 or 256, the first-layer attention scores of four APE models change by 0.42-1.19 of the largest absolute score, and those of four RoPE models change by rounding (at most 4.4e-12 over all layers in float64). The shift changes 14.0-87.4% of the top-1 predictions of the APE models and at most 0.038% of those of the RoPE models, all at near-ties.

**Experiment 5, answers under shifted position indices (Section 9.2; Appendix M.3.2).** Left padding with position indices that count the padding, a prefix cache reused at shifted positions and a sliding window that starts deep into a stream shift the position indices of a request while keeping its tokens and the distances among them; these are the conditions (ii)-(iv) of Section 9.2. In fp32, the 21 ALiBi and RoPE models give the same answers under these three and under (i), apart from 40 of 948,000 answers, and the twelve APE models lose 0.13-0.52 in accuracy at a shift of 256 under (i)-(iii) and change 70-100% of their answers under (i). For APE models, position indices should therefore be computed from the attention mask so that every request starts at position zero, and a cache should be reused only at its own positions. For ALiBi and RoPE models in fp32 the shift leaves the answers unchanged as long as the distances between tokens are kept; a cache reused with a gap before the rest of the prompt changes the distances, and with them the answers under every positional encoding (control (v)). In bf16, the RoPE models other than Pythia 160M stay within 0.022 of their unshifted bf16 accuracy, and the ALiBi models lose 0.28-0.36 at a shift of 1,024, so for them large shifts are better avoided at that precision.

**Experiment 6, two Countdown models (Section 9.3; Appendix M.3.3).** On f1 (RoPE) and f2 (APE) of Section 5 of the paper, a shift of all position indices by 16 or 64 changes none of the 94,278 routing predictions and no outcome of the RoPE model, and 25.7% and 28.3% of the routing predictions of the APE model, whose success falls by 0.40 and 0.39. Experiments 4 to 6 thus show the contrast of Lemma 2 for a shift of all position indices in the attention scores of eight pretrained models, in the answers of 33 public models and in the task success of two models of the paper.

**Experiment 7, nested plans (Section 10.1; Appendix M.4.1).** The splitting rule uses only calls to the model and the nesting of the task: on a small calibration set of increasing nesting depth, find the depth d_ext beyond which a single call fails, and split deeper tasks into calls no deeper than the training depths. In the 114 runs that learned their training depths, splitting gives a mean success of at least 0.987 beyond d_ext, against at most 0.04 for one call. d_ext follows d_train for every number of layers, width and positional encoding, and deeper training extends d_ext and saves calls (20.8 calls per plan at d_train = 2, 4.5-4.6 at d_train = 8).

**Experiment 8, nested tool calls (Section 10.2; Appendix M.4.2).** With the eleven open-weight language models of 7B-32B parameters, d̂ is 1-3, and the single-call accuracy of every model is at most 0.15 at d = 8. One level per call raises the accuracy of the eight Qwen, Phi and Mistral models at d = 8 from 0.01-0.15 to 0.43-0.99. For the four models whose d̂ is 2 or 3, calls of d̂ levels save calls and reach 0.24-0.51, since the errors of the calls compound. Each call should therefore stay at a depth where one call is almost always right, which for most of these models is one level.

## References

- Abdin, M., Aneja, J., Behl, H., et al. (2024). Phi-4 Technical Report.
- Allal, L. B., Lozhkov, A., Bakouch, E., et al. (2025). SmolLM2: When Smol Goes Big — Data-Centric Training of a Fully Open Small Language Model. COLM.
- Badertdinov, I., Golubev, A., Nekrashevich, M., et al. (2025). SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents. NeurIPS Datasets and Benchmarks Track.
- Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML.
- BigScience Workshop (2022). BLOOM: A 176B-Parameter Open-Access Multilingual Language Model.
- Black, S., Gao, L., Wang, P., Leahy, C., Biderman, S. (2021). GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. Zenodo.
- Chevalier-Boisvert, M., Bahdanau, D., Lahlou, S., et al. (2019). BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning. ICLR.
- Chevalier-Boisvert, M., Dai, B., Towers, M., et al. (2023). Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks. NeurIPS Datasets and Benchmarks Track.
- Embodied-CoT (2024). Embodied-CoT/embodied_features_bridge. Hugging Face dataset.
- Flamary, R., Courty, N., Gramfort, A., et al. (2021). POT: Python Optimal Transport. JMLR.
- Jimenez, C. E., Yang, J., Wettig, A., et al. (2024). SWE-bench: Can Language Models Resolve Real-world GitHub Issues? ICLR.
- Kwon, W., Li, Z., Zhuang, S., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP.
- Li, Z., Liu, H., Zhou, D., Ma, T. (2024). Chain of Thought Empowers Transformers to Solve Inherently Serial Problems.
- Liu, B., Ash, J. T., Goel, S., Krishnamurthy, A., Zhang, C. (2023). Transformers Learn Shortcuts to Automata.
- Mistral AI (2025). Mistral Small 3. Mistral AI news.
- Nebius (2024). nebius/SWE-agent-trajectories. Hugging Face dataset.
- OpenAI (2024). Introducing SWE-bench Verified. OpenAI blog.
- Paglieri, D., Cupiał, B., Coward, S., et al. (2024). BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games. arXiv:2411.13543.
- Press, O., Smith, N. A., Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR.
- Python Software Foundation (2025). Python 3.10.19.
- Qwen Team (2024). Qwen2.5 Technical Report.
- Qwen Team (2025). Qwen3 Technical Report.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners.
- Shridhar, M., Yuan, X., Côté, M.-A., et al. (2021). ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. ICLR.
- Song, Y., Yin, D., Yue, X., Huang, J., Li, S., Lin, B. Y. (2024a). Trial and Error: Exploration-Based Trajectory Optimization of LLM Agents. ACL.
- Song, Y., Yin, D., Yue, X., Huang, J., Li, S., Lin, B. Y. (2024b). ETO-SFT-Trajectory: Expert Trajectories for ETO. Hugging Face dataset.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing.
- Team OLMo, Walsh, P., Soldaini, L., et al. (2024). 2 OLMo 2 Furious.
- Trofimova, M., Shevtsov, A., Badertdinov, I., et al. (2025). OpenHands Trajectories with Qwen3-Coder-480B-A35B-Instruct. Nebius blog.
- Walke, H., Black, K., Lee, A., et al. (2023). BridgeData V2: A Dataset for Robot Learning at Scale. CoRL.
- Wang, B., Komatsuzaki, A. (2021). GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.
- Wang, R., Jansen, P., Côté, M.-A., Ammanabrolu, P. (2022). ScienceWorld: Is your Agent Smarter than a 5th Grader? EMNLP.
- Wang, X., Li, B., Song, Y., et al. (2025). OpenHands: An Open Platform for AI Software Developers as Generalist Agents. ICLR.
- Weed, J., Bach, F. (2019). Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance. Bernoulli.
- Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. (2024). Efficient Streaming Language Models with Attention Sinks. ICLR.
- Yang, J., Jimenez, C. E., Wettig, A., et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS.
- Yao, S., Chen, H., Yang, J., Narasimhan, K. (2022). WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. NeurIPS.
- Yao, S., Yu, D., Zhao, J., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS.
- Zawalski, M., Chen, W., Pertsch, K., Mees, O., Finn, C., Levine, S. (2025). Robotic Control via Embodied Chain-of-Thought Reasoning. CoRL.
- Zhang, S., Roller, S., Goyal, N., et al. (2022). OPT: Open Pre-trained Transformer Language Models.
