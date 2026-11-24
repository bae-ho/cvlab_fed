# Aggregation strategies

Beyond the original FedAvg baseline and the FedPGP personalization work, this repo now
implements several aggregation strategies that target *different* known failure modes of
FedAvg under Non-IID clients. They are exposed as three **orthogonal** flags, so any
combination is one flag each.

## Why FedAvg struggles here

Measured on this repo's actual splits (`batch_size 64`, `drop_last=True`):

| split | clients | optimizer steps / epoch | spread |
|---|---|---|---|
| event | 13 | 4 (Shooting) … 22 (Robbery) | **5.5×** |
| scene | 9 | 1 (Gas station, Factory) … 41 (Street) | **41×** |

Three distinct problems follow from this, and each needs a different fix:

1. **Client drift.** With `local_epochs 10`, each client travels far from the shared
   starting point toward its own optimum. Averaging the end points cancels the
   disagreeing components. → *FedProx, SCAFFOLD*
2. **Objective inconsistency.** A client that runs 41 steps moves much further than one
   that runs 1, so it dominates the average for a reason that has nothing to do with its
   data being more informative — and it *already* gets a larger weight `n_i` on top of
   that. The FedAvg fixed point ends up depending on the step counts. → *FedNova*
3. **Noisy, oscillating server updates.** Consecutive rounds pull in partly conflicting
   directions; plain averaging has no memory across rounds. → *FedAvgM / FedAdam / FedYogi*

A fourth, orthogonal issue is **feature shift**: scene-split clients see systematically
different visual statistics (a bank lobby vs. a street), which a single set of shared
normalization parameters has to straddle. → *FedBN-style local LayerNorms*

## The three axes

### 1. `--aggregation_method` — how client models are reduced

| value | what it does | cost |
|---|---|---|
| `fedavg` | sample-weighted average (baseline) | — |
| `fedpgp` | averages shared params only; low-rank adapter stays on the client | +2k params/client |
| `fedprox` | FedAvg + `(mu/2)·‖w − w_global‖²` in the local loss, bounding drift | +43 MB total, ~5-10% slower |
| `scaffold` | FedAvg + control variates correcting the local gradient to `g − c_i + c` | +0.6 GB @ 13 clients |
| `fednova` | normalizes each client's update by its own step count before averaging | free |

FedNova, concretely:

```
d_i     = (w_i − w_global) / tau_i
tau_eff = sum_i p_i · tau_i
delta   = tau_eff · sum_i p_i · d_i
```

It provably reduces to FedAvg when each client's update is proportional to its step
count (i.e. when there is no inconsistency to correct), and diverges from it otherwise.

### 2. `--server_optimizer` — how the server applies the aggregated update

The server treats `delta = Σ p_i (w_i − w_global)` as a pseudo-gradient
(Reddi et al., ICLR 2021) and can run any first-order optimizer on it. **Costs nothing
on the clients** and composes with every `--aggregation_method`.

| value | rule | default `--server_lr` |
|---|---|---|
| `none` | `w ← w + delta` (plain FedAvg) | 1.0 |
| `momentum` | FedAvgM: `m ← β·m + delta`; `w ← w + lr·m` | 1.0 |
| `adam` | FedAdam: per-coordinate normalization by the second moment | 1e-2 |
| `yogi` | FedYogi: Adam with a more conservative second-moment update | 1e-2 |

`adam`/`yogi` normalize the step, so their learning rate is an absolute step size — do
not carry over `1.0` from the plain server.

### 3. `--local_param_policy` — which trainable params never leave the client

| value | kept local | rationale |
|---|---|---|
| `none` | nothing | plain FedAvg |
| `adapter` | `personalized_prompt.*` (2,064 params @ r=4) | FedPGP; default when `--personalization_type` is set |
| `norm` | LayerNorm affine params (10 tensors) | FedBN analogue; absorbs per-client feature shift |
| `adapter+norm` | both | |

There is no BatchNorm in this model, so the FedBN idea is applied to LayerNorm instead.
The parameters are located by module type (`isinstance(m, nn.LayerNorm)`), not by name
matching, and frozen CLIP norms are excluded by the `requires_grad` check.

## Commands

```bash
# baselines (unchanged, still routed to the original server classes)
python train.py --dataset ucf --split_mode scene --clients_num 9
python train.py --dataset ucf --split_mode scene --clients_num 9 --aggregation_method fedpgp --personalized_rank 4

# drift control
python train.py --dataset ucf --split_mode scene --clients_num 9 --aggregation_method fedprox --prox_mu 0.01
python train.py --dataset ucf --split_mode scene --clients_num 9 --aggregation_method scaffold

# objective-inconsistency control (most directly motivated by the 41x step spread)
python train.py --dataset ucf --split_mode scene --clients_num 9 --aggregation_method fednova

# server-side adaptivity (cheapest thing to try first)
python train.py --dataset ucf --split_mode scene --clients_num 9 --server_optimizer momentum
python train.py --dataset ucf --split_mode scene --clients_num 9 --server_optimizer adam

# feature-shift control
python train.py --dataset ucf --split_mode scene --clients_num 9 --local_param_policy norm

# combinations
python train.py --dataset ucf --split_mode scene --clients_num 9 \
    --aggregation_method fednova --server_optimizer adam --local_param_policy norm
```

Add `--verbose_fed_logging` to print per-round `tau` per client, the aggregated update
norm, and each client's cosine similarity against the aggregate (a direct read on how
much clients are disagreeing).

## Implementation notes

* `FedAvgServer` and `FedPGPServer` are deliberately **untouched**, so the two experiments
  already run with them stay reproducible. Everything else routes through
  `FedStrategyServer`, which reduces to plain FedAvg when all three flags are default.
* `FedStrategyServer` hands the initial global weights to clients *before* round 1 — without
  that, FedProx has no anchor and SCAFFOLD no control update during the first round.
* FedProx anchors only shared parameters. A personalized adapter is supposed to diverge, and
  there is no global copy of it to anchor against.
* SCAFFOLD is derived for local SGD; this repo uses AdamW, so the control correction is
  injected into `.grad` and then rescaled by Adam's preconditioner. That is the usual
  practical adaptation, not the analyzed algorithm — treat its numbers as indicative.
* `c_global` and the `w_global` reference are shared objects rather than per-client copies,
  which is what keeps SCAFFOLD's overhead at ~0.6 GB instead of ~1.2 GB at 13 clients.

## References

- FedProx — Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020
- SCAFFOLD — Karimireddy et al., ICML 2020
- FedNova — Wang et al., *Tackling the Objective Inconsistency Problem*, NeurIPS 2020
- FedAvgM — Hsu et al., *Measuring the Effects of Non-Identical Data Distribution*, 2019
- FedAdam / FedYogi — Reddi et al., *Adaptive Federated Optimization*, ICLR 2021
- FedBN — Li et al., *Federated Learning on Non-IID Features via Local Batch Normalization*, ICLR 2021
