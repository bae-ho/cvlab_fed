import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=88888888)
parser.add_argument('--pel_diagnostics', action='store_true',
                    help='Read-only loss-gradient and round-update diagnostics; plain FedAvg UCF/CCTV only')
parser.add_argument('--pel_warmup_rounds', type=int, default=0)
parser.add_argument('--pel_ramp_rounds', type=int, default=0)
parser.add_argument('--pel_pooling', choices=['score', 'uniform'], default='score')
parser.add_argument('--pel_class_balance', choices=['none', 'batch'], default='none')
parser.add_argument('--pel_gradient_scope', choices=['all', 'last_block'], default='all')
parser.add_argument('--pel_projection', action='store_true', help='Shared identity-initialized PEL-only linear head')
parser.add_argument('--pel_allow_aggregation_integration', action='store_true',
                    help='Explicit opt-in fixed-bank connection to existing FedNova/FedPGP; no aggregation changes')
parser.add_argument('--pel4vad_variant', choices=['legacy', 'fixed_bank'], default='legacy',
                    help='legacy: dynamic batch text; fixed_bank: detached full-class anchors, plain FedAvg only')
parser.add_argument('--pel4vad_prompts', type=str, default=None,
                    help='Optional JSON: dataset class key -> list of fixed text prompts')

parser.add_argument("--dataset", type=str, default="ucf", choices=["ucf", "xd", "cctv"])
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Local learning rate")
parser.add_argument("--global_rounds", type=int, default=20, help="Global federated learning rounds")
parser.add_argument("--local_epochs", type=int, default=10, help="local training epochs for each client")
parser.add_argument("--algorithm", type=str, default="FedAvg", choices=["FedAvg", "FedProx", "Scaffold"])
parser.add_argument("--clients_num", type=int, default=13, help="Number of client per round when random")
parser.add_argument("--split_mode", type=str, default="event", choices=["random", "event", "scene"])
parser.add_argument('--scheduler_rate', default=0.1)
parser.add_argument('--scheduler_milestones', default=[])
parser.add_argument('--load_model', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--embed_dim', default=512, type=int, help="CLIP embedding size")
parser.add_argument('--visual_length', default=256, type=int, help="length of video")
parser.add_argument('--visual_width', default=512, type=int)
parser.add_argument('--visual_head', default=1, type=int)
parser.add_argument('--visual_layers', default=2, type=int, help="number of visual layers, default 2 for ucf")

parser.add_argument('--attn_window', default=8, type=int, help="size of attention window, default 8 for ucf")
parser.add_argument('--prompt_prefix', default=10, type=int)
parser.add_argument('--prompt_postfix', default=10, type=int)

parser.add_argument('--aggregation_method', type=str, default='fedavg',
                     choices=['fedavg', 'fedpgp', 'fedprox', 'scaffold', 'fednova'],
                     help="fedavg: FedAvg over all trainable params (baseline). "
                          "fedpgp: FedAvg over shared params only, client low-rank adapter stays local. "
                          "fedprox: FedAvg + proximal term anchoring local weights to the global model. "
                          "scaffold: FedAvg + control variates correcting client drift. "
                          "fednova: normalize each client's update by its local step count "
                          "before averaging (fixes objective inconsistency from unequal steps).")
parser.add_argument('--prox_mu', type=float, default=0.01,
                     help="FedProx proximal coefficient mu; only used with --aggregation_method fedprox")

parser.add_argument('--aggregation_weight_unit', type=str, default='loader_batches',
                     choices=['loader_batches', 'effective_batches'],
                     help="what n_i counts in the server's weighted average. "
                          "loader_batches: len(normal)+len(anomaly), the original unit. "
                          "effective_batches: min(len(normal), len(anomaly)), the batches the "
                          "ucf/cctv loop actually consumes per epoch, since it pairs one normal "
                          "batch with one anomaly batch and stops at the shorter loader. The two "
                          "differ only where a client's loaders are unbalanced (UCF scene most of "
                          "all). No effect on xd, which has one unpaired loader per client. "
                          "Default keeps existing runs comparable.")

parser.add_argument('--server_optimizer', type=str, default='none',
                     choices=['none', 'momentum', 'adam', 'yogi'],
                     help="How the server applies the aggregated update. none: plain FedAvg. "
                          "momentum: FedAvgM. adam: FedAdam. yogi: FedYogi. "
                          "Orthogonal to --aggregation_method, so any pair can be combined.")
parser.add_argument('--server_lr', type=float, default=None,
                     help="Server step size; defaults to 1.0 for none/momentum and 1e-2 for adam/yogi")
parser.add_argument('--server_beta1', type=float, default=0.9,
                     help="Server momentum coefficient (beta for FedAvgM, beta1 for FedAdam/FedYogi)")
parser.add_argument('--server_beta2', type=float, default=0.99, help="FedAdam/FedYogi second-moment decay")
parser.add_argument('--server_tau', type=float, default=1e-3, help="FedAdam/FedYogi adaptivity constant")

parser.add_argument('--local_param_policy', type=str, default=None,
                     choices=['none', 'adapter', 'norm', 'adapter+norm'],
                     help="Which trainable params stay on the client instead of being averaged. "
                          "none: all averaged. adapter: FedPGP low-rank adapter. "
                          "norm: LayerNorm affine params (FedBN-style, absorbs per-client feature shift). "
                          "Defaults to 'adapter' when --personalization_type is set, else 'none'.")
parser.add_argument('--personalization_type', type=str, default='none', choices=['none', 'full_rank', 'low_rank'],
                     help="client-persistent prompt adapter architecture; 'none' reproduces the baseline model exactly")
parser.add_argument('--personalized_rank', type=int, default=4, help="rank r of the low-rank adapter A_i (N_ctx x r) / B_i (r x D)")
parser.add_argument('--personalized_alpha', type=float, default=1.0, help="LoRA-style scale: delta = (alpha / r) * A_i @ B_i")
parser.add_argument('--personalized_lr', type=float, default=None,
                     help="learning rate for the personalized adapter; defaults to --learning_rate if unset")
parser.add_argument('--personalized_weight_decay', type=float, default=None,
                     help="weight decay applied only to the low-rank adapter (AdamW's default 1e-2 is used "
                          "if unset). Acts as a zero-mean prior so the adapter self-disables on IID splits "
                          "where there is no client-specific signal, instead of adding noise.")
parser.add_argument('--use_fedpgp_contrastive', action='store_true',
                     help="add contrastive loss pulling the global prompt toward CLIP's handcrafted class embedding")
parser.add_argument('--contrastive_weight', type=float, default=0.1)
parser.add_argument('--contrastive_temperature', type=float, default=0.07)
parser.add_argument('--use_pel4vad_contrastive', action='store_true',
                     help="add PEL4VAD-style video-to-prompt contrastive loss (Pu et al., arXiv:2306.14451): "
                          "pools each clip's visual features toward its own class prompt and toward the "
                          "normal prompt using the model's own per-snippet scores as soft attention. "
                          "Orthogonal to --use_fedpgp_contrastive; only meaningful for --dataset ucf/cctv "
                          "(single label per clip).")
parser.add_argument('--pel4vad_weight', type=float, default=1.0)
parser.add_argument('--pel4vad_temperature', type=float, default=0.09)
parser.add_argument('--verbose_fed_logging', action='store_true',
                     help="log personalized delta norm / shared update norm / client-vs-aggregate cosine drift each round")
parser.add_argument('--personalized_checkpoint_dir', type=str, default=None,
                     help="directory of client_{i}_personalized.pth files to resume personalized adapters from")

parser.add_argument('--inference_mode', type=str, default='global', choices=['global', 'personalized'],
                     help="global: Delta_P_i disabled (compare against original Fed-WSVAD). "
                          "personalized: apply one client's adapter on top of the shared model.")
parser.add_argument('--personalized_client_id', type=int, default=0,
                     help="client index whose adapter to load for --inference_mode personalized")
