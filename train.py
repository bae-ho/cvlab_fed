import os
import json
import subprocess
import random
import numpy as np
import torch
import utils.config as config
from fed.server import FedAvgServer, FedPGPServer, FedStrategyServer
from utils.dataset import make_xd_dataloader, make_ucf_dataloader, make_cctv_dataloader
from utils.model import Model
from datetime import datetime
from utils.pel_experiments import (aggregation_integration, cli_pel_options,
                                   diagnostics_supported, validate_pel_cli)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = config.parser.parse_args()
    setup_seed(args.seed)
    if args.pel_diagnostics and not diagnostics_supported(args):
        config.parser.error('pel_diagnostics requires UCF/CCTV with PEL off or fixed_bank, under '
                            'either plain FedAvg or --pel_allow_aggregation_integration')
    try:
        validate_pel_cli(args)
    except ValueError as error:
        config.parser.error(str(error))
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    if args.aggregation_method == 'fedpgp' and args.personalization_type == 'none':
        args.personalization_type = 'low_rank'

    # Only the two already-validated configurations keep their original server class, so
    # results produced with them stay reproducible. Anything else (a drift-correcting
    # aggregation rule, a server optimizer, or a non-default local-parameter policy) goes
    # through the generalized server.
    #
    # An opted-in aggregation integration always takes the generalized server even when the
    # axes happen to spell out a legacy-equivalent configuration: FedStrategyServer is the
    # only one carrying the round-level PEL diagnostics, and 'fedpgp there' is numerically
    # the same plain average over shared parameters that FedPGPServer performs.
    use_legacy_server = (
        args.aggregation_method in ('fedavg', 'fedpgp')
        and args.server_optimizer == 'none'
        and args.local_param_policy in (None, 'none', 'adapter')
        and not aggregation_integration(args))

    if use_legacy_server and args.aggregation_method == 'fedavg' and args.personalization_type != 'none':
        raise ValueError("--aggregation_method fedavg aggregates every trainable parameter, "
                          "so a client adapter would be averaged away and stop being personalized. "
                          "Use --aggregation_method fedpgp, or keep fedavg and set "
                          "--local_param_policy adapter to route through FedStrategyServer.")

    start_time = datetime.now()
    dir_name = start_time.strftime("%Y-%m-%d-%H-%M-%S-%f")
    dir_name = args.dataset + "-" + dir_name
    path = os.path.join('save', dir_name)
    os.makedirs(path)

    with open(os.path.join(path, 'README.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            print(f'{key}: {value}', file=f)

    train_loaders = []

    if args.dataset == "xd":
        train_loaders, test_loader = make_xd_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length)

    elif args.dataset == "cctv":
        train_loaders, test_loader = make_cctv_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length)

    else:
        train_loaders, test_loader = make_ucf_dataloader(
            args.split_mode, args.clients_num, args.batch_size, args.visual_length)

    model = Model(args.embed_dim, args.visual_length, args.prompt_prefix,
                  args.prompt_postfix, args.visual_width, args.visual_layers,
                  args.visual_head, args.attn_window, device,
                  personalization_type=args.personalization_type,
                  personalized_rank=args.personalized_rank,
                  personalized_alpha=args.personalized_alpha,
                  local_param_policy=args.local_param_policy,
                  pel_projection=args.pel_projection).to(device)

    if args.load_model == 1:
        checkpoint = torch.load(args.checkpoint)
        # strict=False lets a fedpgp run warm-start its shared backbone from a plain
        # fedavg checkpoint (which has no personalized_prompt.* keys); adapters then
        # start from their zero-delta initialization instead of being force-loaded.
        model.load_state_dict(checkpoint, strict=(args.aggregation_method == 'fedavg'))

    if args.algorithm == "FedAvg":
        if use_legacy_server and args.aggregation_method == 'fedavg':
            server = FedAvgServer(args.dataset, train_loaders, test_loader, args.clients_num,
                                  args.global_rounds, args.local_epochs, args.learning_rate,
                                  args.split_mode, args.scheduler_milestones, args.scheduler_rate,
                                  device, model,
                                  use_pel4vad_contrastive=args.use_pel4vad_contrastive,
                                  pel4vad_variant=args.pel4vad_variant,
                                  pel4vad_weight=args.pel4vad_weight,
                                  pel4vad_temperature=args.pel4vad_temperature,
                                  pel4vad_prompts=args.pel4vad_prompts,
                                  pel_diagnostics=args.pel_diagnostics,
                                  pel_options=cli_pel_options(args),
                                  aggregation_weight_unit=args.aggregation_weight_unit)
        elif use_legacy_server:
            server = FedPGPServer(args.dataset, train_loaders, test_loader, args.clients_num,
                                  args.global_rounds, args.local_epochs, args.learning_rate,
                                  args.split_mode, args.scheduler_milestones, args.scheduler_rate,
                                  device, model,
                                  personalized_lr=args.personalized_lr,
                                  use_fedpgp_contrastive=args.use_fedpgp_contrastive,
                                  contrastive_weight=args.contrastive_weight,
                                  contrastive_temperature=args.contrastive_temperature,
                                  use_pel4vad_contrastive=args.use_pel4vad_contrastive,
                                  pel4vad_weight=args.pel4vad_weight,
                                  pel4vad_temperature=args.pel4vad_temperature,
                                  verbose_fed_logging=args.verbose_fed_logging,
                                  pel4vad_variant=args.pel4vad_variant,
                                  pel4vad_prompts=args.pel4vad_prompts,
                                  pel_options=cli_pel_options(args),
                                  aggregation_weight_unit=args.aggregation_weight_unit)
        else:
            server = FedStrategyServer(args.dataset, train_loaders, test_loader, args.clients_num,
                                       args.global_rounds, args.local_epochs, args.learning_rate,
                                       args.split_mode, args.scheduler_milestones, args.scheduler_rate,
                                       device, model,
                                       aggregation_method=args.aggregation_method,
                                       server_optimizer=args.server_optimizer,
                                       server_lr=args.server_lr,
                                       server_beta1=args.server_beta1,
                                       server_beta2=args.server_beta2,
                                       server_tau=args.server_tau,
                                       prox_mu=args.prox_mu if args.aggregation_method == 'fedprox' else 0.0,
                                       personalized_lr=args.personalized_lr,
                                       personalized_weight_decay=args.personalized_weight_decay,
                                       use_fedpgp_contrastive=args.use_fedpgp_contrastive,
                                       contrastive_weight=args.contrastive_weight,
                                       contrastive_temperature=args.contrastive_temperature,
                                       use_pel4vad_contrastive=args.use_pel4vad_contrastive,
                                       pel4vad_weight=args.pel4vad_weight,
                                       pel4vad_temperature=args.pel4vad_temperature,
                                       verbose_fed_logging=args.verbose_fed_logging,
                                       pel4vad_variant=args.pel4vad_variant,
                                       pel4vad_prompts=args.pel4vad_prompts,
                                       pel_diagnostics=args.pel_diagnostics,
                                       pel_options=cli_pel_options(args),
                                       aggregation_weight_unit=args.aggregation_weight_unit)

        server.pel_experiment_logging = args.pel_allow_aggregation_integration
        if args.personalized_checkpoint_dir is not None:
            for i, client in enumerate(server.clients):
                personalized_path = os.path.join(
                    args.personalized_checkpoint_dir, f"client_{i}_personalized.pth")
                client.model.load_personalized_state_dict(torch.load(personalized_path))

        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, timeout=5).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True, timeout=5).strip())
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
        with open(os.path.join(path, 'manifest.json'), 'w', encoding='utf-8') as stream:
            json.dump({'arguments': vars(args), 'server': type(server).__name__,
                       'clients': [type(c).__name__ for c in server.clients],
                       'torch': torch.__version__, 'device': device,
                       'git_revision': revision, 'git_dirty': dirty,
                       'aggregation_weight_unit': args.aggregation_weight_unit,
                       'pel_bank': getattr(server, 'pel_bank_metadata', None)}, stream, indent=2)
        server.train(path)
