from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.clip import clip
from utils.prompt_net import PromptLearner
from utils.personalized_prompt import build_personalized_prompt, is_personalized_param_name


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float = 0, attn_mask: torch.Tensor = None, ):
        super(TransformerEncoder, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Model(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 visual_length: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 visual_width: int,
                 visual_layers: int,
                 visual_head: int,
                 attn_window: int,
                 device,
                 personalization_type: str = "none",
                 personalized_rank: int = 4,
                 personalized_alpha: float = 1.0,
                 local_param_policy: str = None,
                 pel_projection: bool = False):
        super().__init__()

        self.visual_length = visual_length
        self.embed_dim = embed_dim
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.attn_window = attn_window
        self.device = device

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.temporal = TransformerEncoder(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
        )

        self.prompt_learner = PromptLearner()

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.global_text_prompt_embeddings = nn.Embedding(77, self.embed_dim)
        self.dtype = self.clipmodel.dtype
        self.initialize_parameters()
        if pel_projection:
            # Identity initialization without consuming the base model's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.pel_projection = nn.Linear(visual_width, visual_width, bias=False)
                with torch.no_grad():
                    self.pel_projection.weight.copy_(torch.eye(visual_width))

        self.personalization_type = personalization_type
        n_ctx, prompt_dim = self._trace_prompt_shape(visual_width)
        self.personalized_prompt = build_personalized_prompt(
            personalization_type, n_ctx, prompt_dim, personalized_rank, personalized_alpha)

        self.local_param_policy = self._resolve_local_param_policy(
            local_param_policy, personalization_type)
        self._local_param_names = self._compute_local_param_names(self.local_param_policy)

    def initialize_parameters(self):
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    @staticmethod
    def _resolve_local_param_policy(policy, personalization_type):
        """Decide which trainable parameters stay on the client instead of being averaged.

        'adapter' is forced on whenever a personalized adapter exists: aggregating
        A_i/B_i would average away the very thing that makes them client-specific.
        """
        if policy in (None, ""):
            policy = "adapter" if personalization_type != "none" else "none"
        if personalization_type != "none" and "adapter" not in policy:
            policy = "adapter" if policy == "none" else policy + "+adapter"
        return policy

    def _compute_local_param_names(self, policy):
        """Resolve the policy string into a concrete set of parameter names."""
        names = set()
        if policy == "none":
            return names
        parts = set(policy.split("+"))

        if "adapter" in parts:
            names |= {n for n, _ in self.named_parameters() if is_personalized_param_name(n)}

        if "norm" in parts:
            # FedBN (Li et al., ICLR 2021) keeps normalization layers local so each
            # client can absorb its own feature shift. This model has no BatchNorm, so
            # the analogue is the LayerNorm affine parameters. Located by module type
            # rather than by name matching, since they live under several different
            # prefixes (temporal.*.ln_1/ln_2, prompt_learner PreNorm/FeedForward norms).
            # Frozen CLIP LayerNorms are skipped by the requires_grad check.
            for module_name, module in self.named_modules():
                if isinstance(module, nn.LayerNorm):
                    for param_name, param in module.named_parameters(recurse=False):
                        if param.requires_grad:
                            names.add(f"{module_name}.{param_name}" if module_name else param_name)

        return names

    def _is_local_param(self, name):
        return name in self._local_param_names

    def _trace_prompt_shape(self, visual_width: int):
        """Empirically trace the (n_ctx, prompt_dim) shape produced by the dynamic
        prompt generator instead of assuming it, since it depends on PromptTranslator's
        internal (possibly reconfigured) dimensions."""
        with torch.no_grad():
            dummy_text = torch.zeros(1, self.embed_dim)
            dummy_visual = torch.zeros(2, self.visual_length, visual_width)
            dummy_context = self.prompt_learner(dummy_text, dummy_visual)
        return dummy_context.shape[1], dummy_context.shape[2]

    def shared_parameters(self):
        """Trainable parameters that are communicated to the server and averaged."""
        return [p for n, p in self.named_parameters()
                if p.requires_grad and not self._is_local_param(n)]

    def personalized_parameters(self):
        """Trainable parameters that stay on this client and are never aggregated."""
        return [p for n, p in self.named_parameters()
                if p.requires_grad and self._is_local_param(n)]

    def get_shared_state_dict(self):
        return OrderedDict(
            (n, p.data.clone()) for n, p in self.named_parameters()
            if p.requires_grad and not self._is_local_param(n))

    def get_personalized_state_dict(self):
        return OrderedDict(
            (n, p.data.clone()) for n, p in self.named_parameters()
            if p.requires_grad and self._is_local_param(n))

    # 'local' reads better than 'personalized' now that the same mechanism also
    # carries FedBN-style norm parameters; both names refer to the same split.
    local_parameters = personalized_parameters
    get_local_state_dict = get_personalized_state_dict

    def _load_partial_state_dict(self, partial_state):
        state_dict = self.state_dict()
        for key, value in partial_state.items():
            state_dict[key] = value.data.clone()
        self.load_state_dict(state_dict)

    def load_shared_state_dict(self, shared_state):
        self._load_partial_state_dict(shared_state)

    def load_personalized_state_dict(self, personalized_state):
        self._load_partial_state_dict(personalized_state)

    load_local_state_dict = load_personalized_state_dict

    def encode_video(self, images):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x = self.temporal(images)
        x = x.permute(1, 0, 2)
        return x

    def get_tokenized_classnames(self, classes):
        prompts = [" ".join(["X"] * 4) + " " + name + "." for name in classes]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clipmodel.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)

        return embedding, tokenized_prompts

    def _build_text_features(self, context, prompt_vectors, tokenized_prompts):
        prompt_vectors = torch.cat(
            [
                prompt_vectors[:, :1],
                context[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),
                prompt_vectors[:, 1 + context.shape[1]:],
            ],
            dim=1,
        )
        return self.clipmodel.encode_text(prompt_vectors, tokenized_prompts)

    def encode_text_prompt(self, text, visual, use_personalization: bool = True,
                            need_contrastive_aux: bool = False):
        classes = [name.replace("_", " ") for name in text]
        class_tokens = torch.cat([clip.tokenize(p) for p in classes])
        class_tokens = class_tokens.to(self.device)
        with torch.no_grad():
            class_features = self.clipmodel.encode_text_original(class_tokens)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)

        context_embedding = class_features
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(classes)

        # G_theta_global(T_global, V_i(x)): existing sample-dependent dynamic prompt.
        context = self.prompt_learner(context_embedding, visual)

        if self.personalized_prompt is not None and use_personalization:
            # P_i(x) = G_theta_global(T_global, V_i(x)) + A_i @ B_i
            personalized_context = context + self.personalized_prompt().unsqueeze(0)
        else:
            personalized_context = context

        text_features = self._build_text_features(personalized_context, prompt_vectors, tokenized_prompts)

        if need_contrastive_aux:
            assert self.personalized_prompt is not None and use_personalization, \
                "contrastive aux requires an active personalized_prompt adapter"
            text_features_global = self._build_text_features(context, prompt_vectors, tokenized_prompts)
            aux = {
                "class_features": class_features,
                "text_features_global": text_features_global,
                "text_features_personalized": text_features,
            }
            return text_features, aux

        return text_features

    def forward(self, visual, text, lengths, use_personalization: bool = True,
                return_contrastive: bool = False, return_pel4vad_aux: bool = False):
        visual_features = self.encode_video(visual)

        if return_contrastive:
            text_features, aux = self.encode_text_prompt(
                text, visual_features, use_personalization=True, need_contrastive_aux=True)
        else:
            text_features = self.encode_text_prompt(
                text, visual_features, use_personalization=use_personalization)
            aux = None

        if return_pel4vad_aux:
            # Independent of return_contrastive/personalization -- pel4vad_contrastive_loss
            # (utils/tools.py) only needs the per-snippet visual encoding and the per-class
            # text embeddings actually used to produce `logits` below, whatever forward pass
            # produced them.
            aux = aux or {}
            aux["visual_features"] = visual_features
            aux["text_features"] = text_features

        text_features_b = text_features.unsqueeze(0)
        text_features_b = text_features_b.expand(
            visual_features.shape[0], text_features_b.shape[1], text_features_b.shape[2])

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_b / text_features_b.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)

        logits = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype)
        logits = logits * self.clipmodel.logit_scale.exp()

        if return_contrastive or return_pel4vad_aux:
            return logits, aux
        return logits
