from .decision_transformer import DecisionTransformer


MODEL_REGISTRY = {
    "decision_transformer": DecisionTransformer,
}

def build_model(cfg, state_dim: int, act_dim: int):
    """
    Build model from registry based on config.
    This mimics how HuggingFace transformers and RLlib handle model loading.
    """
    model_name = cfg.model.name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{cfg.model.name}' not found in MODEL_REGISTRY")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=cfg.model.n_blocks,
        h_dim=cfg.model.embed_dim,
        context_len=cfg.model.context_len,
        n_heads=cfg.model.n_heads,
        drop_p=cfg.model.dropout_p,
    )

    return model.to(cfg.device)