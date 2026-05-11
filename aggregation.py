from __future__ import annotations
import torch


def aggregate(hidden_states, attention_mask, input_ids=None, model=None):
    device = hidden_states.device
    selected_layers = list(range(12, 24))
    
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze()
    if real_positions.dim() == 0:
        real_positions = real_positions.unsqueeze(0)
    
    n_tokens = len(real_positions)
    mid = max(1, n_tokens // 2)
    
    features = []
    for layer_idx in selected_layers:
        layer = hidden_states[layer_idx]
        response_tokens = layer[mid:]
        
        if response_tokens.shape[0] > 0:
            n_resp = response_tokens.shape[0]
            weights = torch.linspace(0.5, 1.0, steps=n_resp, device=device)
            weights = weights / weights.sum()
            pooled = (response_tokens * weights.unsqueeze(1)).sum(dim=0)
        else:
            pooled = layer[-1]
        
        features.append(pooled)
    
    base_features = torch.cat(features, dim=0)
    
    response_len = n_tokens - mid
    text_features = torch.tensor([
        response_len / 512.0,
        response_len / max(mid, 1),
        float(n_tokens) / 512.0,
    ], device=device)
    
    return torch.cat([base_features, text_features])


def extract_geometric_features(hidden_states, attention_mask):
    device = hidden_states.device
    features = []
    for layer_idx in [20, 23]:
        layer = hidden_states[layer_idx]
        features.append(torch.norm(layer, p=2).mean().reshape(1))
        features.append(layer.std().reshape(1))
    return torch.cat(features, dim=0)


def aggregation_and_feature_extraction(hidden_states, attention_mask, use_geometric=False, input_ids=None, model=None):
    agg_features = aggregate(hidden_states, attention_mask, input_ids, model)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features
