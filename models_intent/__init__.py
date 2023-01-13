from models_intent.relation.rn import RelationModel, CNN_MLP
 
_MODELS_ = {
    'rn': RelationModel, # relational model
    'cnn_mlp': CNN_MLP,
}

def make_model(cfg):
    model = _MODELS_[cfg.method]
    return model(cfg.model)

