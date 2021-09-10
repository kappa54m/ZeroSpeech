def fix_config(cfg):
    """ Temporary fix for hydra configs """
    for a in ['dataset', 'preprocessing', 'model', 'training']:
        if hasattr(cfg, a):
            setattr(cfg, a, getattr(getattr(cfg, a), a))
    return cfg
