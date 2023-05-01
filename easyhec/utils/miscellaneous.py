def save_config(cfg, path):
    with open(path, 'w') as f:
        f.write(cfg.dump())
