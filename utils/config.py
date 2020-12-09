import toml
from easydict import EasyDict
def get_config():
    with open('config.toml', 'r', encoding='utf-8') as f:
        config = toml.load(f)
        return EasyDict(config)
