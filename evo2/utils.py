MODEL_NAMES = [
    'evo2_40b',
    'evo2_7b',
    'evo2_40b_base',
    'evo2_7b_base',
    'evo2_1b_base',
]

HF_MODEL_NAME_MAP = {
    'evo2_40b': 'arcinstitute/evo2_40b',
    'evo2_7b': 'arcinstitute/evo2_7b',
    'evo2_40b_base': 'arcinstitute/evo2_40b_base',
    'evo2_7b_base': 'arcinstitute/evo2_7b_base',
    'evo2_1b_base': 'arcinstitute/evo2_1b_base',
}

CONFIG_MAP = {
    'evo2_7b': 'evo2/configs/evo2-7b-1m.yml',
    'evo2_40b': 'evo2/configs/evo2-40b-1m.yml',
    'evo2_7b_base': 'evo2/configs/evo2-7b-8k.yml',
    'evo2_40b_base': 'evo2/configs/evo2-40b-8k.yml',
    'evo2_1b_base': 'evo2/configs/evo2-1b-8k.yml',
}
