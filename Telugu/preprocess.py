import yaml
from preprocessor.preprocessor import Preprocessor

cfg = yaml.load(open('config.yaml', encoding='utf-8'), Loader=yaml.FullLoader)
Preprocessor(cfg['preprocessing']).build_from_path()
