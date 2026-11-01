import argparse
import yaml
from preprocessor.preprocessor import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    Preprocessor(cfg["preprocessing"]).build_from_path()
