import argparse
from .infer import blind_source_separation

parser = argparse.ArgumentParser("restore sound for each speaker")
parser.add_argument("-i", "--input_file", type=str, help="the mixed audio file")
parser.add_argument("-m", "--model_dir", type=str, help="the directory \
where the trained model is stored")
parser.add_argument("-o", "--output", type=str, help="the directory \
where the estimated sources should be stored")
args = parser.parse_args()


def infer():
  blind_source_separation(args.input_file, args.model_dir, args.output)
