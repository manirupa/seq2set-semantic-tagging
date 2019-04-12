# python 2
from multisense_prob_fasttext import multift
import argparse

parser = argparse.ArgumentParser(
    description='multi ft',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', nargs='?', type=str, help='model files')
args = parser.parse_args()

ft = multift.MultiFastText(args.i)
