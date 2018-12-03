# Process data stored in an encodings.jsonl

from argparse import ArgumentParser
import json
import os

import numpy as np


p = ArgumentParser()
p.add_argument("-i", "--encodings_file", default="encodings.jsonl")
p.add_argument("-o", "--out_file", default="encodings.npy")
p.add_argument("-l", "--layer", type=int, default=-1)

args = p.parse_args()

with open(args.encodings_file, "r") as encs_f:
  encodings = [json.loads(line) for line in encs_f if line.strip()]


encodings_out = {}
for encoding in encodings:
  t_encodings = []
  for tidx, features in enumerate(encoding["features"]):
    layer = next(l for l in features["layers"] if l["index"] == args.layer)
    t_encodings.append(layer["values"])

  encodings_out[encoding["linex_index"]] = np.mean(t_encodings, axis=0)


assert set(encodings_out.keys()) == set(range(len(encodings_out)))
encodings_out = sorted(encodings_out.items(), key=lambda a: a[0])
encodings_out = [v for k, v in encodings_out]
np.save(args.out_file, np.array(encodings_out))

print("Removing original file " + args.encodings_file)
os.remove(args.encodings_file)
