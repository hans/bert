"""
Evaluate pairwise model transfer performance.
"""

from argparse import ArgumentParser
import itertools
import json
import os.path
import re
from shutil import rmtree, copyfile
import subprocess
from tempfile import mkdtemp


def eval_transfer(model1, model2, **kwargs):
  """
  Evaluate transfer of representations from model1 -> model2.
  """
  model1_name, model2_name = model1, model2
  if "." in model1:
    model1_name = model1[model1.rindex(".") + 1:]
  if "." in model2:
    model2_name = model2[model2.rindex(".") + 1:]

  tmpdir = mkdtemp()

  print("Evaluating transfer %s -> %s" % (model1_name, model2_name))
  model_args = {
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "max_seq_length": 128,
    "ignore_checkpoint_variables": "^output_.+$",
    "train_variables": "^output_.+$",
  }
  model_args.update(kwargs)

  shell_args = [
      "python", "run_classifier.py",
      "--task_name", model2_name,
      "--do_train", "true",
      "--do_eval", "true",
      "--data_dir", os.path.join(args.glue_data_dir, model2_name),
      "--vocab_file", os.path.join(model1, "vocab.txt"),
      "--bert_config_file", os.path.join(model1, "bert_config.json"),
      "--init_checkpoint", os.path.join(model1, "model.ckpt"),
      "--output_dir", tmpdir
  ]
  for k, v in model_args.items():
    shell_args.append("--%s" % k)
    shell_args.append(str(v))

  print("\n".join("%s %s" % pair for pair in zip(shell_args[::2], shell_args[1::2])))
  subprocess.call(shell_args)

  out = {"model1": {"path": model1, "name": model1_name},
         "model2": {"path": model2, "name": model2_name},
         "args": model_args,
         "results": {}}
  with open(os.path.join(tmpdir, "eval_results.txt"), "r") as results_f:
    for k, v in re.findall(r"^([\w_]+)\s*=\s*([\d.-]+)$", results_f.read(), re.M):
      out["results"][k] = float(v)

  with open(os.path.join(args.out_dir, "results-%s-%s.json" % (model1_name, model2_name)), "w") as out_f:
    json.dump(out, out_f, indent=2)

  rmtree(tmpdir)


if __name__ == '__main__':
  p = ArgumentParser()
  p.add_argument("glue_data_dir")
  p.add_argument("model_path", nargs="+")
  p.add_argument("-o", "--out_dir", default=".")
  p.add_argument("--max_train_steps", type=int)

  args = p.parse_args()
  kwargs = {}
  if args.max_train_steps is not None:
    kwargs["max_train_steps"] = args.max_train_steps

  for model1, model2 in itertools.permutations(args.model_path, 2):
    eval_transfer(model1, model2, **kwargs)
