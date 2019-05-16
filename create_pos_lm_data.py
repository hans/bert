# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
import itertools
import mmap
import random

import tensorflow as tf
from tqdm import tqdm, trange

import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_string("masking_policy", "default",
    ("Policy for generating masked LM examples. One of `default`, `maskonly`, "
      "`randomonly`. `default` cues the mixed strategy described in the BERT "
      "paper. `maskonly` specifies that the masked word should always be "
      "replaced with a MASK token. `randomonly` specifies that the masked word "
      "should be replaced with a random vocabulary token."))

flags.DEFINE_bool(
    "shuffle_seqs", False,
    "If True, randomly shuffle word sequences.")


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = instance.masked_lm_labels
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              masking_fn, shuffle_seqs, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]
  all_tagged_documents = [[]]
  all_tags = set()

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    num_lines = get_num_lines(input_file)

    with tf.gfile.GFile(input_file, "r") as reader:
      expect_pos = False
      for line in tqdm(reader, total=num_lines, desc="Reading lines"):
        line = tokenization.convert_to_unicode(line)
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
          all_tagged_documents.append([])
          expect_pos = False
          continue

        if expect_pos:
          tags = line.strip().split(" ")
          all_tagged_documents[-1].append(tags)
          all_tags |= set(tags)
          expect_pos = False
        else:
          tokens = tokenizer.tokenize(line)
          if tokens:
            all_documents[-1].append(tokens)
          # POS tags should follow on next line.
          expect_pos = True

  # Remove empty documents and shuffle
  retain = [i for i, (tokens, tagged) in enumerate(zip(all_documents, all_tagged_documents))
            if tokens and tagged and len(tokens) == len(tagged)]
  rng.shuffle(retain)

  print(len(all_documents), len(all_tagged_documents))
  assert len(all_documents) == len(all_tagged_documents)
  all_documents = [all_documents[i] for i in retain]
  all_tagged_documents = [all_tagged_documents[i] for i in retain]

  vocab_words = list(tokenizer.vocab.keys())
  all_tags = sorted(all_tags)
  tf.logging.info("Available tags: %s" % ",".join(all_tags))

  instances = []
  for _ in range(dupe_factor):
    for document_index in trange(len(all_documents), desc="Reading documents"):
      instances.extend(
          create_instances_from_document(
              all_documents, all_tagged_documents, document_index, max_seq_length,
              short_seq_prob, masked_lm_prob, vocab_words, all_tags, shuffle_seqs,
              masking_fn, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
    all_documents, all_tagged_documents, document_index, max_seq_length,
    short_seq_prob, masked_lm_prob, vocab_words, tags, shuffle_seqs, masking_fn, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]
  tagged_document = all_tagged_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk, current_tagged_chunk = [], []
  current_length = 0
  i = 0
  with tqdm(total=len(document), desc="Creating examples") as pbar:
    pbar.update(i - pbar.n)

    while i < len(document):
      segment, segment_tags = document[i], tagged_document[i]

      current_chunk.append(segment)
      current_tagged_chunk.append(segment_tags)
      current_length += len(segment)

      if i == len(document) - 1 or current_length >= target_seq_length:
        if current_chunk:
          # `a_end` is how many segments from `current_chunk` go into the `A`
          # (first) sentence.
          a_end = 1
          if len(current_chunk) >= 2:
            a_end = rng.randint(1, len(current_chunk) - 1)

          tokens_a, tags_a = [], []
          for j in range(a_end):
            tokens_a.extend(current_chunk[j])
            tags_a.extend(current_tagged_chunk[j])

          tokens_b, tags_b = [], []
          # Random next
          is_random_next = False
          if len(current_chunk) == 1 or rng.random() < 0.5:
            is_random_next = True
            target_b_length = target_seq_length - len(tokens_a)

            # This should rarely go for more than one iteration for large
            # corpora. However, just to be careful, we try to make sure that
            # the random document is not the same as the document
            # we're processing.
            for _ in range(10):
              random_document_index = rng.randint(0, len(all_documents) - 1)
              if random_document_index != document_index:
                break

            random_document = all_documents[random_document_index]
            random_tagged_document = all_tagged_documents[random_document_index]
            random_start = rng.randint(0, len(random_document) - 1)
            for j in range(random_start, len(random_document)):
              tokens_b.extend(random_document[j])
              tags_b.extend(random_tagged_document[j])
              if len(tokens_b) >= target_b_length:
                break
            # We didn't actually use these segments so we "put them back" so
            # they don't go to waste.
            num_unused_segments = len(current_chunk) - a_end
            i -= num_unused_segments
          # Actual next
          else:
            is_random_next = False
            for j in range(a_end, len(current_chunk)):
              tokens_b.extend(current_chunk[j])
              tags_b.extend(current_tagged_chunk[j])
          truncate_seq_pair(tokens_a, tags_a, tokens_b, tags_b, max_num_tokens, rng)

          assert len(tokens_a) >= 1
          assert len(tokens_b) >= 1

          if shuffle_seqs:
            a_idxs, b_idxs = list(range(len(tokens_a))), list(range(len(tokens_b)))
            rng.shuffle(a_idxs)
            rng.shuffle(b_idxs)

            tokens_a = [tokens_a[idx] for idx in a_idxs]
            tags_a = [tags_a[idx] for idx in a_idxs]
            tokens_b = [tokens_b[idx] for idx in b_idxs]
            tags_b = [tags_b[idx] for idx in b_idxs]

          tokens, tags = ["[CLS]"], [None]
          segment_ids = [0]
          for token, tag in zip(tokens_a, tags_a):
            tokens.append(token)
            tags.append(tag)
            segment_ids.append(0)

          tokens.append("[SEP]")
          tags.append(None)
          segment_ids.append(0)

          for token, tag in zip(tokens_b, tags_b):
            tokens.append(token)
            tags.append(tag)
            segment_ids.append(1)

          tokens.append("[SEP]")
          tags.append(None)
          segment_ids.append(1)

          (tokens, masked_lm_positions,
          masked_lm_labels) = masking_fn(tokens, tags, masked_lm_prob,
                                         vocab_words=vocab_words,
                                         all_tags=tags,
                                         rng=rng)
          instance = TrainingInstance(
              tokens=tokens,
              segment_ids=segment_ids,
              is_random_next=is_random_next,
              masked_lm_positions=masked_lm_positions,
              masked_lm_labels=masked_lm_labels)
          instances.append(instance)
        current_chunk = []
        current_length = 0
      i += 1

  return instances


def truncate_seq_pair(tokens_a, tags_a, tokens_b, tags_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    if len(tokens_a) > len(tokens_b):
      trunc_tokens = tokens_a
      trunc_tags = tags_a
    else:
      trunc_tokens = tokens_b
      trunc_tags = tags_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, tags, masked_lm_prob,
                                 max_predictions_per_seq=20,
                                 vocab_words=[],
                                 all_tags=[],
                                 masking_policy="default",
                                 rng=None):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    if masking_policy == "default":
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
    elif masking_policy == "randomonly":
      masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
    elif masking_policy == "maskonly":
      masked_token = "[MASK]"

    output_tokens[index] = masked_token
    label = all_tags.index(tags[index])

    masked_lms.append(MaskedLmInstance(index=index, label=label))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def main(_):
  print("Shuffle seqs:", FLAGS.shuffle_seqs)

  if FLAGS.masking_policy not in ["default", "maskonly", "randomonly"]:
    print("Invalid masking policy:", FLAGS.masking_policy)
    return 1

  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  masking_fn = partial(create_masked_lm_predictions,
                       max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                       masking_policy=FLAGS.masking_policy)

  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, masking_fn,
      FLAGS.shuffle_seqs, rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
