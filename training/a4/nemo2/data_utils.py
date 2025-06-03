"""Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer


def get_dataset_mock(config):
  return llm.MockDataModule(
      seq_length=config.max_length,
      global_batch_size=config.global_train_batch_size,
      micro_batch_size=config.per_device_train_batch_size,
  )


def get_dataset_c4(config):
  import os

  INDEX_MAPPING_DIR = "/cache/dataset"
  os.makedirs(INDEX_MAPPING_DIR, exist_ok=True)
  tokenizer = AutoTokenizer(pretrained_model_name="/app/tokenizer")

  dataset_train = [
      os.path.join(config.dataset.train_dataset_path, "c4-train.en_6_text_document"),
      os.path.join(config.dataset.train_dataset_path, "c4-train.en_7_text_document"),
  ]

  dataset_valid = [
      os.path.join(
          config.dataset.eval_dataset_path, "c4-validation.en_text_document"
      )
  ]

  return llm.PreTrainingDataModule(
      paths={
          "train": dataset_train,
          "validation": dataset_valid,
          "test": dataset_valid,
      },
      seq_length=config.max_length,
      global_batch_size=config.global_train_batch_size,
      micro_batch_size=config.per_device_train_batch_size,
      tokenizer=tokenizer,
      index_mapping_dir=INDEX_MAPPING_DIR,
      num_workers=2,
      persistent_workers=True,
      seed=config.seed,

      # Option to reset the position IDs in the dataset at an interval.
      #reset_position_ids=False,
      # Option to reset the attention mask from the dataset.
      #reset_attention_mask=False,
      # Option to enable the EOD mask loss.
      #eod_mask_loss=False,
      # Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
      #rampup_batch_size=None,
  )