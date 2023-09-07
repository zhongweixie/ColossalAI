import copy

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizer

from colossalai.booster.plugin.dp_plugin_base import DPPluginBase


class GLUEDataBuilder:

    task_text_field_map = {"super_natural_instructions": ["prompt", "completion"]}

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "attention_mask",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        plugin: DPPluginBase,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.plugin = plugin

        self.text_fields = self.task_text_field_map[task_name]
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.setup()

    def setup(self):
        self.dataset = datasets.load_dataset("yizhongw/self_instruct", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return self.plugin.prepare_dataloader(self.dataset["train"],
                                              batch_size=self.train_batch_size,
                                              shuffle=True,
                                              drop_last=True)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return self.plugin.prepare_dataloader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                self.plugin.prepare_dataloader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return self.plugin.prepare_dataloader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [
                self.plugin.prepare_dataloader(self.dataset[x], batch_size=self.eval_batch_size)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(texts_or_text_pairs,
                                                    max_length=self.max_seq_length,
                                                    padding='max_length',
                                                    truncation=True)

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = copy.deepcopy(features["input_ids"])

        return features
