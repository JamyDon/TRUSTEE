# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import json
import logging
import traceback
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.import_utils import load_extern_object

logger = logging.getLogger(__name__)


class AgentDataset(RLHFDataset):
    """
    Load and preprocess agent data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.query_key = config.get("query_key", "query")
        self.tools_key = config.get("tools_key", "tools")
        super().__init__(data_files, tokenizer, config, processor, max_samples)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer

            def doc2len(doc) -> int:
                try:
                    messages = self._build_messages(doc)
                    apply_kwargs = dict(**self.apply_chat_template_kwargs)
                    apply_kwargs["tools"] = json.loads(doc[self.tools_key])
                    return len(
                        tokenizer.apply_chat_template(messages, add_generation_prompt=True, **apply_kwargs)
                    )
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    traceback.print_exc()
                    return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filtered dataset len: {len(dataframe)}")
        return dataframe

    def _build_messages(self, example: dict):
        """Build messages from example.

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages.
        """
        query: str = example[self.query_key]
        messages: list = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]

        return messages

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item]
        row_dict["raw_prompt"] = self._build_messages(row_dict)
        row_dict["tools"] = json.loads(row_dict.get(self.tools_key, "[]"))

        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        row_dict["index"] = row_dict.get("index", 0)
        return row_dict


def get_dataset_class(data_config: DictConfig):
    """Get Agent dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default AgentDataset class if no custom class is specified
        dataset_cls = AgentDataset

    return dataset_cls