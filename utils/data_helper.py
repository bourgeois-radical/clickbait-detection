# built-in libraries
import pathlib
from typing import List, Dict, Tuple

# installed libraries
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter, OrderedDict


def get_data_from_file(path: pathlib.WindowsPath) -> List[str]:
    """Reads data line by line

    Parameters
    ----------
    path : pathlib.WindowsPath
        Path to the data to be read
    Returns
    -------
    data : List[str]
        Single sentences
    """

    with open(path, encoding='utf8') as f:
        data = [line.rstrip() for line in f]
    return data


def building_vocab(sentences: List[str]) -> Dict[str, int]:
    """Builds vocabulary sorted by words' frequency (in descending order).

    Parameters
    ----------
    sentences : List[str]
        Data

    Returns
    -------
    vocab : Dict[str, int]
        Vocabulary sorted by frequency in descending order.
    """
    word_token_counts = Counter()

    for sentence in sentences:
        word_token_counts.update(sentence)

    sorted_by_freq_tuples = sorted(word_token_counts.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    dict_list = [('<pad>', 0), ('<unk>', 1)]
    for idx, word in enumerate(ordered_dict):
        dict_list.append((word, idx + 2))

    print(f"Vocab's length: {len(dict_list)}")

    vocab = dict(dict_list)

    return vocab


class ClickbaitDataSet(Dataset):
    """Dataset for DataLoader"""

    def __init__(self, X: List[str], y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.X[idx], torch.tensor(self.y[idx], dtype=torch.float32)

