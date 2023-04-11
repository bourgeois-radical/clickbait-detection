# built-in libraries
from typing import Dict

# installed libraries
import torch
import numpy as np

# local libraries
from utils.models import RNN
from utils.preprocessing import Preprocess


class ClickbaitClassifier:
    """Testing the model on a single random sentence"""

    def __init__(self, model: RNN, preprocessor: Preprocess, vocab: Dict[str, int]) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.vocab = vocab

    def __call__(self, sentence: str) -> np.float:
        """Returns the probability of a sentence being clickbait"""

        preprocessed_sentence = self.preprocessor([sentence])[0]
        sentence_as_tokens_id = [self.vocab[token] if token in self.vocab else self.vocab["<unk>"]
                                 for token in preprocessed_sentence]
        sentence_as_tokens_id = torch.tensor(sentence_as_tokens_id, requires_grad=False).view(1, -1)
        length = torch.tensor([sentence_as_tokens_id.size(1)], requires_grad=False)
        out = self.model(sentence_as_tokens_id, length)
        out = out.detach().numpy()[0, 0]

        return out

