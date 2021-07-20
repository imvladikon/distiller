import torch


def probability_shift(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    From "Preparing Lessons: Improve Knowledge Distillation with Better Supervision"
    https://arxiv.org/abs/1911.07471. Swaps argmax and correct label in logits.

    Args:
        logits: logits from teacher model
        labels: correct labels

    Returns:
        smoothed labels
    """
    labels_ = labels.long()
    argmax_values, argmax_labels = logits.max(-1)
    arange_indx = torch.arange(logits.size(0))
    logits[arange_indx, argmax_labels] = logits[arange_indx, labels_]
    logits[arange_indx, labels_] = argmax_values
    del labels_
    return logits


if __name__ == '__main__':
    import numpy as np
    import torch

    def test_simple():
        logits = torch.tensor(np.array([[1, 2, 1], [2, 1, 1], [1, 1, 2]]), dtype=torch.float32)
        labels = torch.tensor(np.array([0, 0, 0]))
        s_logits = probability_shift(logits, labels)
        target_tensor = torch.tensor(np.array([[2, 1, 1], [2, 1, 1], [2, 1, 1]]), dtype=torch.float32)
        assert torch.isclose(s_logits, target_tensor).type(torch.long).sum() == 9


    def test_random():
        logits = torch.randn(100, 100, dtype=torch.float32)
        labels = torch.zeros(100, dtype=torch.long)
        s_logits = probability_shift(logits, labels)
        assert (s_logits.argmax(-1) == labels).type(torch.long).sum() == 100

    test_simple()
    test_random()
