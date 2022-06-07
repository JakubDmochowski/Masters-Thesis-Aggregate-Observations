import torch


class RegressionQualityMeasure:
    def MeanSquaredError(expected: torch.tensor, predicted: torch.tensor):
        # expected and predicted args are tensors shaped(entries, values)
        return
