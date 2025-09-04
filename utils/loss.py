import torch
import torch.nn as nn # Import nn module

LOG_EPSILON = 1e-5

class SPMLLLoss(nn.Module):
    """
    Single Positive Multi-Label Learning (SPMLL) Loss function.
    Calculates loss for observed positive labels and applies a regularizer
    to guide the expected number of positive predictions.
    """
    def __init__(self, num_classes: int, expected_num_pos: float, norm: str = '2'):
        super().__init__()
        self.num_classes = num_classes
        self.expected_num_pos = expected_num_pos
        self.norm = norm

    def _neg_log(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates negative logarithm, with epsilon for numerical stability."""
        return -torch.log(x + LOG_EPSILON)

    def _expected_positive_regularizer(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected positive regularizer.
        Assumes predictions are in [0,1].
        """
        if self.norm == '1':
            reg = torch.abs(preds.sum(1).mean(0) - self.expected_num_pos)
        elif self.norm == '2':
            reg = (preds.sum(1).mean(0) - self.expected_num_pos)**2
        else:
            raise NotImplementedError(f"Normalization '{self.norm}' not implemented.")
        return reg

    def forward(self, preds: torch.Tensor, observed_labels: torch.Tensor):
        """
        Computes the SPMLL loss for a batch.

        Args:
            preds (torch.Tensor): Model's predicted probabilities. [B, C]
            observed_labels (torch.Tensor): Observed ground truth labels (one-hot). [B, C]

        Returns:
            tuple: (loss_matrix, regularization_loss)
        """
        # Input validation for observed_labels
        assert torch.min(observed_labels) >= 0, "Observed labels must be non-negative."
        assert preds.size() == observed_labels.size(), "Predictions and labels must have the same shape."

        # Compute loss w.r.t. observed positives:
        loss_mtx = torch.zeros_like(observed_labels)
        loss_mtx[observed_labels == 1] = self._neg_log(preds[observed_labels == 1])

        # Compute regularizer:
        reg_loss = self._expected_positive_regularizer(preds) / (self.num_classes ** 2)
        return loss_mtx, reg_loss


def compute_batch_loss(batch: dict, P: dict) -> dict:
    """
    배치(batch) 단위의 손실(loss)을 계산하는 메인 함수입니다.

    Args:
        batch (dict): 데이터 로더(dataloader)로부터 받은 현재 배치 정보.
                      아래와 같은 키(key)를 포함해야 합니다:
                      - 'preds': 모델의 예측 값 (확률). torch.Tensor [B, C]
                      - 'label_vec_obs': 실제 정답 라벨 (원-핫 벡터). torch.Tensor [B, C]
        P (dict):     손실 함수에 필요한 하이퍼파라미터 및 설정 값.
                      아래와 같은 키(key)를 포함해야 합니다:
                      - 'num_classes': 전체 클래스의 수. int
                      - 'expected_num_pos': SPMLL 손실 함수를 위한 기대 양성 라벨 수. float.
    Returns:
        dict: 기존 batch 딕셔너리에 아래 키(key)들이 추가된 딕셔너리:
              - 'loss_tensor': 최종 계산된 손실 값. torch.Tensor
              - 'reg_loss_np': 정규화(regularization) 손실 값. numpy.float
              - 'loss_np': 최종 손실 값. numpy.float
    """

    assert batch['preds'].dim() == 2, "Predictions must be 2-dimensional (batch_size, num_classes)."

    batch_size = int(batch['preds'].size(0))
    num_classes = int(batch['preds'].size(1))

    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(batch['preds'])

    # input validation:
    assert torch.max(batch['label_vec_obs']) <= 1, "Observed labels must be binary (0 or 1)."
    assert torch.min(batch['label_vec_obs']) >= -1, "Observed labels must be -1, 0, or 1." # Original comment, keeping for consistency
    assert batch['preds'].size() == batch['label_vec_obs'].size(), "Predictions and observed labels must have the same size."
    # Removed: assert P['loss'] in loss_functions

    # validate predictions:
    assert torch.max(batch['preds']) <= 1, "Predictions must be probabilities (<= 1)."
    assert torch.min(batch['preds']) >= 0, "Predictions must be probabilities (>= 0)."

    # Instantiate the SPMLLLoss class
    # Assuming P contains 'num_classes' and 'expected_num_pos'
    spmll_criterion = SPMLLLoss(
        num_classes=P['num_classes'],
        expected_num_pos=P['expected_num_pos']
    )

    # Compute loss for each image and class using the class instance
    loss_mtx, reg_loss = spmll_criterion(batch['preds'], batch['label_vec_obs'])
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    if reg_loss is not None:
        batch['loss_tensor'] = main_loss + reg_loss
        batch['reg_loss_np'] = reg_loss.clone().detach().cpu().numpy()
    else:
        batch['loss_tensor'] = main_loss
        batch['reg_loss_np'] = 0.0
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()

    return batch