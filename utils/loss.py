import torch

LOG_EPSILON = 1e-5


"""

single positive multi-label learning (SPMLL) loss functions)

"""

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)


def expected_positive_regularizer(preds, expected_num_pos = 1.5, norm='2'):
    # Assumes predictions in [0,1].
    if norm == '1':
        reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    elif norm == '2':
        reg = (preds.sum(1).mean(0) - expected_num_pos)**2
    else:
        raise NotImplementedError
    return reg

'''
loss functions
'''

def loss_epr(batch, P):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss w.r.t. observed positives:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    # compute regularizer: 
    reg_loss = expected_positive_regularizer(preds, P['expected_num_pos'], norm='2') / (P['num_classes'] ** 2)
    return loss_mtx, reg_loss

loss_functions = {
    'epr': loss_epr,
}

'''
top-level wrapper
'''

def compute_batch_loss(batch, P):

    """
    배치(batch) 단위의 손실(loss)을 계산하는 메인 함수입니다.

    Args:
        batch (dict): 데이터 로더(dataloader)로부터 받은 현재 배치 정보.
                      아래와 같은 키(key)를 포함해야 합니다:
                      - 'preds': 모델의 예측 값 (확률). torch.Tensor [B, C]
                      - 'label_vec_obs': 실제 정답 라벨 (원-핫 벡터). torch.Tensor [B, C]
        P (dict):     손실 함수에 필요한 하이퍼파라미터 및 설정 값.
                      아래와 같은 키(key)를 포함해야 합니다:
                      - 'loss': 사용할 손실 함수의 이름 (예: 'epr'). str
                      - 'num_classes': 전체 클래스의 수. int
                      - 'expected_num_pos': EPR 손실 함수를 위한 기대 양성 라벨 수. float
    
    Returns:
        dict: 기존 batch 딕셔너리에 아래 키(key)들이 추가된 딕셔너리:
              - 'loss_tensor': 최종 계산된 손실 값. torch.Tensor
              - 'reg_loss_np': 정규화(regularization) 손실 값. numpy.float
              - 'loss_np': 최종 손실 값. numpy.float
    """
    
    assert batch['preds'].dim() == 2
    
    batch_size = int(batch['preds'].size(0))
    num_classes = int(batch['preds'].size(1))
    
    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(batch['preds'])
    
    # input validation:
    assert torch.max(batch['label_vec_obs']) <= 1
    assert torch.min(batch['label_vec_obs']) >= -1
    assert batch['preds'].size() == batch['label_vec_obs'].size()
    assert P['loss'] in loss_functions
    
    # validate predictions:
    assert torch.max(batch['preds']) <= 1
    assert torch.min(batch['preds']) >= 0
    
    # compute loss for each image and class:
    loss_mtx, reg_loss = loss_functions[P['loss']](batch, P)
    main_loss = (loss_mtx / loss_denom_mtx).sum()
    
    if reg_loss is not None:
        batch['loss_tensor'] = main_loss + reg_loss
        batch['reg_loss_np'] = reg_loss.clone().detach().cpu().numpy()
    else:
        batch['loss_tensor'] = main_loss
        batch['reg_loss_np'] = 0.0
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()
    
    return batch
