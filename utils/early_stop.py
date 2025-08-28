import numpy as np

class EarlyStopping():
    def __init__(self, patience=10, verbose=False, delta=5e-2):
        """
        Args:
            patience (int): validation loss가 개선되지 않을 때 기다릴 epoch 수
            verbose (bool): True일 경우 각 validation loss 개선 시 메시지 출력
            delta (float): 개선되었다고 인정할 최소 변화량
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.val_loss_min = np.inf
 
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score  # 오타 수정: bset_score → best_score
            self.save_checkpoint(val_loss)
 
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:  # verbose 조건 추가
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
 
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

        return self.early_stop  # 수정: early_stop 상태 반환
    
    def save_checkpoint(self, val_loss):
        """validation loss가 감소했을 때 호출되는 함수"""
        if self.verbose:  # 오타 수정: verbos → verbose
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss  # val_loss_min 업데이트 추가
        return True
    
    def reset(self):
        """EarlyStopping 상태 초기화"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf