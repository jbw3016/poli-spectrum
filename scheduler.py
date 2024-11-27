import torch
import torch.optim as optim


'''
--------------------------------
1. Scheduler
2. EarlyStopping
--------------------------------
'''

class Scheduler:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
    def get_optimizer(self):
        return self.optimizer
    
    def get_scheduler(self, factor=0.5, patience=5, verbose=True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience, verbose=verbose
        )
        return scheduler

# Early stopping and save the best model
class EarlyStopping:
    def __init__(self, path, patience=10, min_delta=0.0, verbose=True):
        self.path = path
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(loss, model)
            
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose == True:
                print(f"### EARLY STOPPING counter: {self.counter} / {self.patience} ###")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(loss, model)
            self.best_loss = loss
            self.counter = 0
    
    def save_checkpoint(self, loss, model):
        if self.verbose == True:
            print(f"### loss decreased: ({self.best_loss:.6f} -> {loss:.6f}). save the model. ###")
        torch.save(model.state_dict(), self.path)
        self.best_loss = loss