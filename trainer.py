from tqdm import tqdm
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(
        self, model, optimizer, device, scheduler,
        epochs, batch_size, learning_rate, early_stopping, path,
        use_amp=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.path = path
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
    def train(self, dataloader, epochs, recon_beta=1.0, kl_beta=1.0, class_beta=1.0):
        # Training mode
        self.model.train()
        torch.cuda.empty_cache()
        
        # Records History of Loss
        history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'class_loss': [],
            'learning_rates': [],
        }
        # Losses of each batch
        batch_losses = []
        
        # Training in progress
        for epoch in tqdm(range(epochs), desc='Training Progress'):
            epoch_losses = {
                'total': [], 'recon': [], 'kl': [], 'class': []
            }
            
            for _, batch_x, _, batch_labels in dataloader:
                batch_x = batch_x.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    self.optimizer.zero_grad()
                    recon_x, mean, log_var, pred = self.model(batch_x)
                    total_loss = self.model.compute_loss(
                        recon_x, batch_x, mean, log_var, pred, batch_labels,
                        recon_beta=recon_beta, kl_beta=kl_beta, class_beta=class_beta
                    )
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    self.optimizer.step()
                
                # Record the losses
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['recon'].append(self.model.loss_dict['recon'])
                epoch_losses['kl'].append(self.model.loss_dict['kl'])
                epoch_losses['class'].append(self.model.loss_dict['class'])
            
            # Average losses of the epoch
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            
            # Update history
            for k in ['total', 'recon', 'kl', 'class']:
                history[f'{k}_loss'].append(avg_losses[k])
            history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print progress
            self._print_progress(epoch + 1, epochs, avg_losses)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(avg_losses['total'])
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
            
            # Early stopping check
            if self.early_stopping is not None:
                self.early_stopping(avg_losses['total'], self.model)
                if self.early_stopping.early_stop:
                    print('### Early stopping triggered ###')
                    break
            
            # Memory cleanup
            if (epoch + 1) % 5 == 0:  # Every 5 epochs
                torch.cuda.empty_cache()
        
        return history
    
    # Print progress
    def _print_progress(self, epoch, total_epochs, losses):
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"Total Loss: {losses['total']:.6f}")
        print(f"Recon Loss: {losses['recon']:.6f}")
        print(f"KL Loss: {losses['kl']:.6f}")
        print(f"Class Loss: {losses['class']:.6f}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    # Load the best model
    def load_checkpoint(self):
        try:
            state_dict = torch.load(self.path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {self.path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")