import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


class SparseVAE(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], latent_dim=16, dropout_rate=0.2):
        super(SparseVAE, self).__init__()
        self.latent_dim = latent_dim
        self.l1_lambda = 1e-5

        # Encoder Structure
        encoder_layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim
            
        self.encoder_fc = nn.Sequential(*encoder_layers)
        
        # Latent Vector Parameters
        self.mean = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        self.log_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        
        # Decoder Structure
        decoder_layers = []
        hidden_dims.reverse()  # Reverse the dimensions
        
        in_dim = latent_dim
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim
            
        # Output Layer
        decoder_layers.extend([
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Tanh()
        ])
        
        self.decoder_fc = nn.Sequential(*decoder_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]//2),
            nn.BatchNorm1d(hidden_dims[0]//2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0]//2, 1)
        )
        
        # Weight Initialization
        self.apply(self._init_weights)
    
    # Weight Initialization Function
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    # Encode Function
    def encode(self, x):
        h = self.encoder_fc(x)
        return self.mean(h), self.log_var(h)

    def reparameterize(self, mean, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean
    
    # Decode Function
    def decode(self, z):
        return self.decoder_fc(z)
    
    # Classify Function
    def classify(self, z):
        return self.classifier(z)
    
    # Forward Function
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        recon_x = self.decode(z)
        pred = self.classify(z)
        return recon_x, z_mean, z_log_var, pred
    
    # Regularization Function
    def l1_regularization(self):
        l1_reg = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'encoder' in name:
                    l1_reg += torch.norm(param, p=1)
                elif 'decoder' in name:
                    l1_reg += torch.norm(param, p=1)
                elif 'classifier' in name:
                    l1_reg += torch.norm(param, p=1) * 0.5  # Classifier is less regularized
        return self.l1_lambda * l1_reg
    
    # Compute Loss Function
    def compute_loss(self, recon_x, x, mean, log_var, pred, labels, 
                    recon_beta=1.0, kl_beta=1.0, class_beta=1.0):   # Weight for each loss
        # Recon Loss = MSE + L1
        mse_loss = F.mse_loss(recon_x, x, reduction='mean')
        l1_loss = F.l1_loss(recon_x, x, reduction='mean')
        recon_loss = (mse_loss + 0.1 * l1_loss) * recon_beta
        # KL loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp()) * kl_beta
        # Class loss
        class_loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(), labels.float(), 
            reduction='mean'
        ) * class_beta
        
        # L1 regularization
        l1_reg = self.l1_regularization()
        
        # Total loss
        total_loss = recon_loss + kl_loss + class_loss + l1_reg
        
        # Store individual losses
        self.loss_dict = {
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'class': class_loss.item(),
            'l1': l1_reg.item()
        }
        
        # Return total loss
        return total_loss