from trainer import Trainer
from scheduler import Scheduler, EarlyStopping
from model import SparseVAE
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader

def train_main(model, save_path, data_path, save_info_path):
    '''
    --------------------------------
    Hyperparameters Settings
    --------------------------------
    '''
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-3
    scheduler_patience = 3
    
    # model architecture
    input_dim = 768
    hidden_dims = [512, 256, 128]
    latent_dim = 16
    dropout_rate = 0.2
    
    # loss function weights
    recon_beta = 1.0
    kl_beta = 1.0
    class_beta = 1.0
    
    # for saving hyperparameters
    hyperparameters = {
        'model_name': model.__name__,
        'device': str(DEVICE),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'scheduler_patience': scheduler_patience,
        'model_architecture': {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'latent_dim': latent_dim,
            'dropout_rate': dropout_rate
        },
        'scheduler_params': {
            'factor': 0.5,
            'patience': scheduler_patience
        },
        'early_stopping_params': {
            'recon_beta': recon_beta,
            'kl_beta': kl_beta,
            'class_beta': class_beta
        },
        'data_path': data_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
        
    # Information of device
    print(f'Device: {DEVICE}')
    print(f'GPU Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU Name: {torch.cuda.get_device_name(DEVICE)}')

    '''
    --------------------------------
    Data Loading
    --------------------------------
    '''
    # Dataloader (Train)
    train_data = torch.load(data_path, map_location=DEVICE)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    '''
    --------------------------------
    Model Initialization
    --------------------------------
    '''
    # VAE model
    sparse_model = model(
        input_dim = input_dim,
        hidden_dims = hidden_dims,
        latent_dim = latent_dim,
        dropout_rate = dropout_rate
    ).to(DEVICE)

    # scheduler
    scheduler_manager = Scheduler(sparse_model, learning_rate=LEARNING_RATE)
    optimizer = scheduler_manager.get_optimizer()
    scheduler = scheduler_manager.get_scheduler(
        factor=0.5, patience=scheduler_patience, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(
        path=save_path, patience=10, min_delta=0.0, verbose=True
    )
    
    '''
    --------------------------------
    Initailization Trainer
    --------------------------------
    '''
    # Trainer Setting
    trainer = Trainer(
        model=sparse_model,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        early_stopping=early_stopping,
        path=save_path,
        use_amp=True
    )
    
    # Training in-progress
    try:
        print('\n ### Start Training ###')
        history = trainer.train(train_loader, EPOCHS, recon_beta=recon_beta, kl_beta=kl_beta, class_beta=class_beta)
        print('### Training Completed ###')
        
        # Load the Best model
        trainer.load_checkpoint()
        
        hyperparameters['training_results'] = {
            'total_loss': float(history['total_loss'][-1]),
            'recon_loss': float(history['recon_loss'][-1]),
            'kl_loss': float(history['kl_loss'][-1]),
            'class_loss': float(history['class_loss'][-1])
        }
        
        # Save hyperparameters
        with open(save_info_path, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        
        # Trainig Resultss
        print('\n ### Training Results ###')
        print(f'Total loss: {history["total_loss"][-1]:.6f}')
        print(f'Reconstruction loss: {history["recon_loss"][-1]:.6f}')
        print(f'KL divergence loss: {history["kl_loss"][-1]:.6f}')
        print(f'Classification loss: {history["class_loss"][-1]:.6f}')
        
    except Exception as e:
        print(f'### Error during training: {str(e)} ###')
        raise e