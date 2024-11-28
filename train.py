from utils.trainer import Trainer
from utils.scheduler import Scheduler, EarlyStopping
from utils.model import SparseVAE
import json, os, argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader

def arg_parse():
    parser = argparse.ArgumentParser(description = 'SparseVAE model training')
    
    # parameters of the model for training
    parser.add_argument('--model_type', type=str, default='bert-base-uncased',
                        help='model for embedding (defalut: bert-base-uncased)')
    parser.add_argument('--input_dim', type=int, default=768,
                        help='input dimension of the model (default: 768)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help='dimensions of hidden layers (default: [512, 256, 128])')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='latent dimension of the model (default: 16)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='dropout rate of the model (default: 0.2)')
    
    # parameters for training
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='learning rate (default: 2e-3)')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='scheduler patience (default: 3)')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='scheduler factor (default: 0.5)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='early stopping patience (default: 10)')
    
    # Loss weight related arguments
    parser.add_argument('--recon_beta', type=float, default=1.0,
                        help='reconstruction loss weight (default: 1.0)')
    parser.add_argument('--kl_beta', type=float, default=1.0,
                        help='KL divergence loss weight (default: 1.0)')
    parser.add_argument('--class_beta', type=float, default=1.0,
                        help='classification loss weight (default: 1.0)')
    
    # Path related arguments
    parser.add_argument('--save_path', type=str, required=True,
                        help='model save path')
    parser.add_argument('--data_path', type=str, required=True,
                        help='data path')
    parser.add_argument('--save_info_path', type=str, required=True,
                        help='training information save path')
    
    # Other settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision')
    
    args = parser.parse_args()
    return args

def train_main():
    '''
    --------------------------------
    1. Argument Parsing
    2. Saving Directory Setting
    3. Records for Hyperparameters
    --------------------------------
    '''
    # parse the arguments
    args = arg_parse()
    
    # GPU Setting
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # saving directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_info_path), exist_ok=True)
    
    # saving hyperparameters for records
    hyperparameters = {
        'model_type': args.model_type,
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'scheduler_patience': args.scheduler_patience,
        'model_architecture': {
            'input_dim': args.input_dim,
            'hidden_dims': args.hidden_dims,
            'latent_dim': args.latent_dim,
            'dropout_rate': args.dropout_rate
        },
        'scheduler_params': {
            'factor': args.scheduler_factor,
            'patience': args.scheduler_patience
        },
        'loss_weights': {
            'recon_beta': args.recon_beta,
            'kl_beta': args.kl_beta,
            'class_beta': args.class_beta
        },
        'data_path': args.data_path,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
        
    # Information of device
    print(f'Device: {device}')
    print(f'GPU Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU Name: {torch.cuda.get_device_name(device)}')

    '''
    --------------------------------
    1. Data Loading
    2. Model Initialization
    3. Scheduler and Optimizer Setting
    4. Trainer Initialization
    5. Training Execution
    --------------------------------
    '''
    # Data Loading
    train_data = torch.load(args.data_path, map_location=device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    # Model Initialization
    sparse_model = SparseVAE(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Scheduler Setting
    scheduler_manager = Scheduler(sparse_model, learning_rate=args.learning_rate)
    optimizer = scheduler_manager.get_optimizer()
    scheduler = scheduler_manager.get_scheduler(
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=True
    )
    
    # Early stopping Setting
    early_stopping = EarlyStopping(
        path=args.save_path,
        patience=args.early_stopping_patience,
        min_delta=0.0,
        verbose=True
    )
    
    # Trainer Initialization
    trainer = Trainer(
        model=sparse_model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping=early_stopping,
        path=args.save_path,
        use_amp=args.use_amp
    )
    
    # Training Execution
    try:
        print('\n ### Start Training ###')
        history = trainer.train(
            train_loader,
            args.epochs,
            recon_beta=args.recon_beta,
            kl_beta=args.kl_beta,
            class_beta=args.class_beta
        )
        print('### Training Completed ###')
        
        # Load the best model
        trainer.load_checkpoint()
        
        # Save training results
        hyperparameters['training_results'] = {
            'total_loss': float(history['total_loss'][-1]),
            'recon_loss': float(history['recon_loss'][-1]),
            'kl_loss': float(history['kl_loss'][-1]),
            'class_loss': float(history['class_loss'][-1])
        }
        
        # Save hyperparameters
        with open(args.save_info_path, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        
        # Print training results
        print('\n ### Training Results ###')
        print(f'Total loss: {history["total_loss"][-1]:.6f}')
        print(f'Reconstruction loss: {history["recon_loss"][-1]:.6f}')
        print(f'KL divergence loss: {history["kl_loss"][-1]:.6f}')
        print(f'Classification loss: {history["class_loss"][-1]:.6f}')
        
    except Exception as e:
        print(f'### Error during training: {str(e)} ###')
        raise e

if __name__ == "__main__":
    train_main()