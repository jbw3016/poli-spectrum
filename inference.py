from utils.inferencer import SparseVAEInference, GMMInference
from utils.model import SparseVAE
import json, os, argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def arg_parse():
    parser = argparse.ArgumentParser(description='SparseVAE model inference')
    
    # parameters of the model for training
    parser.add_argument('--input_dim', type=int, default=768,
                        help='input dimension of the model (default: 768)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help='dimensions of hidden layers (default: [512, 256, 128])')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='latent dimension of the model (default: 16)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='dropout rate of the model (default: 0.2)')
    
    # parameters for inference
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--n_components', type=int, default=2,
                        help='number of GMM components (default: 2)')
    parser.add_argument('--pca_components', type=int, default=1,
                        help='number of PCA components (default: 1)')
    
    # Path related arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to trained model weights')
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to inference data (train.pt)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save inference results')
    
    # Other settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use')
    parser.add_argument('--save_plot', action='store_true',
                        help='save distribution plot')
    
    args = parser.parse_args()
    return args

def create_distribution_plot(latents_1d, labels, gmm_inference, save_path=None):
    # Set x-axis range
    x = np.linspace(min(latents_1d), max(latents_1d), 1000).reshape(-1, 1)
    
    # Calculate GMM component PDFs
    total_pdf = np.zeros_like(x)
    gmm_pdfs = []
    for i in range(gmm_inference.n_components):
        mu = gmm_inference.gmm_model.means_[i][0]
        sigma = np.sqrt(gmm_inference.gmm_model.covariances_[i][0][0])
        weight = gmm_inference.gmm_model.weights_[i]
        component_pdf = weight * stats.norm.pdf(x, mu, sigma)
        total_pdf += component_pdf.flatten()
        gmm_pdfs.append(component_pdf.flatten())
    
    # Create Plotly graph
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=["Distribution Analysis", "Data Points"],
                        row_heights=[0.7, 0.3])
    
    # Add histogram
    for i, label_name in enumerate(['Left', 'Right']):
        mask = (labels == i)
        fig.add_trace(go.Histogram(
            x=latents_1d[mask].flatten(),
            nbinsx=50,
            histnorm='probability density',
            name=f"Data {label_name}",
            opacity=0.3,
            marker_color='red' if i == 0 else 'blue'
        ), row=1, col=1)
    
    # Add GMM component PDFs
    for i, pdf in enumerate(gmm_pdfs):
        fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=pdf,
            mode='lines',
            name=f'GMM Component {i+1}',
            line=dict(dash='dash')
        ), row=1, col=1)
    
    # Add total PDF
    fig.add_trace(go.Scatter(
        x=x.flatten(),
        y=total_pdf,
        mode='lines',
        name='Total Distribution',
        line=dict(color='black', width=2)
    ), row=1, col=1)
    
    # Add data points
    for i, label_name in enumerate(['Left', 'Right']):
        mask = (labels == i)
        fig.add_trace(go.Scatter(
            x=latents_1d[mask].flatten(),
            y=[0.2] * mask.sum(),
            mode='markers',
            name=f'Data Points ({label_name})',
            marker=dict(size=8, color='red' if i == 0 else 'blue'),
            opacity=0.7
        ), row=2, col=1)
    
    # Set layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text='Political Stance Distribution Analysis',
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Plot saved to: {save_path}")
    
    return fig

def inference_main():
    # Parse arguments
    args = arg_parse()
    
    # Set GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    try:
        # Load data
        print("\n### Loading Data ###")
        data = torch.load(args.data_path, map_location=device)
        dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        print("### Initializing Model ###")
        model = SparseVAE(
            input_dim=args.input_dim,
            hidden_dims=args.hidden_dims,
            latent_dim=args.latent_dim,
            dropout_rate=args.dropout_rate
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Run inference
        print("### Running Inference ###")
        vae_inference = SparseVAEInference(model, device, dataloader)
        
        # Get results
        prediction_results = vae_inference.predict_dataloader()
        latent_representations = vae_inference.get_latent_representations()
        reconstruction_metrics = vae_inference.compute_reconstruction_metrics()
        
        # Run GMM analysis
        print("### Running GMM Analysis ###")
        gmm_inference = GMMInference(n_components=args.n_components)
        latents = latent_representations['all_latents']
        labels = latent_representations['all_labels']
        
        # Apply PCA and run GMM
        pca = PCA(n_components=args.pca_components)
        latents_1d = pca.fit_transform(latents.cpu().numpy())
        gmm_results = gmm_inference.gmm_fit(latents_1d)
        cluster_stats = gmm_inference.get_cluster_stats()
        
        # Save results
        print("\n### Saving Results ###")
        results = {
            'prediction_results': prediction_results,
            'latent_representations': latent_representations,
            'reconstruction_metrics': reconstruction_metrics,
            'gmm_results': gmm_results,
            'cluster_stats': cluster_stats,
            'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
        
        torch.save(results, os.path.join(args.save_path, 'inference_results.pt'))
        
        # Create and save distribution plot
        if args.save_plot:
            print("### Creating Distribution Plot ###")
            plot_path = os.path.join(args.save_path, 'distribution_plot.html')
            create_distribution_plot(
                latents_1d, 
                labels.cpu().numpy(), 
                gmm_inference,
                save_path=plot_path
            )
        
        # Print inference results
        print("\n### Inference Results ###")
        print("\nReconstruction Metrics:")
        for metric, value in reconstruction_metrics['overall'].items():
            print(f"{metric}: {value}")
        
        print("\nGMM Cluster Statistics:")
        for cluster, stats in cluster_stats.items():
            print(f"\n{cluster}:")
            for stat_name, stat_value in stats.items():
                print(f"{stat_name}: {stat_value:.4f}")
        
        print(f"\nResults saved to: {args.save_path}")
        
    except Exception as e:
        print(f"\n### Error during inference: {str(e)} ###")
        raise e

if __name__ == "__main__":
    inference_main()