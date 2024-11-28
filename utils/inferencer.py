from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import numpy as np
import torch


'''
--------------------------------
1. SparseVAEInference
2. GMMInference
--------------------------------
'''

class SparseVAEInference:
    def __init__(self, model, device, dataloader):
        self.model = model
        self.device = device
        # data store variables
        self.batch_indices = []
        self.original_data = []
        self.reconstructed_data = []
        self.latent_means = []
        self.latent_log_vars = []
        self.labels = []
        self.predictions = []
        
        # inference for all data
        self._run_inference(dataloader)
        
        # recording results
        self._process_results()
        
    def _run_inference(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_x, _, batch_labels in dataloader:
                batch_x = batch_x.to(self.device)
                batch_labels = batch_labels.to(self.device)
                recon_x, mean, log_var, pred = self.model(batch_x)
                
                self.batch_indices.append(batch_idx.cpu())
                self.original_data.append(batch_x.cpu())
                self.reconstructed_data.append(recon_x.cpu())
                self.latent_means.append(mean.cpu())
                self.latent_log_vars.append(log_var.cpu())
                self.labels.append(batch_labels.cpu())
                self.predictions.append(pred.cpu())
    
    def _process_results(self):
        # Link all data
        self.batch_indices = torch.cat(self.batch_indices, dim=0)
        self.original_data = torch.cat(self.original_data, dim=0)
        self.reconstructed_data = torch.cat(self.reconstructed_data, dim=0)
        self.latent_means = torch.cat(self.latent_means, dim=0)
        self.latent_log_vars = torch.cat(self.latent_log_vars, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.predictions = torch.cat(self.predictions, dim=0)
        
        # Compose prediction results
        self.prediction_results = {
            'batch_indices': self.batch_indices,
            'reconstruction': self.reconstructed_data,
            'latent_mean': self.latent_means,
            'latent_log_var': self.latent_log_vars,
            'pred_probability': self.predictions,
            'true_labels': self.labels
        }
        
        # Compose latent representations
        self.latent_representations = {
            'batch_indices': self.batch_indices,
            'all_latents': self.latent_means,
            'all_labels': self.labels,
            'all_predictions': self.predictions
        }
        
        # Initialize reconstruction metrics
        self.reconstruction_metrics = None
    
    # Predict DataLoader - simple inference
    def predict_dataloader(self):
        return self.prediction_results
    
    def get_latent_representations(self):
        return self.latent_representations
    
    # Compute reconstruction metrics using pre_computed data
    def compute_reconstruction_metrics(self):
        if self.reconstruction_metrics is not None:
            return self.reconstruction_metrics
        
        try:
            similarities = {
                'cosine': [],
                'euclidean': [],
                'correlation': []
            }            
            
            label_similarities = {
                'left': {'cosine': [], 'euclidean': [], 'correlation': []},
                'right': {'cosine': [], 'euclidean': [], 'correlation': []}
            }
            
            for i in range(len(self.original_data)):
                orig = self.original_data[i]
                recon = self.reconstructed_data[i]
                label = self.labels[i]
                
                # compute similarities
                cos_sim = np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
                euclidean_dist = np.linalg.norm(orig - recon)
                correlation = np.corrcoef(orig, recon)[0, 1]
                
                # save overall similarities
                similarities['cosine'].append(cos_sim)
                similarities['euclidean'].append(euclidean_dist)
                similarities['correlation'].append(correlation)
                
                # save per labels
                label_name = 'left' if label == 0 else 'right'
                label_similarities[label_name]['cosine'].append(cos_sim)
                label_similarities[label_name]['euclidean'].append(euclidean_dist)
                label_similarities[label_name]['correlation'].append(correlation)
            
            self.reconstruction_metrics = {
                'overall': self._compute_metric_stats(similarities),
                'by_label': {
                    label: self._compute_metric_stats(metrics)
                    for label, metrics in label_similarities.items()
                }
            }
            
            return self.reconstruction_metrics
                
        except Exception as e:
            raise RuntimeError(f'### Error in computing reconstruction metrics: {str(e)} ###')
        
    # Helper method to compute statistics for each metric
    def _compute_metric_stats(self, metrics):
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for metric, values in metrics.items()
        }

class GMMInference:
    def __init__(self, n_components=2, random_state=42, covariance_type = 'full'):
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.gmm_model = None
        self.pca = None
        # For save the results
        self.fitted_data = None 
        self.prediction_results = None
    
    def gmm_fit(self, latents, classifier_preds=None, pca_components=1):
        
        try:
            if pca_components is not None:
                self.pca = PCA(n_components=pca_components)
                latents = self.pca.fit_transform(latents)
            if torch.is_tensor(latents):
                latents = latents.cpu().numpy()
                
            self.gmm_model = GaussianMixture(
                n_components=self.n_components,
                random_state=self.random_state,
                covariance_type=self.covariance_type
            )
            self.gmm_model.fit(latents)
            
            if classifier_preds is not None and self.n_components == 2:
                if torch.is_tensor(classifier_preds):
                    classifier_preds = classifier_preds.cpu().numpy()
                    
                gmm_labels = self.gmm_model.predict(latents)
                
                classifier_binary = (classifier_preds > 0).astype(int)
                accuracy_original = np.mean(gmm_labels == classifier_binary)
                accuracy_flipped = np.mean(gmm_labels != classifier_binary)
                
                if accuracy_flipped > accuracy_original:
                    self.gmm_model.means_ = self.gmm_model.means_[::-1]
                    self.gmm_model.covariances_ = self.gmm_model.covariances_[::-1]
                    self.gmm_model.weights_ = self.gmm_model.weights_[::-1]
            
            self.fitted_data = latents
            self.prediction_results = self._get_predictions(latents)
            
            return self.prediction_results
    
    # # Training GMM
    # def gmm_fit(self, latents, pca_components=1):
    #     try:
    #         if pca_components is not None:
    #             self.pca = PCA(n_components=pca_components)
    #             latents = self.pca.fit_transform(latents)
    #         if torch.is_tensor(latents):
    #             latents = latents.cpu().numpy()
                
    #         self.gmm_model = GaussianMixture(
    #             n_components=self.n_components,
    #             random_state=self.random_state,
    #             covariance_type=self.covariance_type
    #         )
    #         self.gmm_model.fit(latents)
            
    #         # Save the results
    #         self.fitted_data = latents
    #         self.prediction_results = self._get_predictions(latents)
            
    #         return self.prediction_results
            
        except Exception as e:
            raise RuntimeError(f'### Error in fitting GMM: {str(e)} ###')

    
    # Predict GMM for new data
    def gmm_predict(self, latents):
        if self.gmm_model is None:
            raise ValueError("### GMM model is not trained yet. gmm_fit() first.###")
        
        try:
            if self.pca is not None:
                latents = self.pca.transform(latents)
            if torch.is_tensor(latents):
                latents = latents.cpu().numpy()
            return self._get_predictions(latents)
            
        except Exception as e:
            raise RuntimeError(f'### Error in predicting GMM: {str(e)} ###')
            
    # Helper method to get predictions
    def _get_predictions(self, latents):
        return {
            'labels': self.gmm_model.predict(latents),
            'responsibilities': self.gmm_model.predict_proba(latents),
            'log_probs': self.gmm_model.score_samples(latents)
        }
        
    def get_model_parameters(self):
        if self.gmm_model is None:
            raise ValueError("### GMM model is not trained yet. gmm_fit() first. ###")
        
        return {
            'means': self.gmm_model.means_,
            'covariances': self.gmm_model.covariances_,
            'weights': self.gmm_model.weights_,
            'n_components': self.n_components,
            'covariance_type': self.covariance_type
        }
    
    # Statistics about the clusters
    def get_cluster_stats(self):
        if self.prediction_results is None:
            raise ValueError("### GMM predictions are not available. gmm_predict() first. ###")
        
        responsibilities = self.prediction_results['responsibilities']
        labels = self.prediction_results['labels']
        
        stats = {}
        for i in range(self.n_components):
            cluster_mask = labels == i
            stats[f'cluster_{i}'] = {
                'size': np.sum(cluster_mask),
                'proportion': np.mean(cluster_mask),
                'avg_responsibility': np.mean(responsibilities[cluster_mask, i]),
                'std_responsibility': np.std(responsibilities[cluster_mask, i])
            }
        return stats