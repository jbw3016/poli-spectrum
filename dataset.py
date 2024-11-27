import re, os, nltk
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


'''
--------------------------------
1. TextEncoder
2. SparseVAEData
--------------------------------
'''

# For Text Encoding
class TextEncoder(nn.Module):
    def __init__(self, model, tokenizer, max_length=512):
        super(TextEncoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(model.parameters()).device
        self.sentence_tokenizer = nltk.sent_tokenize
    
    def preprocessing(self, text):
        text = text.strip()
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    # Text into sentences
    def split_into_sentences(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
        return self.sentence_tokenizer(text)

    # Encode texts by averaging sentence embeddings
    def encode_texts(self, texts):
        # If texts is a single text, convert it to a list
        if isinstance(texts, str):
            texts = [texts]
        # Temporary list for embedding sentences from a single data point
        all_sentences = []
        for paragraph in texts:
            sentences = self.split_into_sentences(self.preprocessing(paragraph))
            all_sentences.extend(sentences)
        if not all_sentences: # Empty text handling
            return torch.zeros(self.model.config.hidden_size).to(self.device)
        embeddings = [] 
        # Weighting sentences based on order (first sentence 1.0, last 0.5)
        weights = torch.linspace(1.0, 0.5, len(all_sentences)).to(self.device)
         # Prepare for batch processing
        batch_inputs = []
        for sentence in all_sentences:
            encoded_input = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            batch_inputs.append({
                'input_ids': encoded_input['input_ids'].to(self.device),
                'attention_mask': encoded_input['attention_mask'].to(self.device)
            })
        # Process in batches
        for i, encoded_input in enumerate(batch_inputs):
            with torch.no_grad():
                output = self.model(**encoded_input)
            embeddings.append(output.last_hidden_state[:, 0, :] * weights[i])
        
        mean_embedding = torch.mean(torch.stack(embeddings), dim=0)
        return mean_embedding.squeeze()
    

# For Creating Dataset and Dataloader
class SparseVAEData:
    def __init__(self, text_encoder, batch_size, save_dir):
        self.text_encoder = text_encoder
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    # Dataset for text
    def create_dataset(self, df):
        # Create dataset from dataframe
        # List to store embeddings of all data points in the entire dataset
        embeddings = []
        labels = []
        indices = []
        
        for idx, (texts, label) in tqdm(enumerate(zip(df['text'], df['label']))):
            with torch.no_grad():
                embedding = self.text_encoder.encode_texts(texts)
                embeddings.append(embedding)
                # Convert label to one-hot encoding
                # 0: left, 1: right
                label_idx = 1 if label == 'right' else 0
                labels.append(label_idx)
                indices.append(idx)
                
        embeddings_tensor = torch.stack(embeddings)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        return TensorDataset(indices_tensor, embeddings_tensor, embeddings_tensor, labels_tensor)

    def create_dataloader(self, dataset, shuffle=True):
        # Create dataloader from dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=16,
            pin_memory=True)
    
    def save_dataset(self, dataset, name):
        # Save dataset to a file
        path = os.path.join(self.save_dir, f'{name}.pt')
        torch.save(dataset, path)
        print(f'dataset has been saved to {path}')
    
    def load_dataset(self, name):
        # dataset loading
        path = os.path.join(self.save_dir, f'{name}.pt')
        if os.path.exists(path):
            dataset = torch.load(path)
            print(f'dataset has been loaded from {path}')
            return dataset
        else:
            raise FileNotFoundError(f'No such file: {path}')
    
    # Complementary functions
    def get_dataset_and_loader(self, df, name, force_create=False, shuffle=True):
        # Create dataset or load and create dataloader
        path = os.path.join(self.save_dir, f"{name}_dataset.pt")
        
        if os.path.exists(path) and not force_create:
            dataset = self.load_dataset(name)
        else:
            dataset = self.create_dataset(df)
            self.save_dataset(dataset, name)
        
        dataloader = self.create_dataloader(dataset, shuffle=shuffle)
        return dataset, dataloader
    
    def prepare_all_datasets(self, train_df, valid_df, test_df, force_create=False):
        # Prepare # train, valid, test dataset
        print("Preparing train dataset...")
        train_dataset, train_loader = self.get_dataset_and_loader(
            train_df, "train", force_create, shuffle=True)
        
        print("Preparing validation dataset...")
        valid_dataset, valid_loader = self.get_dataset_and_loader(
            valid_df, "valid", force_create, shuffle=False)
        
        print("Preparing test dataset...")
        test_dataset, test_loader = self.get_dataset_and_loader(
            test_df, "test", force_create, shuffle=False)
        
        return {
            'train': (train_dataset, train_loader),
            'valid': (valid_dataset, valid_loader),
            'test': (test_dataset, test_loader),
        }