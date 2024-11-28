import re, os, nltk
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


'''
--------------------------------
1. BaseEncoder
2. TextEncoder
3. HierarchicalTextEncoder
4. TrueHierarchicalTextEncoder
5. SparseVAEData
--------------------------------
'''

class BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = self.get_optimal_device()
        
    def get_optimal_device(self):
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        max_free_memory = 0
        optimal_device = 0
        
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = total_memory - allocated_memory
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                optimal_device = i
                
        return torch.device(f'cuda:{optimal_device}')
    
# For Text Encoding
class TextEncoder(BaseEncoder):
    def __init__(self, model, tokenizer, max_length=512):
        super(TextEncoder, self).__init__()
        self.device = self.get_optimal_device()
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
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
            return torch.zeros(self.model.config.hidden_size, device=self.device)
        embeddings = [] 
        # Weighting sentences based on order (first sentence 1.0, last 0.5)
        weights = torch.linspace(1.0, 0.5, len(all_sentences), device=self.device)
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
    
# For Hierarchical Text Encoding
class HierarchicalTextEncoder(BaseEncoder):
    def __init__(self, sentence_encoder, document_encoder, tokenizer, max_sentences=50, max_length=512):
        super().__init__()
        self.device = self.get_optimal_device()
        self.sentence_encoder = sentence_encoder.to(self.device)
        self.document_encoder = document_encoder.to(self.device)
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_length = max_length
        self.sentence_tokenizer = nltk.sent_tokenize

    def preprocessing(self, text):
        text = text.strip()
        text = re.sub(r'[^a-zA-Z0-9\s\.\?\!]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def encode_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        all_sentence_embeddings = []  # 문서별 임베딩을 저장할 리스트
        all_attention_masks = []      # 문서별 어텐션 마스크를 저장할 리스트
        
        for text in texts:
            # 1. split into sentences
            sentences = self.sentence_tokenizer(self.preprocessing(text))
            sentences = sentences[:self.max_sentences]
            
            if not sentences:
                zero_embedding = torch.zeros(self.document_encoder.config.hidden_size, device=self.device)
                return zero_embedding
            
            # Tokenize all sentences at once
            encoded_inputs = self.tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            text_sentence_embeddings = []  # 현재 문서의 문장 임베딩을 저장할 리스트
            text_attention_mask = []       # 현재 문서의 어텐션 마스크를 저장할 리스트
            
            # Create sentence-level embeddings
            with torch.no_grad():
                sentence_outputs = self.sentence_encoder(**encoded_inputs)
                sentence_embeddings = sentence_outputs.last_hidden_state[:, 0, :]  # [num_sentences, hidden_size]
                
            text_sentence_embeddings.append(sentence_embeddings)
            text_attention_mask.extend([1] * len(sentences))
            
            # Padding for sentence-level embeddings
            num_pad = self.max_sentences - len(sentences)
            if num_pad > 0:
                padding_embedding = torch.zeros(
                    (num_pad, sentence_embeddings.size(-1)), 
                    device=self.device
                )
                text_sentence_embeddings.append(padding_embedding)
                text_attention_mask.extend([0] * num_pad)
            
            # Combine document-level embeddings and masks
            document_embeddings = torch.cat(text_sentence_embeddings, dim=0)
            all_sentence_embeddings.append(document_embeddings)
            all_attention_masks.append(torch.tensor(text_attention_mask, device=self.device))
            
        # 3. Create document-level embeddings
        batch_embeddings = torch.stack(all_sentence_embeddings, dim=0)  # [batch_size, max_sentences, hidden_size]
        batch_attention_mask = torch.stack(all_attention_masks, dim=0)  # [batch_size, max_sentences]
        
        with torch.no_grad():
            document_outputs = self.document_encoder(
                inputs_embeds=batch_embeddings,
                attention_mask=batch_attention_mask
            )
            document_embeddings = document_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return document_embeddings.squeeze()

# For True Hierarchical Text Encoding
class TrueHierarchicalTextEncoder(BaseEncoder):
    def __init__(self, vocab_size, embed_dim, word_hidden_dim, sent_hidden_dim, doc_hidden_dim, 
                 max_sent_len=50, max_doc_len=100, dropout=0.1):
        super().__init__()
        self.device = self.get_optimal_device()
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(self.device)
        self.word_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=word_hidden_dim,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if self.word_encoder.num_layers > 1 else 0
        ).to(self.device)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=word_hidden_dim * 2,
            nhead=8,
            dim_feedforward=sent_hidden_dim,
            dropout=dropout,
            batch_first=True
        ).to(self.device)
        
        self.sentence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        ).to(self.device)
        
        self.document_encoder = nn.GRU(
            input_size=word_hidden_dim * 2,
            hidden_size=doc_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if self.document_encoder.num_layers > 1 else 0,
            bidirectional=True
        ).to(self.device)
        
        self.dropout = nn.Dropout(dropout).to(self.device)
        self.layer_norm = nn.LayerNorm(word_hidden_dim * 2).to(self.device)
        
    def _generate_mask(self, seq_len, max_len):
        mask = torch.arange(max_len, device=self.device).expand(len(seq_len), max_len) < seq_len.unsqueeze(1)
        return mask
    
    def forward(self, document, sent_lengths=None, doc_lengths=None):
        document = document.to(self.device)
        batch_size, num_sentences, num_words = document.shape
        
        if sent_lengths is None:
            sent_lengths = torch.full((batch_size, num_sentences), num_words, device=self.device)
        else:
            sent_lengths = sent_lengths.to(self.device)
            
        if doc_lengths is None:
            doc_lengths = torch.full((batch_size,), num_sentences, device=self.device)
        else:
            doc_lengths = doc_lengths.to(self.device)
            
        # 1. Word-level processing
        word_embeddings = []
        for i in range(num_sentences):
            sentence = document[:, i, :]
            embedded = self.dropout(self.embedding(sentence))
            
            curr_sent_lengths = sent_lengths[:, i]
            word_mask = self._generate_mask(curr_sent_lengths, num_words).to(self.device)
            
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, 
                curr_sent_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            
            packed_output, (word_hidden, _) = self.word_encoder(packed_embedded)
            
            sentence_repr = torch.cat([word_hidden[0], word_hidden[1]], dim=-1)
            sentence_repr = self.layer_norm(sentence_repr)
            word_embeddings.append(sentence_repr)
        
        # 2. Sentence-level processing
        sentence_embeddings = torch.stack(word_embeddings, dim=1)
        sent_mask = self._generate_mask(doc_lengths, num_sentences)
        
        sentence_outputs = self.sentence_encoder(
            sentence_embeddings,
            src_key_padding_mask=~sent_mask
        )
        sentence_outputs = self.dropout(sentence_outputs)
        
        # 3. Document-level processing
        packed_sentences = nn.utils.rnn.pack_padded_sequence(
            sentence_outputs,
            doc_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        _, document_hidden = self.document_encoder(packed_sentences)
        last_hidden = torch.cat([document_hidden[-2], document_hidden[-1]], dim=-1)
        
        return last_hidden

    def encode_document(self, document_text, tokenizer):
        # 1. Split into sentences
        sentences = nltk.sent_tokenize(document_text)
        sentences = sentences[:self.max_doc_len]
        
        # 2. Tokenize and prepare tensors
        tokenized_sentences = []
        sent_lengths = []
        
        for sent in sentences:
            tokens = tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=self.max_sent_len,
                truncation=True
            )
            sent_lengths.append(len(tokens))
            
            if len(tokens) < self.max_sent_len:
                tokens.extend([0] * (self.max_sent_len - len(tokens)))
            
            tokenized_sentences.append(tokens)
        
        # 3. Document padding
        doc_length = len(tokenized_sentences)
        while len(tokenized_sentences) < self.max_doc_len:
            tokenized_sentences.append([0] * self.max_sent_len)
            sent_lengths.append(0)
        
        # 4. Convert to tensors on GPU
        document_tensor = torch.tensor(tokenized_sentences, device=self.device).unsqueeze(0)
        sent_lengths = torch.tensor(sent_lengths, device=self.device).unsqueeze(0)
        doc_lengths = torch.tensor([doc_length], device=self.device)
        
        # 5. Model inference
        with torch.no_grad():
            document_embedding = self.forward(
                document_tensor,
                sent_lengths=sent_lengths,
                doc_lengths=doc_lengths
            )
        
        return document_embedding

# For Creating Dataset and Dataloader
class SparseVAEData:
    def __init__(self, text_encoder, batch_size, save_dir):
        self.text_encoder = text_encoder
        self.batch_size = batch_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = text_encoder.device
    
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
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
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