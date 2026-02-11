import pandas as pd
import numpy as np
import torch
import os
import pickle
from typing import Dict, Optional
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix, precision_recall_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ==================== Data Processing Functions ====================

def parse_topics(topics_str):
    """Parse comma-separated topics string into a list."""
    if pd.isna(topics_str) or topics_str == '':
        return []
    return [topic.strip() for topic in str(topics_str).split(',') if topic.strip()]


def prepare_multilabel_data(df, text_col='speech', label_col='topic'):
    """
    Prepare data for multi-label classification. Task Classification needs a 
    binary matrix of labels. Eg, 3 lables (A, B, C) could be yield a datapoint
    like: (0,1,1) for labels B and C for an example datapoint.
    
    Args:
        df: DataFrame with text and comma-separated topics
        text_col: Name of the text column
        label_col: Name of the topics column
    
    Returns:
        texts: List of text samples
        labels_binary: Binary matrix of labels
        mlb: Fitted MultiLabelBinarizer
        label_names: List of all unique labels
    """
    # Parse topics into lists
    df['parsed_topics'] = df[label_col].apply(parse_topics)
    
    # Remove rows with no topics
    df_filtered = df[df['parsed_topics'].apply(len) > 0].copy()
    
    texts = df_filtered[text_col].tolist()
    topics_lists = df_filtered['parsed_topics'].tolist()
    
    # Create binary labels
    mlb = MultiLabelBinarizer()
    labels_binary = mlb.fit_transform(topics_lists)
    
    print(f"Total samples: {len(texts)}")
    print(f"Number of unique topics: {len(mlb.classes_)}")
    print(f"Topics: {list(mlb.classes_)}")
    
    return texts, labels_binary, mlb, list(mlb.classes_)


# ==================== PyTorch Dataset ====================

class MultiLabelTextDataset(Dataset):
    """Dataset for multi-label text classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize the multi-label text dataset.
        
        Args:
            texts: List of text strings to classify
            labels: Binary label matrix (samples x num_labels)
            tokenizer: HuggingFace tokenizer for text encoding (MacBERTh for old english)
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing:
                - input_ids: Tokenized input IDs (tensor)
                - attention_mask: Attention mask for padding (tensor)
                - labels: Multi-label binary vector (tensor)
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text: converts words to token IDs, adds special tokens ([CLS], [SEP]),
        # pads/truncates to max_length, and creates attention mask to ignore padding
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,        # Add [CLS] at start and [SEP] at end
            max_length=self.max_length,     
            padding='max_length',           # Pad shorter sequences to max_length with [PAD] tokens
            truncation=True,                # Truncate longer sequences to max_length
            return_attention_mask=True,     # Return mask: 1 for real tokens, 0 for padding
            return_tensors='pt'             # Return PyTorch tensors instead of lists
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


# ==================== Model Training ====================

def train_bert_multilabel(
    df,
    model_name='wdoppenberg/macberth',
    text_col='speech',
    label_col='topic',
    test_size=0.2, # Fraction of data for validation. 
    batch_size=32, # Batch size adjusted for RCC GPU memory. Change depending on GPU.
    epochs=5,      # Same as above
    learning_rate=3e-5,  # Same as above
    max_length=512,
    device=None,
    save_path='trained_model'
):
    """
    Train a BERT model for multi-label text classification.
    
    This function fine-tunes a pre-trained BERT model to predict multiple topics for each speech.
    It handles comma-separated topics in the dataset and trains the model to predict topic 
    probabilities for each speech. Includes automatic GPU detection, mixed precision training,
    and model checkpointing.
    
    Args:
        df (pd.DataFrame): DataFrame containing speeches and their topics
        model_name (str): HuggingFace model identifier (default: 'wdoppenberg/macberth')
                         Examples: 'emanjavacas/MacBERTh', 'bert-base-uncased'
        text_col (str): Name of column containing speech text (default: 'speech')
        label_col (str): Name of column containing comma-separated topics (default: 'topic')
        test_size (float): Fraction of data to use for validation (default: 0.2)
        batch_size (int): Number of speeches to process simultaneously (default: 32)
                         Reduce if you encounter out-of-memory errors
        epochs (int): Number of complete passes through the training data (default: 5)
        learning_rate (float): Step size for weight updates (default: 3e-5)
                              Typical range for BERT fine-tuning: 2e-5 to 5e-5
        max_length (int): Maximum number of tokens per speech (default: 512)
                         Longer speeches are truncated, shorter ones are padded
        device (torch.device): Device to train on (default: auto-detect GPU/CPU)
        save_path (str): Directory path to save the trained model (default: 'trained_model')
    
    Returns:
        tuple: (model, tokenizer, mlb, train_loader, val_loader)
            - model: Trained PyTorch model
            - tokenizer: HuggingFace tokenizer for text preprocessing
            - mlb: MultiLabelBinarizer for converting topic names to/from binary format
            - train_loader: DataLoader for training data (useful for evaluation)
            - val_loader: DataLoader for validation data (useful for evaluation)
    
    Example:
        >>> model, tokenizer, mlb, train_loader, val_loader = train_bert_multilabel(
        ...     df=df_speeches,
        ...     model_name='emanjavacas/MacBERTh',
        ...     text_col='speech',
        ...     label_col='topic',
        ...     batch_size=8,
        ...     epochs=3
        ... )
    
    Notes:
        - Automatically uses GPU if available (with mixed precision for speed)
        - Saves model, tokenizer, and label encoder to save_path
        - Topics in label_col should be comma-separated (e.g., "Politics, Economy")
        - Training progress is shown with progress bars for each epoch
    """
    
    # 1. Use cpu if not gpu found
    save_path = os.path.abspath(save_path)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    # 2. Prepare Data
    
    texts, labels_binary, mlb, label_names = prepare_multilabel_data(
        df, text_col=text_col, label_col=label_col
    )
    
    #sklearn train/val split. Good for ML tasks since it shuffles data into iid samples.
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels_binary, test_size=test_size, random_state=42
    )
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # 3. Load Tokenizer & Model
    print(f"\n=== Loading Model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification"
    )
    model.to(device)
    
    #Creates PyTorch df
    train_dataset = MultiLabelTextDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = MultiLabelTextDataset(X_val, y_val, tokenizer, max_length)
    
    # Use num_workers only on Linux/GPU systems (Windows can have issues)
    num_workers = 4 if (torch.cuda.is_available() and os.name != 'nt') else 0
    pin_memory = torch.cuda.is_available()
    
    # Create DataLoader: batches the dataset and handles shuffling/parallel data loading
    # shuffle=True randomizes order each epoch to prevent overfitting to sequence order
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # Setup AdamW optimizer: adaptive learning rate optimizer designed for transformers
    # Includes weight decay regularization to prevent overfitting
    # Above my paygrade to understand the theory. 
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    
    # Learning rate scheduler: gradually warms up then linearly decays learning rate
    # Helps model converge more smoothly
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 6. Training Loop (with AMP)
    print("\n=== Starting Training (Optimized) ===")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        #nice visual bar for training progress
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in train_pbar:
            #sets previous gradients to zero
            optimizer.zero_grad()
            
            # Move batch to GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Mixed Precision Forward Pass (only on CUDA)
            if use_amp:
                with autocast():
                    # predicts
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    # calculates error/loss
                    loss = outputs.loss
                
                # Scaled Backward Pass/ Adjusts Weights
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training on CPU
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

    # 7. Save Model
    print(f"\nSaving model to: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save the MultiLabelBinarizer so we can decode predictions later
    import pickle
    with open(os.path.join(save_path, 'mlb.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
    
    print(f"=== Training Complete ===")
    
    return model, tokenizer, mlb, train_loader, val_loader


def create_data_loaders_from_df(
    df,
    tokenizer,
    mlb,
    text_col='speech',
    label_col='topic',
    test_size=0.2,
    batch_size=32,
    max_length=512,
    device=None
):
    """
    Create train and validation DataLoaders from a DataFrame.
    Useful when you have a pre-trained model and want to evaluate it.
    
    Args:
        df: DataFrame with text and labels
        tokenizer: Tokenizer for the model
        mlb: MultiLabelBinarizer (must be fitted - use the one from your trained model)
        text_col: Name of text column
        label_col: Name of label column
        test_size: Fraction for validation set
        batch_size: Batch size for DataLoaders
        max_length: Maximum sequence length
        device: Torch device
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
    """
    # Parse topics and convert to binary using the existing mlb
    df = df.copy()
    df['parsed_topics'] = df[label_col].apply(parse_topics)
    df_filtered = df[df['parsed_topics'].apply(len) > 0].copy()
    
    texts = df_filtered[text_col].tolist()
    topics_lists = df_filtered['parsed_topics'].tolist()
    
    # Use the existing mlb to transform (don't create a new one)
    labels_binary = mlb.transform(topics_lists)
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels_binary, test_size=test_size, random_state=42
    )
    
    # Create datasets
    train_dataset = MultiLabelTextDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = MultiLabelTextDataset(X_val, y_val, tokenizer, max_length)
    
    # Create DataLoaders
    num_workers = 4 if (torch.cuda.is_available() and os.name != 'nt') else 0
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
def evaluate_trained_model(
    df,
    target_col,
    predicted_col,
    label_names=None,
    mlb=None
):
    """
    Evaluate predictions by comparing target and predicted columns.
    Reports F1 scores and confusion matrix (TP, FP, TN, FN) for each class.
    
    Args:
        df: DataFrame with target and predicted columns
        target_col: Name of column with true labels (comma-separated string, list of topics, or binary matrix)
        predicted_col: Name of column with predicted labels (comma-separated string, list of topics, or binary matrix)
        label_names: List of label names (required if columns contain binary matrices)
        mlb: MultiLabelBinarizer (required for comma-separated strings or topic lists)
    
    Returns:
        results: Dictionary with metrics and confusion matrices
    
    Examples:
        # Example 1: Binary matrix columns
        df['target'] = [[1,0,1], [0,1,0], ...]  # Binary vectors
        df['predicted'] = [[1,0,0], [0,1,1], ...]
        results = evaluate_trained_model(df, 'target', 'predicted', label_names=['A', 'B', 'C'])
        
        # Example 2: Topic list columns with MLB
        df['target'] = [['Politics', 'Economy'], ['Sports'], ...]
        df['predicted'] = [['Politics'], ['Sports', 'Weather'], ...]
        results = evaluate_trained_model(df, 'target', 'predicted', mlb=mlb)
        
        # Example 3: Comma-separated string columns (from make_predictions output)
        df['target'] = 'Politics, Economy'
        df['predicted'] = 'Politics, Sports'
        results = evaluate_trained_model(df, 'target', 'predicted', mlb=mlb)
    """
    
    # Convert to binary matrices if needed
    targets = df[target_col].tolist()
    predictions = df[predicted_col].tolist()
    
    # Check if data is already binary matrices, lists, or comma-separated strings
    if isinstance(targets[0], str):
        # Comma-separated strings - parse them into lists
        targets = [parse_topics(t) for t in targets]
        predictions = [parse_topics(p) for p in predictions]
        
        # Convert to binary using mlb
        if mlb is None:
            raise ValueError("mlb (MultiLabelBinarizer) required when columns contain comma-separated strings")
        y_true = mlb.transform(targets)
        y_pred = mlb.transform(predictions)
        label_names = list(mlb.classes_)
        
    elif isinstance(targets[0], (list, np.ndarray)):
        if len(targets[0]) > 0 and isinstance(targets[0][0], str):
            # Lists of topic names - need MLB
            if mlb is None:
                raise ValueError("mlb (MultiLabelBinarizer) required when columns contain topic name lists")
            y_true = mlb.transform(targets)
            y_pred = mlb.transform(predictions)
            label_names = list(mlb.classes_)
        else:
            # Already binary matrices
            y_true = np.array(targets)
            y_pred = np.array(predictions)
            if label_names is None:
                raise ValueError("label_names required when columns contain binary matrices")
    else:
        raise ValueError("Target and predicted columns must contain lists, arrays, or comma-separated strings")
    
    print("\n=== Evaluating Predictions ===")
    print(f"Total samples: {len(y_true)}")
    print(f"Number of classes: {len(label_names)}")
    
    # Calculate overall F1 score
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"F1 Scores:")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Micro F1: {f1_micro:.4f}")
    print(f"{'='*50}")
    
    # Generate confusion matrix for each class
    print(f"\nConfusion Matrix per Class:")
    print(f"{'='*50}")
    
    cm_multi = multilabel_confusion_matrix(y_true, y_pred)
    
    per_class_metrics = []
    
    for i, label_name in enumerate(label_names):
        tn, fp, fn, tp = cm_multi[i].ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nClass: {label_name}")
        print(f"  TP (True Positives):  {tp:4d}  |  FP (False Positives): {fp:4d}")
        print(f"  FN (False Negatives): {fn:4d}  |  TN (True Negatives):  {tn:4d}")
        print(f"  Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1: {f1:.3f}")
        
        per_class_metrics.append({
            'class': label_name,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    print(f"\n{'='*50}")
    
    # Return results dictionary
    results = {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrices': cm_multi,
        'per_class_metrics': per_class_metrics,
        'label_names': label_names
    }
    
    return results


# ==================== Embedding Cache Functions ====================

def create_embedding_cache(
    df_texts: pd.DataFrame,
    id_col: str,
    text_col: str,
    model,
    tokenizer,
    max_length: int = 512,
    chunk_size: int = 400,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    cache_path: Optional[str] = None,
    show_progress_bar: bool = True
) -> Dict[str, np.ndarray]:
    """
    Pre-compute BERT embeddings for all unique texts and return a cache dictionary.
    This is the KEY OPTIMIZATION - embed once, reuse everywhere for threshold testing.
    
    Extracts embeddings from the BERT base model (before classification head) and caches them.
    For long texts, splits into chunks and averages the embeddings.

    Parameters
    ----------
    df_texts : pd.DataFrame
        DataFrame containing unique IDs and their texts.
    id_col : str
        Name of the ID column.
    text_col : str
        Name of the text column.
    model : AutoModelForSequenceClassification
        Trained BERT model (will extract embeddings from base model).
    tokenizer : AutoTokenizer
        Tokenizer for the model.
    max_length : int, default 512
        Maximum sequence length for tokenization.
    chunk_size : int, default 400
        Max tokens per chunk for long texts (before splitting).
    batch_size : int, default 32
        Batch size for processing (for efficiency).
    device : torch.device, optional
        Device to run on (auto-detected if None).
    cache_path : str, optional
        Path to save/load cache file. If file exists, loads from cache.
    show_progress_bar : bool, default True
        Whether to show progress bar during encoding.

    Returns
    -------
    dict
        Dictionary mapping {id: embedding_vector} for all texts.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if cache file exists
    if cache_path and os.path.exists(cache_path):
        print(f"\n>>> Loading Embedding Cache from {cache_path}")
        with open(cache_path, 'rb') as f:
            embedding_cache = pickle.load(f)
        print(f"✓ Loaded {len(embedding_cache)} embeddings from cache\n")
        return embedding_cache
    
    model.to(device)
    model.eval()
    
    # Get the base BERT model (before classification head)
    # Different model architectures have different attribute names
    if hasattr(model, 'bert'):
        base_model = model.bert
    elif hasattr(model, 'base_model'):
        base_model = model.base_model
    elif hasattr(model, 'roberta'):  # For RoBERTa models
        base_model = model.roberta
    else:
        # Try to get the first module that's not the classifier
        base_model = None
        for name, module in model.named_children():
            if 'classifier' not in name.lower() and 'dropout' not in name.lower():
                base_model = module
                break
        if base_model is None:
            raise ValueError("Could not find base model. Model architecture not supported.")
    
    # Get unique id-text pairs (in case of duplicates)
    df_unique = df_texts[[id_col, text_col]].drop_duplicates(subset=[id_col]).copy()
    
    print(f"\n>>> Creating Embedding Cache")
    print(f"Total unique texts: {len(df_unique)}")
    print(f"Device: {device}")
    
    embedding_cache = {}
    
    # Process in batches for efficiency
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(df_unique), batch_size), 
                                desc="Creating embeddings", 
                                disable=not show_progress_bar):
            batch_end = min(batch_start + batch_size, len(df_unique))
            batch_df = df_unique.iloc[batch_start:batch_end]
            
            batch_embeddings = []
            batch_ids = []
            
            for idx, row in batch_df.iterrows():
                text = str(row[text_col])
                text_id = str(row[id_col])
                
                # Tokenize once to check if chunking is needed
                tokens = tokenizer.encode(text, add_special_tokens=True)
                
                if len(tokens) <= max_length:
                    # Text fits in one chunk
                    encoding = tokenizer(
                        text,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    # Extract embeddings from base model
                    base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
                    # Use pooled output (typically the [CLS] token embedding after pooling)
                    if hasattr(base_outputs, 'pooler_output') and base_outputs.pooler_output is not None:
                        embedding = base_outputs.pooler_output.cpu().numpy()[0]
                    else:
                        # Fallback: use [CLS] token (first token)
                        embedding = base_outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                    
                    batch_embeddings.append(embedding)
                    batch_ids.append(text_id)
                    
                else:
                    # Split long text into chunks and average embeddings
                    words = text.split()
                    chunks = []
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        word_tokens = tokenizer.encode(word, add_special_tokens=False)
                        if current_length + len(word_tokens) > chunk_size:
                            if current_chunk:
                                chunks.append(' '.join(current_chunk))
                            current_chunk = [word]
                            current_length = len(word_tokens)
                        else:
                            current_chunk.append(word)
                            current_length += len(word_tokens)
                    
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    # Process chunks and average embeddings
                    chunk_embeddings = []
                    for chunk in chunks:
                        encoding = tokenizer(
                            chunk,
                            add_special_tokens=True,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_attention_mask=True,
                            return_tensors='pt'
                        )
                        
                        input_ids = encoding['input_ids'].to(device)
                        attention_mask = encoding['attention_mask'].to(device)
                        
                        base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(base_outputs, 'pooler_output') and base_outputs.pooler_output is not None:
                            chunk_emb = base_outputs.pooler_output.cpu().numpy()[0]
                        else:
                            chunk_emb = base_outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                        chunk_embeddings.append(chunk_emb)
                    
                    # Average embeddings across chunks
                    embedding = np.mean(chunk_embeddings, axis=0)
                    batch_embeddings.append(embedding)
                    batch_ids.append(text_id)
            
            # Store batch embeddings in cache
            for text_id, embedding in zip(batch_ids, batch_embeddings):
                embedding_cache[text_id] = embedding
    
    # Save cache if path provided
    if cache_path:
        print(f"\n>>> Saving Embedding Cache to {cache_path}")
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding_cache, f)
        print(f"✓ Saved {len(embedding_cache)} embeddings to cache")
    
    print(f"✓ Embedding cache created: {len(embedding_cache)} entries")
    if embedding_cache:
        sample_emb = list(embedding_cache.values())[0]
        print(f"Embedding dimension: {sample_emb.shape[0]}\n")
    
    return embedding_cache


def predict_from_embedding_cache(
    embedding_cache: Dict[str, np.ndarray],
    df: pd.DataFrame,
    id_col: str,
    model,
    mlb,
    device: Optional[torch.device] = None,
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Predict topic probabilities using cached embeddings (much faster than re-encoding).
    
    This function applies only the classification head to pre-computed embeddings,
    avoiding the expensive BERT forward pass.

    Parameters
    ----------
    embedding_cache : dict
        Dictionary mapping {id: embedding_vector} from create_embedding_cache.
    df : pd.DataFrame
        DataFrame with id_col matching keys in embedding_cache.
    id_col : str
        Name of the ID column to look up embeddings.
    model : AutoModelForSequenceClassification
        Trained model (only classification head is used).
    mlb : MultiLabelBinarizer
        Label binarizer for topic names.
    device : torch.device, optional
        Device to run on (auto-detected if None).
    batch_size : int, default 128
        Batch size for classification (can be larger since embeddings are smaller).

    Returns
    -------
    pd.DataFrame
        DataFrame with probability columns for each topic.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Get classification head
    classifier = model.classifier
    
    topic_names = mlb.classes_
    all_probabilities = []
    missing_ids = []
    
    print(f"Predicting from {len(df)} cached embeddings...")
    
    # Get embeddings for all rows
    embeddings_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        text_id = str(row[id_col])
        if text_id in embedding_cache:
            embeddings_list.append(embedding_cache[text_id])
            valid_indices.append(idx)
        else:
            missing_ids.append(text_id)
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} IDs not found in embedding cache")
        if len(missing_ids) <= 10:
            print(f"Missing IDs: {missing_ids}")
    
    if not embeddings_list:
        raise ValueError("No valid embeddings found in cache for the provided DataFrame")
    
    # Process in batches
    embeddings_array = np.array(embeddings_list)
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(embeddings_array), batch_size), 
                               desc="Predicting probabilities"):
            batch_end = min(batch_start + batch_size, len(embeddings_array))
            batch_embeddings = embeddings_array[batch_start:batch_end]
            
            # Convert to tensor
            batch_embeddings_tensor = torch.FloatTensor(batch_embeddings).to(device)
            
            # Apply classification head
            # Note: Some models have dropout before classifier, so we need to handle that
            if hasattr(model, 'dropout') and model.dropout is not None:
                batch_embeddings_tensor = model.dropout(batch_embeddings_tensor)
            
            logits = classifier(batch_embeddings_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probabilities.extend(probs)
    
    # Create DataFrame with probability columns
    all_probabilities = np.array(all_probabilities)
    prob_df = pd.DataFrame(all_probabilities, columns=[f'prob_{topic}' for topic in topic_names])
    
    # Combine with original dataframe (only valid rows)
    df_result = df.loc[valid_indices].reset_index(drop=True)
    prob_df.index = df_result.index
    df_result = pd.concat([df_result, prob_df], axis=1)
    
    return df_result


# ==================== Prediction Functions ====================

def predict_topic_probabilities(
    df,
    model,
    tokenizer,
    mlb,
    text_col='speech',
    max_length=512,
    chunk_size=400,
    device=None,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
    id_col: Optional[str] = None
):
    """
    Predict topic probabilities for each row in dataframe.
    Handles long speeches by splitting into chunks and averaging probabilities.
    
    NEW: Can use pre-computed embedding cache for much faster predictions
    (useful when testing different thresholds).
    
    Args:
        df: DataFrame with text column
        model: Trained model
        tokenizer: Tokenizer
        mlb: MultiLabelBinarizer
        text_col: Name of text column
        max_length: Maximum sequence length
        chunk_size: Max tokens per chunk for long texts (before splitting)
        device: Torch device
        embedding_cache: Optional dict mapping {id: embedding_vector}. 
                        If provided, uses cached embeddings (much faster).
        id_col: Optional ID column name. Required if embedding_cache is provided.
    
    Returns:
        df_result: DataFrame with probability columns for each topic
    """
    # If embedding cache provided, use optimized path
    if embedding_cache is not None:
        if id_col is None:
            raise ValueError("id_col must be provided when using embedding_cache")
        return predict_from_embedding_cache(
            embedding_cache=embedding_cache,
            df=df,
            id_col=id_col,
            model=model,
            mlb=mlb,
            device=device
        )
    
    # Original path: compute embeddings on-the-fly
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    topic_names = mlb.classes_
    all_probabilities = []
    
    print(f"Processing {len(df)} speeches...")
    
    with torch.no_grad():
        for text in tqdm(df[text_col], desc="Predicting probabilities"):
            text = str(text)
            
            # Tokenize once to check if chunking is needed
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            if len(tokens) <= max_length:
                # Text fits in one chunk
                encoding = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                
            else:
                # Split long text into chunks and average
                words = text.split()
                chunks = []
                current_chunk = []
                current_length = 0
                
                for word in words:
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    if current_length + len(word_tokens) > chunk_size:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word_tokens)
                    else:
                        current_chunk.append(word)
                        current_length += len(word_tokens)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Process chunks and average probabilities
                chunk_probs = []
                for chunk in chunks:
                    encoding = tokenizer(
                        chunk,
                        add_special_tokens=True,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    chunk_prob = torch.sigmoid(outputs.logits).cpu().numpy()[0]
                    chunk_probs.append(chunk_prob)
                
                probs = np.mean(chunk_probs, axis=0)
            
            all_probabilities.append(probs)
    
    # Create DataFrame with probability columns
    all_probabilities = np.array(all_probabilities)
    prob_df = pd.DataFrame(all_probabilities, columns=[f'prob_{topic}' for topic in topic_names])
    
    # Combine with original dataframe
    df_result = df.reset_index(drop=True)
    df_result = pd.concat([df_result, prob_df], axis=1)
    
    return df_result


# ==================== Threshold Optimization Functions ====================

def get_optimal_threshold_minimize_fp(y_true, y_probs, beta=0.5):
    """
    Finds the optimal threshold to minimize False Positives (optimize Precision).
    Uses the F-beta score with beta < 1 (default 0.5).
    
    Args:
        y_true: True binary labels (1D array or 2D array for multi-label)
        y_probs: Predicted probabilities (1D array or 2D array for multi-label)
        beta: Beta parameter for F-beta score (default: 0.5, emphasizes precision)
    
    Returns:
        best_thresh: Optimal threshold value
        best_fbeta: Best F-beta score achieved
    """
    # Handle multi-label case: flatten if needed
    if y_probs.ndim > 1:
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Avoid zero division
    numerator = (1 + beta**2) * (precisions * recalls)
    denominator = (beta**2 * precisions) + recalls
    fbeta_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # Locate the best threshold
    ix = np.argmax(fbeta_scores)
    
    if ix < len(thresholds):
        best_thresh = thresholds[ix]
    else:
        best_thresh = thresholds[-1] if len(thresholds) > 0 else 0.5
        
    return best_thresh, fbeta_scores[ix]


def get_optimal_threshold_minimize_fn(y_true=None, y_probs=None, cost_fp=1, cost_fn=5, strategy='cost'):
    """
    Finds the optimal threshold to minimize False Negatives (optimize Recall).
    Two strategies:
    1. 'cost' (Default): Uses Cost-Sensitive Bayes Risk: tau = C_fp / (C_fp + C_fn).
       Does not require y_true/y_probs.
    2. 'f2': Maximizes F2 score (beta=2). Requires y_true and y_probs.
    
    Args:
        y_true: True binary labels (required for 'f2' strategy)
        y_probs: Predicted probabilities (required for 'f2' strategy)
        cost_fp: Cost of false positive (default: 1)
        cost_fn: Cost of false negative (default: 5, higher penalty for missing topics)
        strategy: 'cost' or 'f2' (default: 'cost')
    
    Returns:
        best_thresh: Optimal threshold value
    """
    if strategy == 'cost':
        return cost_fp / (cost_fp + cost_fn)
    
    if strategy == 'f2':
        if y_true is None or y_probs is None:
            raise ValueError("Strategy 'f2' requires y_true and y_probs.")
        
        # Handle multi-label case: flatten if needed
        if y_probs.ndim > 1:
            y_true = y_true.flatten()
            y_probs = y_probs.flatten()
        
        # We reuse logic similar to minimizing FP but with beta=2
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        beta = 2.0
        numerator = (1 + beta**2) * (precisions * recalls)
        denominator = (beta**2 * precisions) + recalls
        f_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        ix = np.argmax(f_scores)
        if ix < len(thresholds):
            return thresholds[ix]
        else:
            return thresholds[-1] if len(thresholds) > 0 else 0.5
            
    raise ValueError(f"Unknown strategy: {strategy}")



def find_optimal_thresholds_multilabel(
    y_true,
    y_probs,
    mlb,
    cost_fp=1,
    cost_fn=5,
    beta=0.5
):
    """
    Find optimal thresholds for multi-label classification using ALL strategies.
    Computes thresholds for minimize_fn, minimize_fp, and per_class approaches.
    
    Args:
        y_true: True binary label matrix (n_samples, n_classes)
        y_probs: Predicted probability matrix (n_samples, n_classes)
        mlb: MultiLabelBinarizer (for class names)
        cost_fp: Cost of false positive (for 'minimize_fn' strategy, default: 1)
        cost_fn: Cost of false negative (for 'minimize_fn' strategy, default: 5)
        beta: Beta parameter for F-beta score (for 'minimize_fp' strategy, default: 0.5)
    
    Returns:
        dict: Dictionary with all strategies and their thresholds:
            - 'minimize_fn': Dict with cost-sensitive threshold (optimizes recall)
            - 'minimize_fp': Dict with precision-optimized threshold
            - 'per_class_f1': Dict with per-class F1-optimized thresholds (balanced)
    """
    n_classes = y_probs.shape[1]
    results = {}
    
    print("\n=== Computing Optimal Thresholds (All Strategies) ===\n")
    
    # Strategy 1: Minimize False Negatives (Cost-Sensitive)
    print("1. Minimize FN (Cost-Sensitive Approach)...")
    global_thresh_fn = get_optimal_threshold_minimize_fn(
        cost_fp=cost_fp, cost_fn=cost_fn, strategy='cost'
    )
    preds_fn = (y_probs > global_thresh_fn).astype(int)
    f1_macro_fn = f1_score(y_true, preds_fn, average='macro', zero_division=0)
    f1_micro_fn = f1_score(y_true, preds_fn, average='micro', zero_division=0)
    
    results['minimize_fn'] = {
        'global_threshold': global_thresh_fn,
        'method': 'cost_sensitive',
        'cost_fp': cost_fp,
        'cost_fn': cost_fn,
        'f1_macro': f1_macro_fn,
        'f1_micro': f1_micro_fn
    }
    print(f"   Threshold: {global_thresh_fn:.4f} | F1 Macro: {f1_macro_fn:.4f} | F1 Micro: {f1_micro_fn:.4f}")
    
    # Strategy 2: Minimize False Positives (Precision-Optimized)
    print("\n2. Minimize FP (Precision-Optimized)...")
    all_thresholds_fp = []
    all_fbetas = []
    for i in range(n_classes):
        thresh, fbeta = get_optimal_threshold_minimize_fp(
            y_true[:, i], y_probs[:, i], beta=beta
        )
        all_thresholds_fp.append(thresh)
        all_fbetas.append(fbeta)
    
    global_thresh_fp = np.mean(all_thresholds_fp)
    preds_fp = (y_probs > global_thresh_fp).astype(int)
    f1_macro_fp = f1_score(y_true, preds_fp, average='macro', zero_division=0)
    f1_micro_fp = f1_score(y_true, preds_fp, average='micro', zero_division=0)
    
    results['minimize_fp'] = {
        'global_threshold': global_thresh_fp,
        'per_class_thresholds': {
            mlb.classes_[i]: all_thresholds_fp[i] for i in range(n_classes)
        },
        'method': 'minimize_fp',
        'beta': beta,
        'avg_fbeta': np.mean(all_fbetas),
        'f1_macro': f1_macro_fp,
        'f1_micro': f1_micro_fp
    }
    print(f"   Threshold: {global_thresh_fp:.4f} | F1 Macro: {f1_macro_fp:.4f} | F1 Micro: {f1_micro_fp:.4f}")
    
    # Strategy 3: Per-Class F1 Optimization (Balanced)
    print("\n3. Per-Class F1 Optimization (Balanced)...")
    per_class_thresholds = {}
    all_f1_scores = []
    for i in range(n_classes):
        thresh, f1 = get_optimal_threshold_minimize_fp(
            y_true=y_true[:, i],
            y_probs=y_probs[:, i],
            beta=1.0  # F1 score: balanced precision and recall
        )
        per_class_thresholds[mlb.classes_[i]] = thresh
        all_f1_scores.append(f1)
    
    # Apply per-class thresholds (each class uses its own threshold)
    preds_pc = np.zeros_like(y_probs)
    for i, topic in enumerate(mlb.classes_):
        preds_pc[:, i] = (y_probs[:, i] > per_class_thresholds[topic]).astype(int)
    
    f1_macro_pc = f1_score(y_true, preds_pc, average='macro', zero_division=0)
    f1_micro_pc = f1_score(y_true, preds_pc, average='micro', zero_division=0)
    
    # Compute average threshold for reference only
    avg_thresh_pc = np.mean(list(per_class_thresholds.values()))
    
    results['per_class_f1'] = {
        'per_class_thresholds': per_class_thresholds,
        'method': 'per_class_f1',
        'avg_threshold': avg_thresh_pc,  # For reference only
        'avg_f1': np.mean(all_f1_scores),
        'f1_macro': f1_macro_pc,
        'f1_micro': f1_micro_pc
    }
    print(f"   Avg Threshold: {avg_thresh_pc:.4f} | F1 Macro: {f1_macro_pc:.4f} | F1 Micro: {f1_micro_pc:.4f}")
    
    print("\n" + "="*50)
    print("Summary of All Strategies:")
    print(f"  Minimize FN (Recall):     {global_thresh_fn:.4f} (F1 Macro: {f1_macro_fn:.4f})")
    print(f"  Minimize FP (Precision):  {global_thresh_fp:.4f} (F1 Macro: {f1_macro_fp:.4f})")
    print(f"  Per-Class F1 (Balanced):  {avg_thresh_pc:.4f} avg (F1 Macro: {f1_macro_pc:.4f})")
    print("="*50 + "\n")
    
    return results


def make_predictions(df, threshold):
    """
    Create predictions from probability columns based on a threshold.
    
    Takes a DataFrame with probability columns (prob_TopicName format) and creates
    a new column with predicted topics that exceed the threshold.
    
    Args:
        df: DataFrame with probability columns (must have 'prob_' prefix)
        threshold: Either a single float threshold (0-1) for all topics,
                  or a dict mapping topic names to their specific thresholds
                  (e.g., {'Politics': 0.3, 'Economy': 0.5})
    
    Returns:
        df: DataFrame with new column 'predicted_topic_{threshold}' or 
            'predicted_topic_per_class' containing comma-separated predicted topic names
    
    Examples:
        >>> # Single threshold for all topics
        >>> df_result = make_predictions(df_with_probs, threshold=0.5)
        >>> print(df_result['predicted_topic_0.5'].iloc[0])
        'Politics: Domestic, Religious Issues, Public Finance'
        
        >>> # Per-class thresholds
        >>> pc_thresholds = {'Politics': 0.3, 'Economy': 0.5, 'Sports': 0.7}
        >>> df_result = make_predictions(df_with_probs, threshold=pc_thresholds)
        >>> print(df_result['predicted_topic_per_class'].iloc[0])
        'Politics: Domestic, Religious Issues'
    """
    # Find all probability columns
    prob_cols = [col for col in df.columns if col.startswith('prob_')]
    
    if not prob_cols:
        raise ValueError("No probability columns found. Columns must start with 'prob_'")
    
    # Extract topic names from column names (remove 'prob_' prefix)
    topic_names = [col.replace('prob_', '') for col in prob_cols]
    
    # Check if threshold is a dictionary (per-class) or single value
    is_per_class = isinstance(threshold, dict)
    
    if is_per_class:
        # Per-class thresholds
        col_name = 'predicted_topic_per_class'
        
        def get_predictions_for_row(row):
            predictions = []
            for prob_col, topic_name in zip(prob_cols, topic_names):
                # Use topic-specific threshold if available, otherwise skip
                if topic_name in threshold:
                    if row[prob_col] > threshold[topic_name]:
                        predictions.append(topic_name)
            return ', '.join(predictions)
    else:
        # Single threshold for all topics
        col_name = f'predicted_topic_{threshold}'
        
        def get_predictions_for_row(row):
            predictions = []
            for prob_col, topic_name in zip(prob_cols, topic_names):
                if row[prob_col] > threshold:
                    predictions.append(topic_name)
            return ', '.join(predictions)
    
    df[col_name] = df.apply(get_predictions_for_row, axis=1)
    
    return df



# ==================== Helper Functions ====================

def load_trained_model(model_path, device=None):
    """
    Load a trained model, tokenizer, and MultiLabelBinarizer.
    
    Args:
        model_path: Path to saved model directory (or '.' for current directory)
        device: Torch device
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        mlb: Loaded MultiLabelBinarizer
    """
    import pickle
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to absolute path and check if it exists
    model_path = os.path.abspath(model_path)
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Use local_files_only=True to prevent API calls to HuggingFace Hub
    # This ensures we only load from local files
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True
    )
    model.to(device)
    
    # Load MultiLabelBinarizer
    mlb_path = os.path.join(model_path, 'mlb.pkl')
    if not os.path.exists(mlb_path):
        raise FileNotFoundError(f"MultiLabelBinarizer file not found: {mlb_path}")
    
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)
    
    return model, tokenizer, mlb

