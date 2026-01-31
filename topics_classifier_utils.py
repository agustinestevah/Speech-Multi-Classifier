import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix
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
                with autocast(device_type='cuda'):
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


def evaluate_trained_model(
    model,
    train_loader,
    val_loader,
    mlb,
    device=None,
    threshold=0.5
):
    """
    Evaluate trained model on both training and validation sets.
    Reports Macro F1 scores and confusion matrix.
    
    Args:
        model: Trained model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        mlb: MultiLabelBinarizer
        device: Torch device
        threshold: Probability threshold for predictions (default: 0.5)
    
    Returns:
        results: Dictionary with metrics and confusion matrices
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    print("\n=== Evaluating Model ===")
    
    # Function to get predictions for a dataloader
    def get_predictions(dataloader, name):
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                #probabilities from logits
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels)
    
    # Get predictions for train and validation
    train_preds, train_labels = get_predictions(train_loader, "Train")
    val_preds, val_labels = get_predictions(val_loader, "Validation")
    
    # Calculate F1 scores
    train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"F1 Scores (Macro):")
    print(f"  Train: {train_f1_macro:.4f}")
    print(f"  Test:  {val_f1_macro:.4f}")
    print(f"{'='*50}")
    
    # Generate confusion matrix for validation set
    print(f"\nConfusion Matrix for Test Set:")
    print(f"{'='*50}")
    
    cm_multi = multilabel_confusion_matrix(val_labels, val_preds)
    
    for i, topic in enumerate(mlb.classes_):
        tn, fp, fn, tp = cm_multi[i].ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nTopic: {topic}")
        print(f"  TP: {tp:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TN: {tn:4d}")
        print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    
    print(f"\n{'='*50}")
    
    # Return results dictionary
    results = {
        'train_f1_macro': train_f1_macro,
        'val_f1_macro': val_f1_macro,
        'train_preds': train_preds,
        'train_labels': train_labels,
        'val_preds': val_preds,
        'val_labels': val_labels,
        'confusion_matrices': cm_multi
    }
    
    return results


# ==================== Prediction Functions ====================

def predict_topic_probabilities(
    df,
    model,
    tokenizer,
    mlb,
    text_col='speech',
    max_length=512,
    chunk_size=400,
    device=None
):
    """
    Predict topic probabilities for each row in dataframe.
    Handles long speeches by splitting into chunks and averaging probabilities.
    
    Args:
        df: DataFrame with text column
        model: Trained model
        tokenizer: Tokenizer
        mlb: MultiLabelBinarizer
        text_col: Name of text column
        max_length: Maximum sequence length
        chunk_size: Max tokens per chunk for long texts (before splitting)
        device: Torch device
    
    Returns:
        df_result: DataFrame with probability columns for each topic
    """
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


# ==================== Helper Functions ====================

def load_trained_model(model_path, device=None):
    """
    Load a trained model, tokenizer, and MultiLabelBinarizer.
    
    Args:
        model_path: Path to saved model directory
        device: Torch device
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        mlb: Loaded MultiLabelBinarizer
    """
    import pickle
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # Load MultiLabelBinarizer
    with open(os.path.join(model_path, 'mlb.pkl'), 'rb') as f:
        mlb = pickle.load(f)
    
    return model, tokenizer, mlb

