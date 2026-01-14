# MLP_train.py
import torch
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')  # 用于无图形界面的环境
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm

def main():
    # ------------------------------
    # 1. Set random seeds
    # ------------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ------------------------------
    # 2. Data Loading and Preprocessing
    # ------------------------------

    # Load the CSV data and perform NaN filling
    df_herb_nes = pd.read_csv("df_herb_nes_mini.txt", sep='\t')
    df_herb_nes.fillna(0, inplace=True)
    df_numeric = df_herb_nes.iloc[:, 1:]  # Exclude the first column (assuming it's text)

    # Define a normalization function to scale data to [-1, 1]
    def normalize_column(col):
        return 2 * (col - col.min()) / (col.max() - col.min()) - 1

    df_numeric = df_numeric.apply(normalize_column, axis=0)

    def generate_matrix(rows, cols):
        """
        Generate a matrix of shape (rows, cols) with values in (-1, 1).
        Each column is drawn from either a truncated normal distribution 
        or a bimodal (two truncated normals combined), with 50% probability for each.

        :param rows: number of rows
        :param cols: number of columns
        :return: A matrix of shape (rows, cols) with float32 values in (-1, 1).
        """
        matrix = np.empty((rows, cols), dtype=np.float32)
        
        for col in tqdm(range(cols), desc="Generating matrix"):
            # Randomly choose between normal distribution (50% chance) or a bimodal distribution
            if np.random.rand() < 0.5:  
                # Single truncated normal distribution
                mu = np.random.uniform(-0.7, 0.7)
                sigma = np.random.uniform(0.1, 0.3)
                a, b = (-1 - mu) / sigma, (1 - mu) / sigma
                data = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=rows)
            else:
                # Bimodal distribution
                mu1 = np.random.uniform(-0.8, -0.2)
                mu2 = np.random.uniform(0.2, 0.8)
                sigma1 = np.random.uniform(0.05, 0.15)
                sigma2 = np.random.uniform(0.05, 0.15)
                w1 = np.random.uniform(0.3, 0.7)
                size1 = int(rows * w1)
                size1 = max(1, min(size1, rows - 1))
                size2 = rows - size1
                
                # Generate truncated normal for the first mode
                a1, b1 = (-1 - mu1) / sigma1, (1 - mu1) / sigma1
                data1 = truncnorm.rvs(a1, b1, loc=mu1, scale=sigma1, size=size1)
                
                # Generate truncated normal for the second mode
                a2, b2 = (-1 - mu2) / sigma2, (1 - mu2) / sigma2
                data2 = truncnorm.rvs(a2, b2, loc=mu2, scale=sigma2, size=size2)
                
                data = np.concatenate([data1, data2])
                np.random.shuffle(data)  # Shuffle to mix the two modes
            
            # Randomly flip the sign of the generated column
            sign = 1 if np.random.rand() < 0.5 else -1
            matrix[:, col] = data * sign
        
        # Clip the matrix values to stay within (-1, 1), avoiding floating-point overflow
        matrix = np.clip(matrix, -1 + 1e-8, 1 - 1e-8)
        return matrix

    # Generate the matrix
    print("Generating data matrix...")
    data = generate_matrix(2683, 20000)

    # Define input and output dimensions
    input_size = data.shape[0]          # e.g., 2683 (number of features)
    output_size = df_numeric.shape[1]   # e.g., 1800 (number of columns in the CSV)

    # ------------------------------
    # 3. Create Dataset and DataLoader
    # ------------------------------
    # Note: 'data' has the shape (input_size, num_samples), 
    # so each sample is effectively one column of 'data'.
    samples = data.T  # shape: (num_samples, input_size)

    # Use sklearn to split into train/test sets
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

    class HerbDataset(Dataset):
        """
        A simple Dataset wrapper for our sample data.
        Expects 'samples' with shape (num_samples, input_size).
        """
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return self.samples.shape[0]

        def __getitem__(self, idx):
            return self.samples[idx]

    train_dataset = HerbDataset(train_samples)
    test_dataset = HerbDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    # Construct a tensor for df_herb_nes to be used in the custom loss function
    # This has shape (output_size, input_size) because we use the transpose
    df_herb_nes_tensor = torch.tensor(df_numeric.values.T, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    df_herb_nes_tensor = df_herb_nes_tensor.to(device)

    # ------------------------------
    # 4. Model Definition
    # ------------------------------
    class Net(nn.Module):
        """
        A fully connected network with variable hidden sizes, dropout, 
        and a final softplus activation to ensure non-negative outputs.
        """
        def __init__(self, input_size, hidden_sizes=(2048, 1024, 512, 256), output_size=1800, dropout_p=0.3):
            super(Net, self).__init__()
            layers = []
            in_size = input_size
            for h_size in hidden_sizes:
                layers.append(nn.Linear(in_size, h_size))
                layers.append(nn.BatchNorm1d(h_size))
                layers.append(nn.ReLU())  # Using ReLU as activation
                layers.append(nn.Dropout(dropout_p))
                in_size = h_size
            layers.append(nn.Linear(in_size, output_size))
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.layers(x)
            # Use softplus to ensure non-negative output
            x = F.softplus(x)
            return x

    # ------------------------------
    # 5. Custom Loss Function
    # ------------------------------
    # Original text indicates we want "the Spearman's correlation between weighted_data and inputs 
    # to be as negative as possible", but here it's implemented as an absolute product-based approach.
    def custom_loss(logits, inputs, df_tensor, k, std_weight=1):
        """
        logits: (batch_size, output_size)
        inputs: (batch_size, input_size)
        df_tensor: (output_size, input_size)
        k: The number of top logits to keep
        std_weight: An unused parameter here; keep for potential extensions

        1. Select the top-k values in each row of 'logits' (by value).
        2. Construct a sparse version of 'logits' that only keeps these top-k entries.
        3. Multiply the sparse logits with 'df_tensor' to get 'weighted_data'.
        4. Combine 'weighted_data' with 'inputs' (simply add).
        5. Compute loss = mean of ( |(weighted_data + inputs) * inputs| / sum(inputs^2) ).

        The idea is to measure how the added terms interact with the original inputs.
        """
        # Select top-k values per row
        topk_values, topk_indices = torch.topk(logits, k=k, dim=1)
        modified_logits = torch.zeros_like(logits)
        modified_logits.scatter_(1, topk_indices, topk_values)
        
        # Multiply with df_tensor to get weighted_data
        weighted_data = torch.matmul(modified_logits, df_tensor)  # shape: (batch_size, input_size)
        total = weighted_data + inputs

        epsilon = 1e-8  # Prevent division by zero

        # Numerator: sum of absolute products of 'total' and 'inputs'
        numerator = torch.sum(torch.abs(total * inputs), dim=1)
        
        # Denominator: sum of squares of inputs
        denominator = torch.sum(inputs * inputs, dim=1)
        
        # Compute per-sample loss and average
        loss_per_sample = numerator / (denominator + epsilon)
        return loss_per_sample.mean()

    # ------------------------------
    # 6. Training and Evaluation
    # ------------------------------
    max_epochs = 1000    # Maximum training epochs
    patience = 5         # Early stopping patience
    k = 50               # Modify 'k' if needed

    model = Net(input_size=input_size, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_loss = float('inf')
    epochs_without_improve = 0
    train_losses = []
    epoch = 0

    print("Starting training...")
    while epoch < max_epochs:
        epoch += 1
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = custom_loss(logits, batch, df_herb_nes_tensor, k)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        print(f"Iteration 1 (k={k}) | Epoch {epoch}: "
              f"Avg Train Loss = {avg_loss:.4f} | "
              f"LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for significant improvement
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        
        # Early stopping
        if epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break

    # Testing phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            batch = batch.to(device)
            logits = model(batch)
            loss = custom_loss(logits, batch, df_herb_nes_tensor, k)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Iteration 1 (k={k}) | Test Loss: {avg_test_loss:.4f}")

    # Plot training curve and save the figure
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(train_losses, label='Train Loss')
    plt.axhline(y=avg_test_loss, color='r', linestyle='--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train/Test Loss for k = {k}')
    plt.legend()
    plt.text(0.05, 0.95, 
             f"Final Epoch: {epoch}\nFinal Train Loss: {train_losses[-1]:.4f}\nTest Loss: {avg_test_loss:.4f}",
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    plt.savefig(f'weighted_loss_plot_k_{k}.png')
    plt.close()

    # Save the model parameters
    torch.save(model.state_dict(), f'weighted_herb_model_re_{k}.pth')
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()