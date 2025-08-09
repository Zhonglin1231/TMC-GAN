import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import math
from scipy.fft import fft, fftfreq
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# 🔧 Data Validation Tools
# ============================================
def verify_no_data_leakage(train_data, test_data, train_dates, test_dates):
    """验证数据无泄漏"""
    if train_dates and test_dates:
        if max(train_dates) >= min(test_dates):
            raise ValueError("❌ 时间泄漏: 训练集包含测试集时间范围内的数据")
    print("✅ 数据泄漏验证通过！")
    return True

# ============================================
# Enhanced Components
# ============================================
class VarianceLoss(nn.Module):
    def __init__(self, weight=1.0, target_var_ratio=1.15):
        super().__init__()
        self.weight = weight
        self.target_var_ratio = target_var_ratio
        
    def forward(self, pred, target):
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        desired_var = target_var * self.target_var_ratio
        
        if pred_var < desired_var:
            var_penalty = torch.pow((desired_var - pred_var) / (desired_var + 1e-8), 2)
        else:
            var_penalty = 0.1 * torch.abs(pred_var - desired_var) / (desired_var + 1e-8)
        
        return self.weight * var_penalty

class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)

    def forward(self, x):
        seq_len, batch_size, d_model = x.shape
        pos_encoding = self.pe[:seq_len, :] + self.learnable_pe[:seq_len, :]
        pos_encoding = pos_encoding.unsqueeze(1).expand(-1, batch_size, -1)
        return x + pos_encoding

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, pred, target, prev_values=None):
        regression_loss = nn.SmoothL1Loss()(pred, target)
        
        if prev_values is not None:
            # 计算变化方向
            pred_change = pred - prev_values
            true_change = target - prev_values
            pred_direction = torch.sign(pred_change)
            true_direction = torch.sign(true_change)
        else:
            pred_direction = torch.sign(pred)
            true_direction = torch.sign(target)
            
        direction_accuracy = torch.mean((pred_direction == true_direction).float())
        direction_loss = 1 - direction_accuracy
        return regression_loss + self.alpha * direction_loss

class FrequencyExtractor:
    def __init__(self, n_frequencies=10):
        self.n_frequencies = n_frequencies
    
    def extract_frequency_features(self, data):
        freq_features = []
        for i in range(data.shape[1]):
            series = data[:, i]
            detrended = series - np.linspace(series[0], series[-1], len(series))
            fft_vals = fft(detrended)
            freqs = fftfreq(len(detrended))
            power_spectrum = np.abs(fft_vals) ** 2
            
            positive_idx = freqs > 0
            positive_freqs = freqs[positive_idx]
            positive_power = power_spectrum[positive_idx]
            
            top_indices = np.argsort(positive_power)[-self.n_frequencies:]
            freq_vals = positive_freqs[top_indices]
            freq_magnitudes = positive_power[top_indices]
            
            if np.sum(freq_magnitudes) > 0:
                freq_magnitudes = freq_magnitudes / np.sum(freq_magnitudes)
            
            freq_features.extend(freq_vals)
            freq_features.extend(freq_magnitudes)
        
        return np.array(freq_features)

class CyclicalEncoder:
    def __init__(self):
        self.cycles = {'daily': 1, 'weekly': 5, 'monthly': 22, 'quarterly': 66, 'yearly': 252}
    
    def encode_time_cycles(self, position):
        cyclical_features = []
        for cycle_length in self.cycles.values():
            angle = 2 * np.pi * position / cycle_length
            cyclical_features.extend([np.sin(angle), np.cos(angle)])
        return np.array(cyclical_features)
    
    def get_feature_dim(self):
        return len(self.cycles) * 2

# ============================================
# Dataset
# ============================================
class EnhancedUnifiedDataset(Dataset):
    def __init__(self, stock_sequences, macro_sequences, targets, freq_features,
                 cyclical_features, prev_values=None):
        self.stock_sequences = torch.FloatTensor(stock_sequences)
        self.macro_sequences = torch.FloatTensor(macro_sequences)
        self.targets = torch.FloatTensor(targets)
        self.freq_features = torch.FloatTensor(freq_features)
        self.cyclical_features = torch.FloatTensor(cyclical_features)
        self.prev_values = torch.FloatTensor(prev_values) if prev_values is not None else None
    
    def __len__(self):
        return len(self.stock_sequences)
    
    def __getitem__(self, idx):
        items = [self.stock_sequences[idx], self.macro_sequences[idx],
                 self.targets[idx], self.freq_features[idx], self.cyclical_features[idx]]
        if self.prev_values is not None:
            items.append(self.prev_values[idx])
        return tuple(items)

# ============================================
# Models
# ============================================
class EnhancedGenerator(nn.Module):
    def __init__(self, stock_input_dim, macro_input_dim, freq_dim, cyclical_dim,
                 hidden_dim=128, output_dim=1, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = stock_input_dim
        
        self.stock_projection = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = EnhancedPositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.transformer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.stock_residual_projection = nn.Linear(self.feature_dim, output_dim)
        
        self.macro_processor = self._create_processor(macro_input_dim, 64, 32, output_dim, dropout)
        self.freq_processor = self._create_processor(freq_dim, 64, 32, output_dim, dropout)
        self.cyclical_processor = self._create_processor(cyclical_dim, 32, 16, output_dim, dropout)
        
        self.fusion_weights = nn.Parameter(torch.tensor([2.0, 0.3, 0.2, 0.1]))
        self.importance_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Softmax(dim=-1)
        )
    
    def _create_processor(self, input_dim, hidden1, hidden2, output_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, output_dim),
            nn.Tanh()
        )
    
    def forward(self, stock_sequences, macro_sequences, freq_features, cyclical_features, noise=None):
        batch_size = stock_sequences.size(0)
        
        stock_residual_input = stock_sequences.mean(dim=1)
        stock_projected = self.stock_projection(stock_sequences)
        stock_encoded = stock_projected.transpose(0, 1)
        stock_encoded = self.pos_encoder(stock_encoded)
        stock_encoded = stock_encoded.transpose(0, 1)
        stock_transformed = self.transformer_encoder(stock_encoded)
        stock_final = stock_transformed[:, -1, :]
        
        main_prediction = self.transformer_head(stock_final)
        stock_residual = self.stock_residual_projection(stock_residual_input)
        enhanced_main_prediction = main_prediction + stock_residual * 0.1
        
        macro_flat = macro_sequences.view(batch_size, -1)
        macro_signal = self.macro_processor(macro_flat)
        freq_signal = self.freq_processor(freq_features)
        cyclical_signal = self.cyclical_processor(cyclical_features)
        
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        dynamic_weights = self.importance_gate(stock_final)
        combined_weights = 0.7 * fusion_weights + 0.3 * dynamic_weights
        
        signals = torch.stack([
            enhanced_main_prediction.squeeze(-1),
            macro_signal.squeeze(-1),
            freq_signal.squeeze(-1),
            cyclical_signal.squeeze(-1)
        ], dim=1)
        
        final_prediction = torch.sum(combined_weights.unsqueeze(-1) * signals.unsqueeze(-1), dim=1)
        return final_prediction, combined_weights

class EnhancedDiscriminator(nn.Module):
    def __init__(self, stock_input_dim, macro_input_dim, freq_dim, cyclical_dim,
                 hidden_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.stock_projection = nn.Sequential(
            nn.Linear(stock_input_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = EnhancedPositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.aux_processor = nn.Sequential(
            nn.Linear(macro_input_dim + freq_dim + cyclical_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, stock_sequences, macro_sequences, targets, freq_features, cyclical_features):
        batch_size = stock_sequences.size(0)
        seq_len = stock_sequences.size(1)
        
        targets_expanded = targets.unsqueeze(1).expand(-1, seq_len, -1)
        stock_with_targets = torch.cat([stock_sequences, targets_expanded], dim=2)
        
        stock_projected = self.stock_projection(stock_with_targets)
        stock_encoded = stock_projected.transpose(0, 1)
        stock_encoded = self.pos_encoder(stock_encoded)
        stock_encoded = stock_encoded.transpose(0, 1)
        stock_transformed = self.transformer_encoder(stock_encoded)
        stock_final = stock_transformed.mean(dim=1)
        
        macro_flat = macro_sequences.view(batch_size, -1)
        aux_features = torch.cat([macro_flat, freq_features, cyclical_features], dim=1)
        aux_processed = self.aux_processor(aux_features)
        
        combined = torch.cat([stock_final, aux_processed], dim=1)
        logits = self.classifier(combined)
        output = torch.sigmoid(logits)
        return torch.clamp(output, 0, 1)

# ============================================
# Data Preprocessing
# ============================================
def preprocess_data(file_path, window_size=20, use_returns=False, test_split=0.2, dataset ='BTC'):
    """预处理数据，防止数据泄漏"""
    print(f"\n🔧 数据预处理 - 窗口: {window_size}, 收益率: {use_returns}")
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').replace('-', np.nan).dropna().reset_index(drop=True)
    print(f"📊 数据加载完成: {len(df)} 行")
    
    stock_cols = [f'Open_{dataset}', f'High_{dataset}', f'Low_{dataset}', f'Close_{dataset}', f'Adj Close_{dataset}', f'Volume_{dataset}']
    macro_cols = ['Close_Gold', 'Close_Oil', 'Close_VIX']
    
    for col in stock_cols + macro_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    
    if use_returns:
        price_cols = [f'Open_{dataset}', f'High_{dataset}', f'Low_{dataset}', f'Close_{dataset}', f'Adj Close_{dataset}']
        for col in price_cols:
            df[f'{col}_return'] = df[col].pct_change()
        df[f'Volume_{dataset}_change'] = df[f'Volume_{dataset}'].pct_change()
        df['price_range'] = (df[f'High_{dataset}'] - df[f'Low_{dataset}']) / df[f'Close_{dataset}']
        df['price_position'] = (df[f'Close_{dataset}'] - df[f'Low_{dataset}']) / (df[f'High_{dataset}'] - df[f'Low_{dataset}'])
        df['price_range'] = df['price_range'].replace([np.inf, -np.inf], 0)
        df['price_position'] = df['price_position'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
        df = df.iloc[1:].reset_index(drop=True)
        
        stock_feature_cols = [col for col in df.columns if '_return' in col or '_change' in col] + \
                           ['price_range', 'price_position']
        stock_features_raw = df[stock_feature_cols].values
        targets_raw = df[f'Close_{dataset}_return'].values
    else:
        stock_features_raw = df[stock_cols].values
        targets_raw = df[f'Close_{dataset}'].values
    
    macro_features_raw = df[macro_cols].values
    
    X_stock_raw, X_macro_raw, y_raw = [], [], []
    freq_features_raw, cyclical_features_raw, prev_values_raw = [], [], []
    base_prices, date_correspondences = [], []
    
    freq_extractor = FrequencyExtractor(n_frequencies=5)
    cyclical_encoder = CyclicalEncoder()
    
    for i in range(window_size, len(stock_features_raw) - 1):
        window_start = i - window_size + 1
        window_end = i
        target_idx = i + 1
        
        stock_window = stock_features_raw[window_start:window_end+1]
        macro_window = macro_features_raw[window_start:window_end+1]
        
        X_stock_raw.append(stock_window)
        X_macro_raw.append(macro_window)
        y_raw.append(targets_raw[target_idx])
        prev_values_raw.append(targets_raw[window_end])
        base_prices.append(df.loc[window_end, f'Close_{dataset}'])
        
        freq_features_raw.append(freq_extractor.extract_frequency_features(stock_window))
        cyclical_features_raw.append(cyclical_encoder.encode_time_cycles(target_idx))
        
        date_correspondences.append({
            'input_end_date': df.loc[window_end, 'Date'],
            'target_date': df.loc[target_idx, 'Date'],
            'target_close_price': df.loc[target_idx, f'Close_{dataset}']
        })
    
    X_stock_raw = np.array(X_stock_raw)
    X_macro_raw = np.array(X_macro_raw)
    y_raw = np.array(y_raw).reshape(-1, 1)
    prev_values_raw = np.array(prev_values_raw).reshape(-1, 1)
    freq_features_raw = np.array(freq_features_raw)
    cyclical_features_raw = np.array(cyclical_features_raw)
    base_prices = np.array(base_prices)
    
    split_idx = int(len(X_stock_raw) * (1 - test_split))
    
    if use_returns:
        stock_scaler = RobustScaler()
        macro_scaler = StandardScaler()
        freq_scaler = StandardScaler()
        target_scaler = None
    else:
        stock_scaler = MinMaxScaler(feature_range=(0, 1))
        macro_scaler = MinMaxScaler(feature_range=(0, 1))
        freq_scaler = StandardScaler()
        target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    stock_train_flat = X_stock_raw[:split_idx].reshape(-1, X_stock_raw.shape[-1])
    macro_train_flat = X_macro_raw[:split_idx].reshape(-1, X_macro_raw.shape[-1])
    
    stock_scaler.fit(stock_train_flat)
    macro_scaler.fit(macro_train_flat)
    freq_scaler.fit(freq_features_raw[:split_idx])
    if target_scaler:
        target_scaler.fit(y_raw[:split_idx])
    
    def transform_data(X_stock, X_macro, y, prev, freq, cyc, start_idx, end_idx):
        stock_flat = X_stock[start_idx:end_idx].reshape(-1, X_stock.shape[-1])
        macro_flat = X_macro[start_idx:end_idx].reshape(-1, X_macro.shape[-1])
        
        X_stock_scaled = stock_scaler.transform(stock_flat).reshape(X_stock[start_idx:end_idx].shape)
        X_macro_scaled = macro_scaler.transform(macro_flat).reshape(X_macro[start_idx:end_idx].shape)
        freq_scaled = freq_scaler.transform(freq[start_idx:end_idx])
        
        if target_scaler:
            y_scaled = target_scaler.transform(y[start_idx:end_idx])
            prev_scaled = target_scaler.transform(prev[start_idx:end_idx])
        else:
            y_scaled = y[start_idx:end_idx]
            prev_scaled = prev[start_idx:end_idx]
        
        return (X_stock_scaled, X_macro_scaled, y_scaled, prev_scaled, 
                freq_scaled, cyc[start_idx:end_idx])
    
    train_data = transform_data(X_stock_raw, X_macro_raw, y_raw, prev_values_raw,
                               freq_features_raw, cyclical_features_raw, 0, split_idx)
    test_data = transform_data(X_stock_raw, X_macro_raw, y_raw, prev_values_raw,
                              freq_features_raw, cyclical_features_raw, split_idx, len(X_stock_raw))
    
    train_dates = [d['target_date'] for d in date_correspondences[:split_idx]]
    test_dates = [d['target_date'] for d in date_correspondences[split_idx:]]
    verify_no_data_leakage(train_data[0], test_data[0], train_dates, test_dates)
    
    scalers = {
        'stock_scaler': stock_scaler,
        'macro_scaler': macro_scaler,
        'freq_scaler': freq_scaler,
        'target_scaler': target_scaler
    }
    
    return (*train_data, *test_data, scalers, df, date_correspondences, 
            base_prices[:split_idx], base_prices[split_idx:])

# ============================================
# Training
# ============================================
def spectral_loss(generated_sequence, real_sequence, weight=1.0):
    """
    频谱损失 - 使用PyTorch FFT以保持梯度流.
    比较两个序列的频域表示。
    确保输入的序列长度大于1。
    """
    seq_len = generated_sequence.size(-1)

    # If sequence is too short for a meaningful FFT, do not compute loss
    if seq_len <= 1:
        return torch.tensor(0.0, device=generated_sequence.device)

    # Calculate Fast Fourier Transform
    gen_fft = torch.fft.fft(generated_sequence, dim=-1)
    real_fft = torch.fft.fft(real_sequence, dim=-1)

    # Calculate magnitude spectrum |F(x)|
    gen_fft_magnitude = torch.abs(gen_fft)
    real_fft_magnitude = torch.abs(real_fft)

    # Calculate Mean Squared Error between the magnitude spectra
    # We only need to compare the first half of the spectrum due to symmetry
    loss = F.mse_loss(gen_fft_magnitude[:, :seq_len//2], real_fft_magnitude[:, :seq_len//2])
    
    return loss * weight

def train_gan(generator, discriminator, dataloader, num_epochs, g_lr=0.0001, d_lr=0.0001,
              lambda_reg=5.0, lambda_variance=1.0, lambda_spectral=1.0, lambda_directional=0.5,
              clip_value=1.0, use_directional_loss=True):
    """训练GAN"""
    adversarial_loss = nn.BCELoss()
    regression_loss = nn.SmoothL1Loss()
    variance_loss = VarianceLoss(weight=lambda_variance)
    directional_criterion = DirectionalLoss(alpha=lambda_directional) if use_directional_loss else None
    
    g_optimizer = optim.AdamW(generator.parameters(), lr=g_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    d_optimizer = optim.AdamW(discriminator.parameters(), lr=d_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=num_epochs)
    d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=num_epochs)
    
    losses = {'g': [], 'd': [], 'reg': [], 'var': [], 'dir': [], 'spec': []}
    fusion_weights_history = []
    
    print("\n🚀 开始训练...")
    
    for epoch in range(num_epochs):
        epoch_losses = {k: 0 for k in losses}
        epoch_fusion_weights = []
        
        for batch_data in dataloader:
            if len(batch_data) == 6:
                stock_seq, macro_seq, targets, freq_feat, cyc_feat, prev_vals = batch_data
                prev_vals = prev_vals.to(device)
            else:
                stock_seq, macro_seq, targets, freq_feat, cyc_feat = batch_data
                prev_vals = None

            batch_size = stock_seq.size(0)
            stock_seq = stock_seq.to(device)
            macro_seq = macro_seq.to(device)
            targets = targets.to(device)
            freq_feat = freq_feat.to(device)
            cyc_feat = cyc_feat.to(device)
            
            smoothing = 0.05 * (1 - epoch / num_epochs)
            real_labels = torch.ones(batch_size, 1).to(device) * (1 - smoothing)
            fake_labels = torch.zeros(batch_size, 1).to(device) + smoothing
            
            d_optimizer.zero_grad()
            real_output = discriminator(stock_seq, macro_seq, targets, freq_feat, cyc_feat)
            d_real_loss = adversarial_loss(real_output, real_labels)
            
            with torch.no_grad():
                fake_targets, _ = generator(stock_seq, macro_seq, freq_feat, cyc_feat)
            
            fake_output = discriminator(stock_seq, macro_seq, fake_targets.detach(), freq_feat, cyc_feat)

            d_fake_loss = adversarial_loss(fake_output, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            fake_targets, fusion_weights = generator(stock_seq, macro_seq, freq_feat, cyc_feat)
            fake_output = discriminator(stock_seq, macro_seq, fake_targets, freq_feat, cyc_feat)
            
            g_adv_loss = adversarial_loss(fake_output, real_labels)
            g_reg_loss = regression_loss(fake_targets, targets)
            g_var_loss = variance_loss(fake_targets, targets)
            
            g_dir_loss = torch.tensor(0.0, device=fake_targets.device)
            if use_directional_loss and directional_criterion and prev_vals is not None:
                g_dir_loss = directional_criterion(fake_targets, targets, prev_vals)
            
            # Spectral loss calculation
            g_spectral_loss = torch.tensor(0.0).to(device)
            if lambda_spectral > 0:
                target_feature_index = 3 
                
                if stock_seq.shape[2] > target_feature_index:
                    real_history_target_feature = stock_seq[:, :, target_feature_index]
                    fake_sequence_for_fft = torch.clone(real_history_target_feature)
                    fake_sequence_for_fft[:, -1] = fake_targets.squeeze(-1)
                    real_sequence_for_fft = real_history_target_feature
                    g_spectral_loss = spectral_loss(fake_sequence_for_fft, real_sequence_for_fft, weight=lambda_spectral)
                else:
                    if not hasattr(spectral_loss, 'has_warned'):
                        print("⚠️ WARNING: `target_feature_index` for spectral_loss is out of bounds. Skipping spectral loss.")
                        spectral_loss.has_warned = True
            
            weight_reg = -torch.mean(torch.log(fusion_weights[:, 0] + 1e-8)) * 0.01
            adv_weight = 0.05 + 0.2 * (epoch / num_epochs)
            
            g_loss = (adv_weight * g_adv_loss + lambda_reg * g_reg_loss + 
                     g_var_loss + g_dir_loss + g_spectral_loss + weight_reg)
            
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            g_optimizer.step()
            
            epoch_losses['g'] += g_loss.item()
            epoch_losses['d'] += d_loss.item()
            epoch_losses['reg'] += g_reg_loss.item()
            epoch_losses['var'] += g_var_loss.item()
            epoch_losses['dir'] += g_dir_loss.item()
            epoch_losses['spec'] += g_spectral_loss.item()
            epoch_fusion_weights.append(fusion_weights.mean(dim=0).detach().cpu().numpy())
        
        g_scheduler.step()
        d_scheduler.step()
        
        for k in losses:
            losses[k].append(epoch_losses[k] / len(dataloader))
        fusion_weights_history.append(np.mean(epoch_fusion_weights, axis=0))
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] G: {losses["g"][-1]:.4f}, D: {losses["d"][-1]:.4f}, '
                  f'Reg: {losses["reg"][-1]:.4f}, Var: {losses["var"][-1]:.4f}, Spec: {losses["spec"][-1]:.4f}')
    
    return losses, fusion_weights_history

# ============================================
# Evaluation    
# ============================================
def predict_unified(generator, stock_seq, macro_seq, freq_feat, cyc_feat, scalers, 
                   last_price=None, use_returns=False):
    """预测函数"""
    generator.eval()
    with torch.no_grad():
        inputs = [torch.FloatTensor(x).unsqueeze(0).to(device) 
                 for x in [stock_seq, macro_seq, freq_feat, cyc_feat]]
        prediction, fusion_weights = generator(*inputs)
        prediction = prediction.cpu().numpy()[0, 0]
    
    if use_returns and last_price is not None:
        if scalers['target_scaler']:
            original_return = scalers['target_scaler'].inverse_transform([[prediction]])[0, 0]
        else:
            original_return = prediction
        predicted_price = last_price * (1 + original_return)
        return predicted_price, fusion_weights, original_return
    else:
        if scalers['target_scaler']:
            original_price = scalers['target_scaler'].inverse_transform([[prediction]])[0, 0]
        else:
            original_price = prediction
        return original_price, fusion_weights, None

def calculate_metrics(predictions, actuals, prev_values=None, prefix=""):
    """计算评估指标 - 修正了价格方向准确性计算"""
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((actuals[non_zero_mask] - predictions[non_zero_mask]) / 
                             actuals[non_zero_mask])) * 100
    else:
        mape = float('inf')
    
    correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
    
    # 修正的方向准确性计算
    if prefix == "Return":
        # 对于收益率，直接比较符号
        direction_accuracy = np.mean(np.sign(predictions) == np.sign(actuals)) * 100
    else:
        # 对于价格，比较相对于前一天的变化方向
        if prev_values is not None and len(prev_values) == len(predictions):
            # 实际变化方向：actual[t] - actual[t-1]
            actual_changes = actuals - prev_values
            # 预测变化方向：predicted[t] - actual[t-1]  
            pred_changes = predictions - prev_values
            
            actual_directions = np.sign(actual_changes)
            pred_directions = np.sign(pred_changes)
            
            direction_accuracy = np.mean(actual_directions == pred_directions) * 100
        else:
            # 如果没有前一天的值，使用连续差分（但这不是最准确的方法）
            if len(actuals) > 1:
                actual_dir = np.diff(actuals) > 0
                pred_dir = np.diff(predictions) > 0
                direction_accuracy = np.mean(actual_dir == pred_dir) * 100
            else:
                direction_accuracy = 0
    
    return {
        f'{prefix} MAE': mae,
        f'{prefix} RMSE': rmse,
        f'{prefix} MAPE': mape if mape != float('inf') else None,
        f'{prefix} Direction Accuracy': direction_accuracy,
        f'{prefix} Correlation': correlation
    }


