"""
Layer 1: Momentum/Sentiment Analysis for Long-term Stock Investment
Features exponentially weighted recent signals with LSTM + Attention mechanism
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MomentumDataProcessor:
    """Handles data fetching and momentum feature engineering"""
    
    def __init__(self, lookback_days=60, decay_alpha=0.9):
        self.lookback_days = lookback_days
        self.decay_alpha = decay_alpha
        self.scaler = StandardScaler()
        
    def fetch_stock_data(self, symbol, period="2y"):
        """Fetch stock price and volume data"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_momentum_features(self, data):
        """Calculate momentum features with exponential weighting"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based momentum
        features['price_momentum_1d'] = data['Close'].pct_change(1)
        features['price_momentum_5d'] = data['Close'].pct_change(5)
        features['price_momentum_20d'] = data['Close'].pct_change(20)
        
        # Volume-based signals
        features['volume_sma_20'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
        features['volume_momentum'] = data['Volume'].pct_change(5)
        
        # Volatility signals
        features['returns'] = data['Close'].pct_change()
        features['volatility_20d'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = (features['returns'].rolling(5).std() / 
                                       features['volatility_20d'])
        
        # Price position relative to recent highs/lows
        features['high_20d'] = data['High'].rolling(20).max()
        features['low_20d'] = data['Low'].rolling(20).min()
        features['price_position'] = ((data['Close'] - features['low_20d']) / 
                                     (features['high_20d'] - features['low_20d']))
        
        # RSI-like momentum
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Technical indicators
        features['sma_10'] = data['Close'].rolling(10).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['sma_ratio'] = features['sma_10'] / features['sma_50']
        
        # MACD-like feature
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        return features
    
    def apply_exponential_weights(self, features):
        """Apply exponential decay weighting to emphasize recent data"""
        weighted_features = features.copy()
        
        for col in features.columns:
            if col in ['price_momentum_1d', 'price_momentum_5d', 'volume_ratio', 
                      'volatility_ratio', 'macd_histogram']:
                # Apply stronger weighting to momentum signals
                weights = np.array([self.decay_alpha ** i for i in range(len(features))][::-1])
                weights = weights / weights.sum()  # Normalize
                
                # Apply exponential weighting
                weighted_values = features[col].fillna(0) * weights
                weighted_features[f'{col}_weighted'] = weighted_values
        
        return weighted_features
    
    def create_sequences(self, features, target_col='price_momentum_5d', sequence_length=None):
        """Create sequences for LSTM training"""
        if sequence_length is None:
            sequence_length = self.lookback_days
            
        # Clean data
        features = features.dropna()
        
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data. Need {sequence_length}, got {len(features)}")
        
        # Select feature columns (exclude target and some derived columns)
        feature_cols = [col for col in features.columns 
                       if col not in [target_col, 'returns', 'high_20d', 'low_20d', 'volume_sma_20']]
        
        # Prepare features and targets
        X_data = features[feature_cols].values
        y_data = features[target_col].shift(-5).ffill().values  # Predict 5 days ahead
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:(i + sequence_length)])
            y_sequences.append(y_data[i + sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences), feature_cols


class StockMomentumDataset(Dataset):
    """PyTorch Dataset for stock momentum data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMWithAttention(nn.Module):
    """LSTM with Attention Mechanism for Momentum Analysis"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, 
                                             dropout=dropout, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Self-attention (helps model focus on important time steps)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last attention output
        last_output = attn_out[:, -1, :]
        
        # Final prediction layers
        out = self.dropout(self.relu(self.fc1(last_output)))
        out = self.fc2(out)
        
        return out.squeeze(), attn_weights


class MomentumPredictor:
    """Main class for momentum-based stock prediction"""
    
    def __init__(self, lookback_days=60, hidden_size=64, learning_rate=0.001):
        self.lookback_days = lookback_days
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        self.data_processor = MomentumDataProcessor(lookback_days)
        self.model = None
        self.feature_cols = None
        
    def prepare_data(self, symbol, test_size=0.2):
        """Prepare training and testing data for a stock"""
        print(f"Preparing data for {symbol}...")
        
        # Fetch and process data
        raw_data = self.data_processor.fetch_stock_data(symbol)
        if raw_data is None:
            return None
            
        features = self.data_processor.calculate_momentum_features(raw_data)
        features = self.data_processor.apply_exponential_weights(features)
        
        # Create sequences
        X, y, feature_cols = self.data_processor.create_sequences(features)
        self.feature_cols = feature_cols
        
        # Train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """Train the LSTM attention model"""
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = LSTMWithAttention(input_size, self.hidden_size)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Create data loaders
        train_dataset = StockMomentumDataset(X_train, y_train)
        test_dataset = StockMomentumDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        print("Starting training...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions, _ = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    predictions, _ = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    test_loss += loss.item()
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        print("Training completed!")
        return train_losses, test_losses
    
    def get_momentum_score(self, symbol, days_ahead=5):
        """Get momentum score for a stock (0-1 scale)"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Get recent data
        raw_data = self.data_processor.fetch_stock_data(symbol, period="6mo")
        features = self.data_processor.calculate_momentum_features(raw_data)
        features = self.data_processor.apply_exponential_weights(features)
        
        # Get latest sequence
        X_data = features[self.feature_cols].dropna().iloc[-self.lookback_days:].values
        X_scaled = self.data_processor.scaler.transform(X_data)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            momentum_raw, attention_weights = self.model(X_tensor)
            
        # Convert to 0-1 score (using sigmoid transformation)
        momentum_score = torch.sigmoid(momentum_raw * 10).item()  # Scale and sigmoid
        
        return {
            'momentum_score': momentum_score,
            'raw_prediction': momentum_raw.item(),
            'attention_weights': attention_weights.numpy(),
            'confidence': 1.0 - abs(0.5 - momentum_score) * 2  # Higher when score is closer to 0 or 1
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = MomentumPredictor(lookback_days=60, hidden_size=64)
    
    # Test with a popular stock
    symbol = "AAPL"
    
    try:
        # Prepare data
        train_data = predictor.prepare_data(symbol, test_size=0.2)
        if train_data is not None:
            X_train, X_test, y_train, y_test = train_data
            
            # Train model
            train_losses, test_losses = predictor.train_model(
                X_train, y_train, X_test, y_test, epochs=50, batch_size=16
            )
            
            # Get momentum score
            result = predictor.get_momentum_score(symbol)
            print(f"\n{symbol} Momentum Analysis:")
            print(f"Momentum Score: {result['momentum_score']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Raw Prediction: {result['raw_prediction']:.6f}")
            
    except Exception as e:
        print(f"Error: {e}")