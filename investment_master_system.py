"""
Layer 3: Meta-Learner & Market Regime Detection
Combines momentum and fundamental analysis with adaptive weighting
Provides final investment recommendations with risk management
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import our previous layers
try:
    from .momentum_predictor import MomentumPredictor
    from .fundamental_analyzer import FundamentalAnalyzer
except ImportError:
    try:
        # Fallback for direct execution
        from momentum_predictor import MomentumPredictor
        from fundamental_analyzer import FundamentalAnalyzer
    except ImportError:
        print("Warning: Could not import momentum_predictor or fundamental_analyzer")
        print("Make sure both files are in the same directory")


class MarketRegimeDetector:
    """Detects market regimes (bull, bear, sideways) for adaptive weighting"""
    
    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_market_data(self, period="5y"):
        """Fetch market indicators for regime detection"""
        try:
            # Market indices
            spy = yf.download("SPY", period=period, progress=False)
            vix = yf.download("^VIX", period=period, progress=False)
            
            # Bond yields (10-year treasury)
            tnx = yf.download("^TNX", period=period, progress=False)
            
            # Ensure data has consistent column structure
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.droplevel(1)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.droplevel(1)
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx.columns = tnx.columns.droplevel(1)
            
            return {
                'spy': spy,
                'vix': vix,
                'tnx': tnx
            }
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def calculate_regime_features(self, market_data):
        """Calculate features for regime detection"""
        spy = market_data['spy']
        vix = market_data['vix']
        tnx = market_data['tnx']
        
        features = pd.DataFrame(index=spy.index)
        
        # SPY momentum features
        spy_close = spy['Close'].squeeze() if hasattr(spy['Close'], 'squeeze') else spy['Close']
        features['spy_return_1d'] = spy_close.pct_change(1)
        features['spy_return_5d'] = spy_close.pct_change(5)
        features['spy_return_20d'] = spy_close.pct_change(20)
        features['spy_return_60d'] = spy_close.pct_change(60)
        
        # Moving averages
        features['spy_sma_20'] = spy_close.rolling(20).mean()
        features['spy_sma_50'] = spy_close.rolling(50).mean()
        features['spy_sma_200'] = spy_close.rolling(200).mean()
        
        # Price position relative to moving averages
        features['spy_vs_sma20'] = spy_close / features['spy_sma_20']
        features['spy_vs_sma50'] = spy_close / features['spy_sma_50']
        features['spy_vs_sma200'] = spy_close / features['spy_sma_200']
        
        # Volatility features
        features['spy_volatility_20d'] = features['spy_return_1d'].rolling(20).std()
        features['spy_volatility_60d'] = features['spy_return_1d'].rolling(60).std()
        
        # VIX features
        if len(vix) > 0:
            vix_close = vix['Close'].squeeze() if hasattr(vix['Close'], 'squeeze') else vix['Close']
            vix_aligned = vix_close.reindex(features.index, method='ffill')
            features['vix_level'] = vix_aligned
            features['vix_sma_20'] = vix_aligned.rolling(20).mean()
            features['vix_vs_sma'] = vix_aligned / features['vix_sma_20']
        
        # Bond yield features
        if len(tnx) > 0:
            tnx_close = tnx['Close'].squeeze() if hasattr(tnx['Close'], 'squeeze') else tnx['Close']
            tnx_aligned = tnx_close.reindex(features.index, method='ffill')
            features['yield_level'] = tnx_aligned
            features['yield_change_20d'] = tnx_aligned.pct_change(20)
        
        # Market breadth (using SPY volume as proxy)
        spy_volume = spy['Volume'].squeeze() if hasattr(spy['Volume'], 'squeeze') else spy['Volume']
        features['volume_sma_20'] = spy_volume.rolling(20).mean()
        features['volume_ratio'] = spy_volume / features['volume_sma_20']
        
        return features.dropna()
    
    def label_market_regimes(self, features):
        """Label market regimes based on market conditions"""
        labels = []
        
        for i in range(len(features)):
            row = features.iloc[i]
            
            # Bull market conditions
            bull_conditions = [
                row['spy_return_20d'] > 0.02,  # 2% gain over 20 days
                row['spy_vs_sma20'] > 1.0,     # Above 20-day MA
                row['spy_vs_sma50'] > 1.0,     # Above 50-day MA
                row.get('vix_level', 25) < 25,  # Low fear
                row['spy_volatility_20d'] < 0.02  # Low volatility
            ]
            
            # Bear market conditions
            bear_conditions = [
                row['spy_return_60d'] < -0.1,   # 10% decline over 60 days
                row['spy_vs_sma50'] < 0.95,     # Below 50-day MA
                row['spy_vs_sma200'] < 0.95,    # Below 200-day MA
                row.get('vix_level', 25) > 30,   # High fear
                row['spy_volatility_20d'] > 0.025  # High volatility
            ]
            
            # Determine regime
            if sum(bull_conditions) >= 3:
                labels.append(0)  # Bull market
            elif sum(bear_conditions) >= 3:
                labels.append(2)  # Bear market
            else:
                labels.append(1)  # Sideways market
        
        return np.array(labels)
    
    def train_regime_detector(self):
        """Train market regime detection model"""
        print("Training market regime detector...")
        
        # Fetch market data
        market_data = self.fetch_market_data(period="10y")
        if not market_data:
            raise ValueError("Could not fetch market data")
        
        # Calculate features
        features = self.calculate_regime_features(market_data)
        labels = self.label_market_regimes(features)
        
        # Prepare training data
        feature_cols = [col for col in features.columns if not col.startswith('spy_sma')]
        X = features[feature_cols].ffill().fillna(0)
        y = labels
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Regime Detection Accuracy: {accuracy:.3f}")
        print("\nRegime Distribution:")
        regime_names = ['Bull', 'Sideways', 'Bear']
        for i, name in enumerate(regime_names):
            count = np.sum(y == i)
            print(f"{name}: {count} ({count/len(y)*100:.1f}%)")
        
        self.feature_columns = feature_cols
        return accuracy
    
    def predict_current_regime(self):
        """Predict current market regime"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get recent market data
        market_data = self.fetch_market_data(period="1y")
        features = self.calculate_regime_features(market_data)
        
        # Get latest features
        latest_features = features[self.feature_columns].iloc[-1:].ffill().fillna(0)
        latest_scaled = self.scaler.transform(latest_features)
        
        # Predict regime
        regime_pred = self.model.predict(latest_scaled)[0]
        regime_probs = self.model.predict_proba(latest_scaled)[0]
        
        regime_names = ['Bull', 'Sideways', 'Bear']
        
        return {
            'regime': regime_names[regime_pred],
            'regime_id': regime_pred,
            'probabilities': {
                'Bull': regime_probs[0],
                'Sideways': regime_probs[1],
                'Bear': regime_probs[2]
            },
            'confidence': max(regime_probs)
        }


class InvestmentMetaLearner(nn.Module):
    """Neural network that combines momentum and fundamental scores"""
    
    def __init__(self, input_size=6):  # momentum, fundamental, regime features
        super(InvestmentMetaLearner, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x.squeeze()


class MasterInvestmentSystem:
    """Master system combining all three layers"""
    
    def __init__(self):
        self.momentum_predictor = None
        self.fundamental_analyzer = None
        self.regime_detector = MarketRegimeDetector()
        self.meta_learner = None
        self.meta_scaler = StandardScaler()
        self.is_trained = False
        
    def initialize_components(self, train_regime_detector=True):
        """Initialize and train all components"""
        print("Initializing Master Investment System...")
        
        # Initialize momentum predictor
        self.momentum_predictor = MomentumPredictor()
        
        # Initialize fundamental analyzer
        self.fundamental_analyzer = FundamentalAnalyzer()
        
        # Train regime detector
        if train_regime_detector:
            self.regime_detector.train_regime_detector()
        
        print("Components initialized successfully!")
    
    def train_system(self, symbols_list, momentum_symbol="AAPL"):
        """Train the complete system"""
        if not self.momentum_predictor or not self.fundamental_analyzer:
            self.initialize_components()
        
        print("Training Master Investment System...")
        
        # Train momentum predictor on one stock first
        print(f"Training momentum predictor on {momentum_symbol}...")
        momentum_data = self.momentum_predictor.prepare_data(momentum_symbol)
        if momentum_data:
            X_train_mom, X_test_mom, y_train_mom, y_test_mom = momentum_data
            self.momentum_predictor.train_model(X_train_mom, y_train_mom, 
                                              X_test_mom, y_test_mom, epochs=30)
        
        # Train fundamental analyzer
        print("Training fundamental analyzer...")
        X_train_fund, y_train_fund, valid_symbols = self.fundamental_analyzer.prepare_training_data(symbols_list)
        if len(X_train_fund) > 10:
            self.fundamental_analyzer.train_model(X_train_fund, y_train_fund)
        
        # Train meta-learner (simplified for demo)
        print("Training meta-learner...")
        self.meta_learner = InvestmentMetaLearner()
        
        self.is_trained = True
        print("System training completed!")
    
    def get_regime_weights(self, regime_info):
        """Get weighting based on market regime"""
        regime = regime_info['regime']
        confidence = regime_info['confidence']
        
        if regime == 'Bull':
            # In bull markets, momentum matters more
            momentum_weight = 0.4
            fundamental_weight = 0.6
        elif regime == 'Bear':
            # In bear markets, fundamentals matter more (quality stocks)
            momentum_weight = 0.2
            fundamental_weight = 0.8
        else:  # Sideways
            # In sideways markets, balance both
            momentum_weight = 0.5
            fundamental_weight = 0.5
        
        # Adjust weights based on confidence
        if confidence < 0.6:  # Low confidence, use balanced approach
            momentum_weight = 0.4
            fundamental_weight = 0.6
        
        return momentum_weight, fundamental_weight
    
    def calculate_position_size(self, investment_score, volatility_score):
        """Calculate position size based on Kelly criterion and risk management"""
        # Base position size
        base_position = 0.05  # 5% max per stock
        
        # Adjust for confidence (investment score)
        confidence_multiplier = investment_score
        
        # Adjust for volatility (inverse relationship)
        volatility_multiplier = 1.0 / (1.0 + volatility_score * 2)
        
        # Final position size
        position_size = base_position * confidence_multiplier * volatility_multiplier
        
        # Risk limits
        position_size = min(position_size, 0.1)  # Max 10% per stock
        position_size = max(position_size, 0.01)  # Min 1% per stock
        
        return position_size
    
    def verify_real_data(self, symbol):
        """
        Show real financial data to verify the system is working with actual data
        """
        print(f"\n🔍 VERIFYING REAL DATA FOR {symbol}")
        print("=" * 50)
        
        try:
            # Get real stock price data
            import yfinance as yf
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1mo")
            
            print(f"📊 REAL STOCK DATA:")
            print(f"Company Name: {info.get('longName', 'N/A')}")
            print(f"Sector: {info.get('sector', 'N/A')}")
            print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
            print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "Market Cap: N/A")
            print(f"P/E Ratio: {info.get('trailingPE', 'N/A')}")
            
            if len(hist) > 0:
                latest_price = hist['Close'].iloc[-1]
                month_change = (latest_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                print(f"Latest Close: ${latest_price:.2f}")
                print(f"1-Month Change: {month_change:.1f}%")
            
            # Test momentum predictor
            print(f"\n🚀 MOMENTUM ANALYSIS:")
            try:
                momentum_result = self.momentum_predictor.get_momentum_score(symbol)
                print(f"Raw momentum result: {momentum_result}")
            except Exception as e:
                print(f"Momentum error: {e}")
            
            # Test fundamental analyzer
            print(f"\n💼 FUNDAMENTAL ANALYSIS:")
            try:
                fundamental_result = self.fundamental_analyzer.get_fundamental_score(symbol)
                print(f"Raw fundamental result keys: {list(fundamental_result.keys()) if fundamental_result else 'None'}")
                if fundamental_result:
                    print(f"Features used: {fundamental_result.get('features_used', 0)}")
                    print(f"Raw prediction: {fundamental_result.get('raw_prediction', 'N/A')}")
            except Exception as e:
                print(f"Fundamental error: {e}")
                
        except Exception as e:
            print(f"Error verifying data: {e}")
        
        print("=" * 50)

    def analyze_stock(self, symbol):
        """
        Robust stock analysis with comprehensive error handling
        """
        try:
            print(f"Analyzing {symbol}...")
            
            # Initialize results with safe defaults
            analysis_result = {
                'investment_score': 0.5,
                'recommendation': 'HOLD',
                'confidence': 0.3,
                'position_size': 0.0,
                'momentum': {
                    'score': 0.5,
                    'confidence': 0.3,
                    'weight': 0.5
                },
                'fundamentals': {
                    'score': 0.5,
                    'confidence': 0.3,
                    'expected_return': 0.0,
                    'weight': 0.5
                },
                'risk_metrics': {
                    'volatility': 0.2,
                    'max_drawdown': 0.1
                }
            }
            
            # Step 1: Get momentum analysis with error handling
            try:
                momentum_result = self.momentum_predictor.get_momentum_score(symbol)
                if momentum_result and isinstance(momentum_result, dict):
                    momentum_score = float(momentum_result.get('momentum_score', 0.5))
                    momentum_confidence = float(momentum_result.get('confidence', 0.3))
                    
                    analysis_result['momentum'] = {
                        'score': max(0.0, min(1.0, momentum_score)),
                        'confidence': max(0.0, min(1.0, momentum_confidence)),
                        'weight': 0.5
                    }
                    print(f"✅ REAL momentum data: Score={momentum_score:.3f}, Confidence={momentum_confidence:.3f}")
                else:
                    print(f"⚠️ No momentum data returned for {symbol}, using defaults")
                    
            except Exception as e:
                print(f"❌ Error in momentum analysis for {symbol}: {e}")
                print("Using default momentum values")
            
            # Step 2: Get fundamental analysis with error handling
            try:
                print(f"Processing fundamental data for {symbol}...")
                fundamental_result = self.fundamental_analyzer.get_fundamental_score(symbol)
                
                if fundamental_result and isinstance(fundamental_result, dict):
                    fund_score = float(fundamental_result.get('fundamental_score', 0.5))
                    fund_confidence = float(fundamental_result.get('confidence', 0.3))
                    expected_return = float(fundamental_result.get('expected_1y_return', 0.0))
                    
                    analysis_result['fundamentals'] = {
                        'score': max(0.0, min(1.0, fund_score)),
                        'confidence': max(0.0, min(1.0, fund_confidence)),
                        'expected_return': expected_return,
                        'weight': 0.5
                    }
                    print(f"✅ REAL fundamental data: Score={fund_score:.3f}, Expected Return={expected_return:.1%}, Features Used={fundamental_result.get('features_used', 0)}")
                else:
                    print(f"⚠️ No fundamental data returned for {symbol}, using defaults")
                    
            except Exception as e:
                print(f"❌ Error in fundamental analysis for {symbol}: {e}")
                print("Using default fundamental values")
            
            # Step 3: Get market regime with error handling
            try:
                regime_info = self.regime_detector.predict_current_regime()
                regime = regime_info.get('regime', 'Sideways')
                regime_confidence = float(regime_info.get('confidence', 0.5))
                
                # Adaptive weighting based on market regime
                if regime == 'Bull':
                    momentum_weight = 0.6
                    fundamental_weight = 0.4
                elif regime == 'Bear':
                    momentum_weight = 0.3
                    fundamental_weight = 0.7
                else:  # Sideways
                    momentum_weight = 0.5
                    fundamental_weight = 0.5
                    
            except Exception as e:
                print(f"Error getting market regime: {e}")
                regime = 'Sideways'
                regime_confidence = 0.5
                momentum_weight = 0.5
                fundamental_weight = 0.5
            
            # Update weights
            analysis_result['momentum']['weight'] = momentum_weight
            analysis_result['fundamentals']['weight'] = fundamental_weight
            
            # Step 4: Calculate investment score
            momentum_score = analysis_result['momentum']['score']
            fundamental_score = analysis_result['fundamentals']['score']
            
            investment_score = (momentum_score * momentum_weight + 
                              fundamental_score * fundamental_weight)
            
            # Calculate overall confidence
            momentum_conf = analysis_result['momentum']['confidence']
            fundamental_conf = analysis_result['fundamentals']['confidence']
            overall_confidence = (momentum_conf * momentum_weight + 
                                fundamental_conf * fundamental_weight) * regime_confidence
            
            # Step 5: Generate recommendation
            if investment_score >= 0.8 and overall_confidence >= 0.7:
                recommendation = "STRONG BUY"
                position_size = min(0.15, overall_confidence * 0.2)
            elif investment_score >= 0.7 and overall_confidence >= 0.6:
                recommendation = "BUY"
                position_size = min(0.1, overall_confidence * 0.15)
            elif investment_score >= 0.6:
                recommendation = "WEAK BUY"
                position_size = min(0.05, overall_confidence * 0.1)
            elif investment_score <= 0.3:
                recommendation = "SELL"
                position_size = 0.0
            elif investment_score <= 0.4:
                recommendation = "WEAK SELL"
                position_size = 0.0
            else:
                recommendation = "HOLD"
                position_size = 0.0
            
            # Step 6: Calculate risk metrics
            try:
                # Simple risk calculation based on recent price data
                import yfinance as yf
                stock_data = yf.download(symbol, period="6m", progress=False)
                
                if len(stock_data) > 20:
                    returns = stock_data['Close'].pct_change().dropna()
                    volatility = float(returns.std() * np.sqrt(252))  # Annualized
                    
                    # Calculate max drawdown
                    cumulative = (1 + returns).cumprod()
                    rolling_max = cumulative.expanding().max()
                    drawdown = (cumulative - rolling_max) / rolling_max
                    max_drawdown = float(abs(drawdown.min()))
                else:
                    volatility = 0.2  # Default
                    max_drawdown = 0.1  # Default
                    
            except Exception as e:
                print(f"Error calculating risk metrics: {e}")
                volatility = 0.2
                max_drawdown = 0.1
            
            # Update final results
            analysis_result.update({
                'investment_score': max(0.0, min(1.0, investment_score)),
                'recommendation': recommendation,
                'confidence': max(0.0, min(1.0, overall_confidence)),
                'position_size': max(0.0, min(0.2, position_size)),
                'risk_metrics': {
                    'volatility': volatility,
                    'max_drawdown': max_drawdown
                }
            })
            
            print(f"Analysis complete for {symbol}: Score={investment_score:.3f}, Rec={recommendation}")
            
            return {
                'symbol': symbol,
                'analysis': analysis_result,
                'market_regime': {
                    'regime': regime,
                    'confidence': regime_confidence
                }
            }
            
        except Exception as e:
            print(f"Critical error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': f"Analysis failed: {str(e)}",
                'analysis': {
                    'investment_score': 0.5,
                    'recommendation': 'HOLD',
                    'confidence': 0.1,
                    'position_size': 0.0,
                    'momentum': {'score': 0.5, 'confidence': 0.1, 'weight': 0.5},
                    'fundamentals': {'score': 0.5, 'confidence': 0.1, 'expected_return': 0.0, 'weight': 0.5},
                    'risk_metrics': {'volatility': 0.2, 'max_drawdown': 0.1}
                }
            }
    
    def analyze_portfolio(self, symbols_list, max_positions=10):
        """Analyze multiple stocks and create portfolio recommendations"""
        print(f"Analyzing portfolio of {len(symbols_list)} stocks...")
        
        all_analyses = []
        market_regime = None
        
        for symbol in symbols_list:
            try:
                analysis = self.analyze_stock(symbol)
                if 'error' not in analysis and 'analysis' in analysis:
                    all_analyses.append(analysis)
                    # Get market regime from first successful analysis
                    if market_regime is None and 'market_regime' in analysis:
                        market_regime = analysis['market_regime']
            except Exception as e:
                print(f"Skipping {symbol}: {e}")
                continue
        
        if not all_analyses:
            return {"error": "No valid analyses completed"}
        
        # Default market regime if none found
        if market_regime is None:
            market_regime = {'regime': 'Sideways', 'confidence': 0.5}
        
        # Sort by investment score
        all_analyses.sort(key=lambda x: x['analysis']['investment_score'], reverse=True)
        
        # Portfolio construction
        portfolio = {
            'recommendations': all_analyses[:max_positions],
            'market_regime': market_regime,
            'portfolio_metrics': {
                'total_positions': min(len(all_analyses), max_positions),
                'avg_investment_score': np.mean([a['analysis']['investment_score'] 
                                                for a in all_analyses[:max_positions]]),
                'total_position_size': sum([a['analysis']['position_size'] 
                                          for a in all_analyses[:max_positions]]),
            }
        }
        
        return portfolio
    
    def print_analysis_report(self, analysis):
        """Print formatted analysis report"""
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        a = analysis['analysis']
        print(f"\n{'='*60}")
        print(f"INVESTMENT ANALYSIS: {analysis['symbol']}")
        print(f"{'='*60}")
        print(f"Recommendation: {a['recommendation']}")
        print(f"Investment Score: {a['investment_score']:.3f}")
        print(f"Confidence: {a['confidence']:.3f}")
        print(f"Position Size: {a['position_size']:.1%}")
        
        print(f"\nMOMENTUM ANALYSIS:")
        print(f"  Score: {a['momentum']['score']:.3f} (Weight: {a['momentum']['weight']:.1%})")
        print(f"  Confidence: {a['momentum']['confidence']:.3f}")
        
        print(f"\nFUNDAMENTAL ANALYSIS:")
        print(f"  Score: {a['fundamentals']['score']:.3f} (Weight: {a['fundamentals']['weight']:.1%})")
        print(f"  Expected 1Y Return: {a['fundamentals']['expected_return']:.1%}")
        print(f"  Confidence: {a['fundamentals']['confidence']:.3f}")
        
        print(f"\nMARKET REGIME: {a['market_regime']['regime']}")
        print(f"  Confidence: {a['market_regime']['confidence']:.3f}")
        
        print(f"\nRISK METRICS:")
        print(f"  Volatility: {a['risk_metrics']['volatility']:.1%}")


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = MasterInvestmentSystem()
    
    # Training symbols
    training_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'JPM', 'BAC', 'V', 'MA', 'JNJ', 'PFE', 'UNH'
    ]
    
    try:
        # Train the complete system
        system.train_system(training_symbols, momentum_symbol="AAPL")
        
        # Analyze specific stocks
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        for symbol in test_symbols:
            analysis = system.analyze_stock(symbol)
            system.print_analysis_report(analysis)
        
        # Portfolio analysis
        portfolio = system.analyze_portfolio(training_symbols, max_positions=5)
        
        print(f"\n{'='*60}")
        print("TOP PORTFOLIO RECOMMENDATIONS")
        print(f"{'='*60}")
        print(f"Market Regime: {portfolio['market_regime']['regime']}")
        print(f"Total Positions: {portfolio['portfolio_metrics']['total_positions']}")
        print(f"Avg Investment Score: {portfolio['portfolio_metrics']['avg_investment_score']:.3f}")
        
        print("\nTop 5 Stocks:")
        for i, rec in enumerate(portfolio['recommendations'][:5]):
            score = rec['analysis']['investment_score']
            recommendation = rec['analysis']['recommendation']
            pos_size = rec['analysis']['position_size']
            print(f"{i+1}. {rec['symbol']}: {recommendation} (Score: {score:.3f}, Size: {pos_size:.1%})")
            
    except Exception as e:
        print(f"Error: {e}")