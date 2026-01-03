"""
Layer 2: Fundamental Analysis for Long-term Stock Investment
Focuses on financial health, valuation metrics, and stability signals
Uses XGBoost for fundamental strength scoring
"""

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class FundamentalDataProcessor:
    """Handles fundamental data fetching and feature engineering"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # Better for financial data with outliers
        self.feature_columns = []
        
    def fetch_fundamental_data(self, symbol, period="5y"):
        """Fetch comprehensive fundamental data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get different types of financial data
            info = ticker.info
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            quarterly_financials = ticker.quarterly_financials
            quarterly_balance_sheet = ticker.quarterly_balance_sheet
            quarterly_cashflow = ticker.quarterly_cashflow
            
            return {
                'info': info,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow,
                'quarterly_financials': quarterly_financials,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'quarterly_cashflow': quarterly_cashflow
            }
        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            return None
    
    def calculate_profitability_metrics(self, financials, balance_sheet):
        """Calculate profitability and efficiency ratios"""
        metrics = {}
        
        try:
            # Revenue and earnings growth
            if 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue'].dropna()
                if len(revenue) >= 2:
                    metrics['revenue_growth_1y'] = (revenue.iloc[0] - revenue.iloc[1]) / revenue.iloc[1]
                if len(revenue) >= 4:
                    metrics['revenue_growth_3y_avg'] = ((revenue.iloc[0] / revenue.iloc[3]) ** (1/3)) - 1
                    metrics['revenue_volatility'] = revenue.pct_change().std()
            
            # Profit margins
            if 'Total Revenue' in financials.index and 'Net Income' in financials.index:
                revenue = financials.loc['Total Revenue'].dropna()
                net_income = financials.loc['Net Income'].dropna()
                
                if len(revenue) > 0 and len(net_income) > 0:
                    # Align the series
                    common_dates = revenue.index.intersection(net_income.index)
                    if len(common_dates) > 0:
                        metrics['net_profit_margin'] = (net_income[common_dates[0]] / 
                                                       revenue[common_dates[0]])
                        
                        # Margin stability
                        if len(common_dates) >= 3:
                            margins = net_income[common_dates] / revenue[common_dates]
                            metrics['margin_stability'] = 1 / (1 + margins.std())
            
            # Return metrics
            if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index:
                net_income = financials.loc['Net Income'].dropna()
                equity = balance_sheet.loc['Total Stockholder Equity'].dropna()
                
                common_dates = net_income.index.intersection(equity.index)
                if len(common_dates) > 0:
                    metrics['roe'] = net_income[common_dates[0]] / equity[common_dates[0]]
            
            # Asset turnover
            if 'Total Revenue' in financials.index and 'Total Assets' in balance_sheet.index:
                revenue = financials.loc['Total Revenue'].dropna()
                assets = balance_sheet.loc['Total Assets'].dropna()
                
                common_dates = revenue.index.intersection(assets.index)
                if len(common_dates) > 0:
                    metrics['asset_turnover'] = revenue[common_dates[0]] / assets[common_dates[0]]
                    
        except Exception as e:
            print(f"Error calculating profitability metrics: {e}")
        
        return metrics
    
    def calculate_financial_health_metrics(self, balance_sheet, cashflow):
        """Calculate debt, liquidity, and cash flow metrics"""
        metrics = {}
        
        try:
            # Debt ratios
            if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                debt = balance_sheet.loc['Total Debt'].dropna()
                equity = balance_sheet.loc['Total Stockholder Equity'].dropna()
                
                if len(debt) > 0 and len(equity) > 0:
                    metrics['debt_to_equity'] = debt.iloc[0] / equity.iloc[0]
            
            # Current ratio
            if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Current Assets'].dropna()
                current_liabilities = balance_sheet.loc['Current Liabilities'].dropna()
                
                if len(current_assets) > 0 and len(current_liabilities) > 0:
                    metrics['current_ratio'] = current_assets.iloc[0] / current_liabilities.iloc[0]
            
            # Cash metrics
            if 'Cash And Cash Equivalents' in balance_sheet.index and 'Total Assets' in balance_sheet.index:
                cash = balance_sheet.loc['Cash And Cash Equivalents'].dropna()
                total_assets = balance_sheet.loc['Total Assets'].dropna()
                
                if len(cash) > 0 and len(total_assets) > 0:
                    metrics['cash_ratio'] = cash.iloc[0] / total_assets.iloc[0]
            
            # Free cash flow
            if 'Free Cash Flow' in cashflow.index:
                fcf = cashflow.loc['Free Cash Flow'].dropna()
                if len(fcf) >= 2:
                    metrics['fcf_growth'] = (fcf.iloc[0] - fcf.iloc[1]) / abs(fcf.iloc[1])
                    metrics['fcf_stability'] = 1 / (1 + fcf.pct_change().std())
            
            # Working capital efficiency
            if ('Current Assets' in balance_sheet.index and 
                'Current Liabilities' in balance_sheet.index and 
                'Total Revenue' in balance_sheet.index):
                
                current_assets = balance_sheet.loc['Current Assets'].dropna()
                current_liab = balance_sheet.loc['Current Liabilities'].dropna()
                
                if len(current_assets) > 0 and len(current_liab) > 0:
                    working_capital = current_assets.iloc[0] - current_liab.iloc[0]
                    if 'Total Revenue' in balance_sheet.index:
                        revenue = balance_sheet.loc['Total Revenue'].dropna()
                        if len(revenue) > 0:
                            metrics['working_capital_to_revenue'] = working_capital / revenue.iloc[0]
                            
        except Exception as e:
            print(f"Error calculating financial health metrics: {e}")
        
        return metrics
    
    def calculate_valuation_metrics(self, info, financials, balance_sheet):
        """Calculate valuation ratios and metrics"""
        metrics = {}
        
        try:
            # Price ratios from info
            if 'forwardPE' in info:
                metrics['forward_pe'] = info['forwardPE']
            if 'trailingPE' in info:
                metrics['trailing_pe'] = info['trailingPE']
            if 'priceToBook' in info:
                metrics['price_to_book'] = info['priceToBook']
            if 'priceToSalesTrailing12Months' in info:
                metrics['price_to_sales'] = info['priceToSalesTrailing12Months']
            if 'pegRatio' in info:
                metrics['peg_ratio'] = info['pegRatio']
            
            # Market cap metrics
            if 'marketCap' in info:
                market_cap = info['marketCap']
                
                # Market cap to revenue
                if 'Total Revenue' in financials.index:
                    revenue = financials.loc['Total Revenue'].dropna()
                    if len(revenue) > 0:
                        metrics['market_cap_to_revenue'] = market_cap / revenue.iloc[0]
                
                # Market cap to free cash flow
                if 'enterpriseValue' in info and 'freeCashflow' in info:
                    fcf_value = info['freeCashflow']
                    
                    # Handle Series or scalar values safely
                    if fcf_value is not None:
                        if hasattr(fcf_value, 'iloc'):
                            fcf_value = fcf_value.iloc[0] if len(fcf_value) > 0 else 0
                        
                        # Convert to float and check validity
                        try:
                            fcf_float = float(fcf_value) if fcf_value != '' else 0
                            if fcf_float > 0:
                                ev_value = info['enterpriseValue']
                                if hasattr(ev_value, 'iloc'):
                                    ev_value = ev_value.iloc[0] if len(ev_value) > 0 else 0
                                if float(ev_value) > 0:
                                    metrics['ev_to_fcf'] = float(ev_value) / fcf_float
                        except (ValueError, TypeError):
                            pass
            
            # Book value metrics
            if ('Total Stockholder Equity' in balance_sheet.index and 
                'sharesOutstanding' in info):
                
                equity = balance_sheet.loc['Total Stockholder Equity'].dropna()
                shares_value = info['sharesOutstanding']
                if len(equity) > 0 and shares_value is not None:
                    if hasattr(shares_value, 'iloc'):
                        shares_value = shares_value.iloc[0] if len(shares_value) > 0 else 0
                    
                    try:
                        shares_float = float(shares_value) if shares_value != '' else 0
                        if shares_float > 0:
                            book_value_per_share = equity.iloc[0] / shares_float
                            if 'currentPrice' in info:
                                current_price = info['currentPrice']
                                if hasattr(current_price, 'iloc'):
                                    current_price = current_price.iloc[0] if len(current_price) > 0 else 0
                                if float(current_price) > 0:
                                    metrics['price_to_book_calculated'] = float(current_price) / book_value_per_share
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            print(f"Error calculating valuation metrics: {e}")
        
        return metrics
    
    def calculate_growth_metrics(self, financials, info):
        """Calculate growth rates and trends"""
        metrics = {}
        
        try:
            # Revenue growth trends
            if 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue'].dropna().sort_index()
                if len(revenue) >= 3:
                    # Calculate growth rates
                    growth_rates = revenue.pct_change().dropna()
                    metrics['revenue_growth_avg'] = growth_rates.mean()
                    metrics['revenue_growth_trend'] = np.polyfit(range(len(growth_rates)), 
                                                               growth_rates.values, 1)[0]
            
            # Earnings growth
            if 'Net Income' in financials.index:
                earnings = financials.loc['Net Income'].dropna().sort_index()
                if len(earnings) >= 3:
                    growth_rates = earnings.pct_change().dropna()
                    metrics['earnings_growth_avg'] = growth_rates.mean()
                    metrics['earnings_growth_volatility'] = growth_rates.std()
            
            # Future growth expectations
            if 'earningsGrowth' in info:
                metrics['expected_earnings_growth'] = info['earningsGrowth']
            if 'revenueGrowth' in info:
                metrics['expected_revenue_growth'] = info['revenueGrowth']
                
        except Exception as e:
            print(f"Error calculating growth metrics: {e}")
        
        return metrics
    
    def calculate_management_quality_metrics(self, info, financials, cashflow):
        """Calculate metrics indicating management quality"""
        metrics = {}
        
        try:
            # Capital allocation efficiency
            if ('Capital Expenditures' in cashflow.index and 
                'Total Revenue' in financials.index):
                
                capex = cashflow.loc['Capital Expenditures'].dropna()
                revenue = financials.loc['Total Revenue'].dropna()
                
                if len(capex) >= 2 and len(revenue) >= 2:
                    # CapEx efficiency
                    common_dates = capex.index.intersection(revenue.index)
                    if len(common_dates) >= 2:
                        capex_ratio = abs(capex[common_dates]) / revenue[common_dates]
                        metrics['capex_to_revenue'] = capex_ratio.iloc[0]
                        metrics['capex_consistency'] = 1 / (1 + capex_ratio.std())
            
            # Dividend metrics (capital return policy)
            if 'dividendRate' in info and 'payoutRatio' in info:
                div_rate = info['dividendRate']
                payout_ratio = info['payoutRatio']
                
                # Handle potential Series values
                if hasattr(div_rate, 'iloc'):
                    div_rate = div_rate.iloc[0] if len(div_rate) > 0 else 0
                if hasattr(payout_ratio, 'iloc'):
                    payout_ratio = payout_ratio.iloc[0] if len(payout_ratio) > 0 else 0
                    
                try:
                    div_float = float(div_rate) if div_rate not in ['', None] else 0
                    payout_float = float(payout_ratio) if payout_ratio not in ['', None] else 0
                    
                    if div_float > 0 and payout_float > 0:
                        current_price = info.get('currentPrice', 0)
                        if hasattr(current_price, 'iloc'):
                            current_price = current_price.iloc[0] if len(current_price) > 0 else 0
                        
                        if float(current_price) > 0:
                            metrics['dividend_yield'] = div_float / float(current_price)
                            metrics['payout_ratio'] = payout_float
                            metrics['dividend_sustainability'] = 1.0 if payout_float < 0.6 else 0.5
                except (ValueError, TypeError):
                    pass
            
            # Share count changes (buyback activity)
            if 'sharesOutstanding' in info and 'impliedSharesOutstanding' in info:
                shares_out = info['sharesOutstanding']
                implied_shares = info['impliedSharesOutstanding']
                
                # Handle potential Series values
                if hasattr(shares_out, 'iloc'):
                    shares_out = shares_out.iloc[0] if len(shares_out) > 0 else 0
                if hasattr(implied_shares, 'iloc'):
                    implied_shares = implied_shares.iloc[0] if len(implied_shares) > 0 else 0
                    
                try:
                    shares_float = float(shares_out) if shares_out not in ['', None] else 0
                    implied_float = float(implied_shares) if implied_shares not in ['', None] else 0
                    
                    if shares_float > 0 and implied_float > 0:
                        shares_change = (shares_float - implied_float) / implied_float
                        metrics['shares_change'] = shares_change  # Negative is good (buybacks)
                except (ValueError, TypeError):
                    pass
                    
        except Exception as e:
            print(f"Error calculating management quality metrics: {e}")
        
        return metrics
    
    def create_fundamental_features(self, symbol):
        """Create comprehensive fundamental feature set"""
        print(f"Processing fundamental data for {symbol}...")
        
        # Fetch data
        data = self.fetch_fundamental_data(symbol)
        if not data:
            return None
        
        # Calculate different metric categories
        profitability = self.calculate_profitability_metrics(
            data['financials'], data['balance_sheet']
        )
        
        financial_health = self.calculate_financial_health_metrics(
            data['balance_sheet'], data['cashflow']
        )
        
        valuation = self.calculate_valuation_metrics(
            data['info'], data['financials'], data['balance_sheet']
        )
        
        growth = self.calculate_growth_metrics(data['financials'], data['info'])
        
        management = self.calculate_management_quality_metrics(
            data['info'], data['financials'], data['cashflow']
        )
        
        # Combine all features
        all_features = {**profitability, **financial_health, **valuation, 
                       **growth, **management}
        
        # Add sector and industry features
        if 'sector' in data['info']:
            all_features['sector'] = data['info']['sector']
        if 'industry' in data['info']:
            all_features['industry'] = data['info']['industry']
        
        return all_features


class FundamentalAnalyzer:
    """XGBoost-based fundamental analysis system"""
    
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.data_processor = FundamentalDataProcessor()
        
    def prepare_training_data(self, symbols, target_period_days=252):
        """Prepare training data from multiple stocks"""
        print(f"Preparing training data for {len(symbols)} stocks...")
        
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            try:
                # Get fundamental features
                features = self.data_processor.create_fundamental_features(symbol)
                if not features:
                    continue
                
                # Get price data for target calculation
                stock_data = yf.download(symbol, period="2y", progress=False)
                if len(stock_data) < target_period_days:
                    continue
                
                # Calculate future return as target (1-year forward return)
                current_price = stock_data['Close'].iloc[-target_period_days]
                future_price = stock_data['Close'].iloc[-1]
                target_return = (future_price - current_price) / current_price
                
                # Prepare features
                features_numeric = {}
                for key, value in features.items():
                    if key in ['sector', 'industry']:
                        continue  # Skip categorical for now
                    
                    # Handle potential Series values
                    if hasattr(value, 'iloc') and len(value) > 0:
                        value = value.iloc[0]
                    elif hasattr(value, 'item'):
                        value = value.item()
                    
                    if isinstance(value, (int, float, np.number)) and not (np.isnan(value) if isinstance(value, (float, np.number)) else False):
                        features_numeric[key] = float(value)
                
                if len(features_numeric) > 10:  # Minimum features threshold
                    all_features.append(features_numeric)
                    all_targets.append(target_return)
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features).fillna(0)
        self.feature_columns = features_df.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features_df)
        y = np.array(all_targets)
        
        print(f"Training data prepared: {len(X_scaled)} samples, {len(self.feature_columns)} features")
        return X_scaled, y, symbols[:len(all_features)]
    
    def train_model(self, X_train, y_train, validation_split=0.2):
        """Train XGBoost model for fundamental analysis"""
        print("Training XGBoost fundamental analysis model...")
        
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params, eval_metric='rmse', early_stopping_rounds=20)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False
        )
        
        # Calculate performance metrics
        train_pred = self.model.predict(X_tr)
        val_pred = self.model.predict(X_val)
        
        train_r2 = r2_score(y_tr, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        print(f"Training R²: {train_r2:.4f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def get_fundamental_score(self, symbol):
        """Get fundamental strength score for a stock (0-1 scale)"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get features for the stock
        features = self.data_processor.create_fundamental_features(symbol)
        if not features:
            raise ValueError(f"Could not get fundamental data for {symbol}")
        
        # Prepare features
        features_numeric = {}
        for key, value in features.items():
            if key in self.feature_columns:
                if isinstance(value, (int, float)) and not np.isnan(value):
                    features_numeric[key] = value
                else:
                    features_numeric[key] = 0  # Fill missing with 0
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features_numeric:
                features_numeric[col] = 0
        
        # Convert to array and scale
        features_array = np.array([features_numeric[col] for col in self.feature_columns]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Predict
        raw_prediction = self.model.predict(features_scaled)[0]
        
        # Convert to 0-1 score (expected return to probability-like score)
        # Higher expected returns get higher scores
        fundamental_score = 1 / (1 + np.exp(-raw_prediction * 5))  # Sigmoid transformation
        
        # Calculate confidence based on feature completeness
        feature_completeness = sum(1 for v in features_numeric.values() if v != 0) / len(self.feature_columns)
        
        return {
            'fundamental_score': fundamental_score,
            'raw_prediction': raw_prediction,
            'expected_1y_return': raw_prediction,
            'confidence': feature_completeness,
            'features_used': len([v for v in features_numeric.values() if v != 0])
        }


# Example usage and testing
if __name__ == "__main__":
    # Popular stocks for training
    training_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'AMD', 'CRM', 'ADBE', 'PYPL', 'INTC', 'ORCL', 'CSCO', 'IBM',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'CVS', 'LLY', 'TMO'
    ]
    
    try:
        # Initialize analyzer
        analyzer = FundamentalAnalyzer()
        
        # Prepare training data
        X_train, y_train, valid_symbols = analyzer.prepare_training_data(training_symbols)
        
        if len(X_train) > 10:  # Minimum samples for training
            # Train model
            feature_importance = analyzer.train_model(X_train, y_train)
            
            # Test on a specific stock
            test_symbol = "AAPL"
            result = analyzer.get_fundamental_score(test_symbol)
            
            print(f"\n{test_symbol} Fundamental Analysis:")
            print(f"Fundamental Score: {result['fundamental_score']:.3f}")
            print(f"Expected 1Y Return: {result['expected_1y_return']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Features Used: {result['features_used']}")
            
        else:
            print("Not enough training data collected")
            
    except Exception as e:
        print(f"Error: {e}")