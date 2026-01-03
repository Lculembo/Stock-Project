"""
Streamlit UI for Master Investment System
Combines momentum, fundamental, and meta-learning analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time

# Import our analysis systems
try:
    from investment_master_system import MasterInvestmentSystem
    from momentum_predictor import MomentumPredictor
    from fundamental_analyzer import FundamentalAnalyzer
except ImportError as e:
    st.error(f"Could not import analysis modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Investment Advisor", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .recommendation-strong-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-buy {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'training_symbols' not in st.session_state:
    st.session_state.training_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'JPM', 'BAC', 'V', 'MA', 'JNJ', 'PFE', 'UNH', 'AMD', 'CRM'
    ]

def initialize_system():
    """Initialize the master investment system"""
    if st.session_state.system is None:
        st.session_state.system = MasterInvestmentSystem()
    return st.session_state.system

def get_recommendation_style(recommendation):
    """Get CSS class for recommendation styling"""
    if recommendation == "STRONG BUY":
        return "recommendation-strong-buy"
    elif recommendation == "BUY":
        return "recommendation-buy"
    elif recommendation in ["HOLD", "WEAK SELL"]:
        return "recommendation-hold"
    else:
        return "recommendation-sell"

def create_score_gauge(score, title):
    """Create a gauge chart for scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.8
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_price_chart(symbol, days=90):
    """Create price chart with technical indicators"""
    try:
        # Fetch price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if len(data) == 0:
            return None
        
        # Calculate moving averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Price Chart', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='green', dash='dash')),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            title_text=f"{symbol} Technical Analysis",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating price chart: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">AI Investment Advisor</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Multi-Layer Stock Analysis System**")
    
    # Sidebar
    st.sidebar.title("System Controls")
    
    # Initialize system
    system = initialize_system()
    
    # Training section
    with st.sidebar.expander("Model Training", expanded=not st.session_state.trained):
        st.write("**Training Symbols:**")
        
        # Allow editing of training symbols
        symbols_text = st.text_area(
            "Symbols (one per line):",
            value='\n'.join(st.session_state.training_symbols),
            height=150
        )
        
        if st.button("Update Training Symbols"):
            st.session_state.training_symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]
            st.success(f"Updated to {len(st.session_state.training_symbols)} symbols")
        
        if st.button("Train System", type="primary"):
            with st.spinner("Training AI models... This may take several minutes..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize components
                    status_text.text("Initializing components...")
                    progress_bar.progress(20)
                    system.initialize_components()
                    
                    # Train system
                    status_text.text("Training models...")
                    progress_bar.progress(60)
                    system.train_system(st.session_state.training_symbols)
                    
                    progress_bar.progress(100)
                    status_text.text("Training completed!")
                    
                    st.session_state.trained = True
                    st.success("System trained successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
    
    # Market regime display
    if st.session_state.trained:
        try:
            regime_info = system.regime_detector.predict_current_regime()
            regime = regime_info['regime']
            confidence = regime_info['confidence']
            
            # Color coding for regime
            regime_colors = {
                'Bull': 'Green',
                'Bear': 'Red', 
                'Sideways': 'Yellow'
            }
            
            st.sidebar.markdown("### Market Regime")
            st.sidebar.markdown(f"**{regime_colors[regime]} {regime} Market**")
            st.sidebar.progress(confidence)
            st.sidebar.caption(f"Confidence: {confidence:.1%}")
            
        except Exception as e:
            st.sidebar.error(f"Could not get market regime: {e}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single Stock Analysis", "Portfolio Analysis", "Comparison Tool", "System Info"])
    
    with tab1:
        st.header("Single Stock Analysis")
        
        if not st.session_state.trained:
            st.warning("⚠️ Please train the system first using the sidebar.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Stock input
            symbol = st.text_input("Enter Stock Symbol:", value="AAPL", key="single_stock").upper()
            
            if st.button("Analyze Stock", type="primary"):
                if symbol:
                    with st.spinner(f"Analyzing {symbol}..."):
                        try:
                            analysis = system.analyze_stock(symbol)
                            
                            if 'error' in analysis:
                                st.error(f"Error: {analysis['error']}")
                            else:
                                st.session_state.latest_analysis = analysis
                                
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
        
        with col2:
            if 'latest_analysis' in st.session_state and st.session_state.latest_analysis['symbol'] == symbol:
                analysis = st.session_state.latest_analysis['analysis']
                
                # Main metrics
                col2_1, col2_2, col2_3 = st.columns(3)
                
                with col2_1:
                    st.metric(
                        "Investment Score", 
                        f"{analysis['investment_score']:.3f}",
                        delta=f"Confidence: {analysis['confidence']:.3f}"
                    )
                
                with col2_2:
                    recommendation = analysis['recommendation']
                    st.markdown(f'<div class="{get_recommendation_style(recommendation)}">{recommendation}</div>', 
                              unsafe_allow_html=True)
                
                with col2_3:
                    st.metric(
                        "Position Size",
                        f"{analysis['position_size']:.1%}",
                        delta=f"Risk: {analysis['risk_metrics']['volatility']:.1%}"
                    )
        
        # Detailed analysis display
        if 'latest_analysis' in st.session_state and st.session_state.latest_analysis['symbol'] == symbol:
            analysis = st.session_state.latest_analysis['analysis']
            
            # Score gauges
            st.subheader("Detailed Scores")
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            
            with gauge_col1:
                fig1 = create_score_gauge(analysis['investment_score'], "Overall Score")
                st.plotly_chart(fig1, use_container_width=True)
            
            with gauge_col2:
                fig2 = create_score_gauge(analysis['momentum']['score'], "Momentum Score")
                st.plotly_chart(fig2, use_container_width=True)
            
            with gauge_col3:
                fig3 = create_score_gauge(analysis['fundamentals']['score'], "Fundamental Score")
                st.plotly_chart(fig3, use_container_width=True)
            
            # Price chart
            st.subheader("Technical Analysis")
            price_chart = create_price_chart(symbol)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            
            # Detailed breakdown
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Momentum Analysis")
                st.metric("Score", f"{analysis['momentum']['score']:.3f}")
                st.metric("Weight in Decision", f"{analysis['momentum']['weight']:.1%}")
                st.metric("Confidence", f"{analysis['momentum']['confidence']:.3f}")
            
            with col_right:
                st.subheader("Fundamental Analysis")
                st.metric("Score", f"{analysis['fundamentals']['score']:.3f}")
                st.metric("Expected 1Y Return", f"{analysis['fundamentals']['expected_return']:.1%}")
                st.metric("Weight in Decision", f"{analysis['fundamentals']['weight']:.1%}")
    
    with tab2:
        st.header("Portfolio Analysis")
        
        if not st.session_state.trained:
            st.warning("Please train the system first using the sidebar.")
            return
        
        # Portfolio input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            portfolio_symbols = st.text_area(
                "Enter stock symbols (one per line):",
                value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nMETA\nNVDA",
                height=150
            )
        
        with col2:
            max_positions = st.slider("Max Positions", 3, 20, 10)
            
            if st.button("Analyze Portfolio", type="primary"):
                symbols = [s.strip().upper() for s in portfolio_symbols.split('\n') if s.strip()]
                
                if symbols:
                    with st.spinner("Analyzing portfolio..."):
                        try:
                            portfolio = system.analyze_portfolio(symbols, max_positions)
                            
                            if 'error' in portfolio:
                                st.error(f"Error: {portfolio['error']}")
                            else:
                                st.session_state.portfolio_analysis = portfolio
                                
                        except Exception as e:
                            st.error(f"Portfolio analysis failed: {e}")
        
        # Display portfolio results
        if 'portfolio_analysis' in st.session_state:
            portfolio = st.session_state.portfolio_analysis
            
            # Portfolio metrics
            st.subheader("Portfolio Overview")
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            
            with met_col1:
                st.metric("Total Positions", portfolio['portfolio_metrics']['total_positions'])
            with met_col2:
                st.metric("Avg Score", f"{portfolio['portfolio_metrics']['avg_investment_score']:.3f}")
            with met_col3:
                st.metric("Total Allocation", f"{portfolio['portfolio_metrics']['total_position_size']:.1%}")
            with met_col4:
                regime = portfolio['market_regime']['regime']
                st.metric("Market Regime", regime)
            
            # Top recommendations table
            st.subheader("Top Recommendations")
            
            portfolio_data = []
            for rec in portfolio['recommendations']:
                analysis = rec['analysis']
                portfolio_data.append({
                    'Symbol': rec['symbol'],
                    'Recommendation': analysis['recommendation'],
                    'Investment Score': f"{analysis['investment_score']:.3f}",
                    'Position Size': f"{analysis['position_size']:.1%}",
                    'Momentum': f"{analysis['momentum']['score']:.3f}",
                    'Fundamentals': f"{analysis['fundamentals']['score']:.3f}",
                    'Expected Return': f"{analysis['fundamentals']['expected_return']:.1%}"
                })
            
            df = pd.DataFrame(portfolio_data)
            st.dataframe(df, use_container_width=True)
            
            # Portfolio allocation chart
            st.subheader("Portfolio Allocation")
            
            symbols = [rec['symbol'] for rec in portfolio['recommendations']]
            sizes = [rec['analysis']['position_size'] for rec in portfolio['recommendations']]
            
            fig = px.pie(values=sizes, names=symbols, title="Recommended Position Sizes")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Stock Comparison Tool")
        
        if not st.session_state.trained:
            st.warning("Please train the system first using the sidebar.")
            return
        
        # Stock comparison input
        compare_symbols = st.text_input("Enter symbols to compare (comma-separated):", value="AAPL,MSFT,GOOGL")
        
        if st.button("Compare Stocks", type="primary"):
            symbols = [s.strip().upper() for s in compare_symbols.split(',') if s.strip()]
            
            if len(symbols) >= 2:
                comparison_data = []
                
                with st.spinner("Comparing stocks..."):
                    for symbol in symbols:
                        try:
                            analysis = system.analyze_stock(symbol)
                            if 'error' not in analysis:
                                a = analysis['analysis']
                                comparison_data.append({
                                    'Symbol': symbol,
                                    'Investment Score': a['investment_score'],
                                    'Recommendation': a['recommendation'],
                                    'Momentum Score': a['momentum']['score'],
                                    'Fundamental Score': a['fundamentals']['score'],
                                    'Position Size': a['position_size'],
                                    'Expected Return': a['fundamentals']['expected_return'],
                                    'Volatility': a['risk_metrics']['volatility']
                                })
                        except Exception as e:
                            st.warning(f"Could not analyze {symbol}: {e}")
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Display comparison table
                    st.subheader("Comparison Table")
                    st.dataframe(df_comparison, use_container_width=True)
                    
                    # Comparison charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.bar(df_comparison, x='Symbol', y='Investment Score', 
                                     title="Investment Scores Comparison")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = px.scatter(df_comparison, x='Momentum Score', y='Fundamental Score', 
                                        text='Symbol', title="Momentum vs Fundamentals")
                        fig2.update_traces(textposition="top center")
                        st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.header("System Information")
        
        st.subheader("About the AI Investment Advisor")
        
        st.markdown("""
        This advanced investment system combines three layers of analysis:
        
        **Layer 1: Momentum Analysis**
        - LSTM neural network with attention mechanism
        - Analyzes price patterns, volume, and technical indicators
        - Emphasizes recent market behavior with exponential weighting
        
        **Layer 2: Fundamental Analysis**
        - XGBoost model analyzing financial health and valuation
        - Profitability, growth, debt ratios, and management quality
        - Long-term value assessment with 5+ years of data
        
        **Layer 3: Meta-Learning & Regime Detection**
        - Market regime classification (Bull/Bear/Sideways)
        - Adaptive weighting based on market conditions
        - Risk management and position sizing
        """)
        
        if st.session_state.trained:
            st.subheader("System Status")
            st.success("System is trained and ready for analysis!")
            
            st.subheader("Training Information")
            st.write(f"**Training Symbols:** {len(st.session_state.training_symbols)}")
            st.write(f"**Symbols:** {', '.join(st.session_state.training_symbols[:10])}{'...' if len(st.session_state.training_symbols) > 10 else ''}")
            
            # Market regime info
            try:
                regime_info = system.regime_detector.predict_current_regime()
                st.subheader("Current Market Analysis")
                
                probs = regime_info['probabilities']
                regime_df = pd.DataFrame({
                    'Regime': ['Bull', 'Sideways', 'Bear'],
                    'Probability': [probs['Bull'], probs['Sideways'], probs['Bear']]
                })
                
                fig = px.bar(regime_df, x='Regime', y='Probability', 
                           title="Market Regime Probabilities")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not get market regime info: {e}")
        
        else:
            st.subheader("System Status")
            st.warning("System not trained yet. Use the sidebar to train the models.")
        
        st.subheader("Usage Tips")
        st.markdown("""
        1. **Train First:** Always train the system with diverse stocks
        2. **Market Context:** Pay attention to the detected market regime
        3. **Position Sizing:** Follow recommended position sizes for risk management
        4. **Diversification:** Use portfolio analysis for balanced exposure
        5. **Confidence Scores:** Higher confidence = more reliable predictions
        """)

if __name__ == "__main__":
    main()