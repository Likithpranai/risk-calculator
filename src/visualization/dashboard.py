import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.data.data_loader import DataLoader
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator
from src.models.risk_predictor import RiskPredictor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("AI-Driven Trade Risk Assessment System", style={'textAlign': 'center'}),
        html.Hr(),
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='Portfolio Overview', value='tab-1'),
            dcc.Tab(label='Trade-Level Risk', value='tab-2'),
            dcc.Tab(label='Stress Testing', value='tab-3'),
            dcc.Tab(label='Sentiment Impact', value='tab-4'),
            dcc.Tab(label='ML Predictions', value='tab-5'),
        ]),
        html.Div(id='tabs-content')
    ]),
])

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Portfolio Risk Overview'),
            html.Div([
                html.Div([
                    html.Label('Select Date Range:'),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date_placeholder_text='Start Date',
                        end_date_placeholder_text='End Date',
                        calendar_orientation='horizontal',
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Button('Update', id='update-button', n_clicks=0),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='portfolio-var-graph'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='portfolio-es-graph'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                html.Div([
                    html.H4('Portfolio Statistics'),
                    html.Table(id='portfolio-stats-table'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='asset-allocation-pie'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Trade-Level Risk Assessment'),
            html.Div([
                html.Div([
                    html.Label('Select Symbol:'),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[],
                        value=None,
                        placeholder='Select a symbol',
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Risk Metric:'),
                    dcc.Dropdown(
                        id='risk-metric-dropdown',
                        options=[
                            {'label': 'Value at Risk (VaR)', 'value': 'VaR_95'},
                            {'label': 'Expected Shortfall (ES)', 'value': 'ES_95'},
                            {'label': 'Loss Probability (5%)', 'value': 'Loss_Prob_5pct'},
                            {'label': 'Loss Probability (10%)', 'value': 'Loss_Prob_10pct'},
                        ],
                        value='VaR_95',
                    ),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                dcc.Graph(id='trade-risk-graph'),
            ]),
            
            html.Div([
                html.H4('Trade Risk Details'),
                html.Table(id='trade-risk-table'),
            ]),
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Stress Testing & Monte Carlo Simulation'),
            html.Div([
                html.Div([
                    html.Label('Scenario:'),
                    dcc.Dropdown(
                        id='scenario-dropdown',
                        options=[
                            {'label': 'Base Case', 'value': 'base_case'},
                            {'label': 'Market Crash', 'value': 'market_crash'},
                            {'label': 'Volatility Spike', 'value': 'volatility_spike'},
                            {'label': 'Correlation Breakdown', 'value': 'correlation_breakdown'},
                            {'label': 'Stagflation', 'value': 'stagflation'},
                            {'label': 'Liquidity Crisis', 'value': 'liquidity_crisis'},
                        ],
                        value='base_case',
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Time Horizon (Days):'),
                    dcc.Slider(
                        id='time-horizon-slider',
                        min=5,
                        max=60,
                        value=20,
                        marks={i: str(i) for i in [5, 10, 20, 30, 40, 50, 60]},
                        step=5,
                    ),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                dcc.Graph(id='monte-carlo-graph'),
            ]),
            
            html.Div([
                html.Div([
                    html.H4('Scenario Statistics'),
                    html.Table(id='scenario-stats-table'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='var-comparison-graph'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Market Sentiment Analysis'),
            html.Div([
                html.Div([
                    html.Label('Symbol:'),
                    dcc.Dropdown(
                        id='sentiment-symbol-dropdown',
                        options=[],
                        value=None,
                        placeholder='Select a symbol',
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Time Window (Days):'),
                    dcc.Slider(
                        id='sentiment-window-slider',
                        min=1,
                        max=30,
                        value=7,
                        marks={i: str(i) for i in [1, 7, 14, 21, 30]},
                        step=1,
                    ),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                dcc.Graph(id='sentiment-time-graph'),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='sentiment-price-correlation'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.H4('Sentiment Impact'),
                    html.Table(id='sentiment-impact-table'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Machine Learning Risk Predictions'),
            html.Div([
                html.Div([
                    html.Label('Prediction Horizon (Days):'),
                    dcc.Slider(
                        id='prediction-horizon-slider',
                        min=1,
                        max=30,
                        value=10,
                        marks={i: str(i) for i in [1, 5, 10, 15, 20, 25, 30]},
                        step=1,
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label('Use Sentiment Data:'),
                    dcc.RadioItems(
                        id='use-sentiment-radio',
                        options=[
                            {'label': 'Yes', 'value': 'yes'},
                            {'label': 'No', 'value': 'no'},
                        ],
                        value='yes',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    ),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Div([
                dcc.Graph(id='volatility-prediction-graph'),
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='var-prediction-graph'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.H4('Model Performance Metrics'),
                    html.Table(id='model-metrics-table'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
        ])

def get_sample_data():
    dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='B')
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL']
    
    market_data = []
    for symbol in symbols:
        np.random.seed(symbols.index(symbol))
        close = 100 * (1 + 0.0002 * np.random.randn(len(dates)) + 0.0001).cumprod()
        high = close * (1 + 0.005 + 0.01 * np.random.rand(len(dates)))
        low = close * (1 - 0.005 - 0.01 * np.random.rand(len(dates)))
        volume = np.random.randint(1000000, 10000000, len(dates))
        
        symbol_data = pd.DataFrame({
            'Date': dates,
            'Symbol': symbol,
            'Open': close[0] if i == 0 else close[i-1] for i in range(len(dates)),
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        market_data.append(symbol_data)
    
    market_df = pd.concat(market_data)
    market_df['DailyReturn'] = market_df.groupby('Symbol')['Close'].pct_change()
    market_df['LogReturn'] = np.log(market_df.groupby('Symbol')['Close'].pct_change() + 1)
    
    trade_data = []
    for symbol in symbols:
        np.random.seed(symbols.index(symbol))
        shares = np.random.randint(100, 1000)
        last_price = market_df[market_df['Symbol'] == symbol]['Close'].iloc[-1]
        
        trade_data.append({
            'Symbol': symbol,
            'Quantity': shares,
            'Price': last_price,
            'TradeDate': dates[-1]
        })
    
    trade_df = pd.DataFrame(trade_data)
    
    return market_df, trade_df

@app.callback(
    [Output('portfolio-var-graph', 'figure'),
     Output('portfolio-es-graph', 'figure'),
     Output('portfolio-stats-table', 'children'),
     Output('asset-allocation-pie', 'figure')],
    [Input('update-button', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date')]
)
def update_portfolio_tab(n_clicks, start_date, end_date):
    market_df, trade_df = get_sample_data()
    
    if start_date is not None and end_date is not None:
        market_df = market_df[(market_df['Date'] >= start_date) & (market_df['Date'] <= end_date)]
    
    risk_calc = RiskCalculator(confidence_level=0.95)
    trade_risk = risk_calc.calculate_trade_risk_metrics(trade_df, market_df)
    portfolio_metrics = risk_calc.calculate_portfolio_metrics(trade_risk, market_df)
    
    var_fig = go.Figure()
    es_fig = go.Figure()
    
    for symbol in trade_df['Symbol'].unique():
        symbol_data = trade_risk[trade_risk['Symbol'] == symbol]
        var_fig.add_trace(go.Bar(
            x=[symbol],
            y=[symbol_data['VaR_95'].iloc[0]],
            name=symbol
        ))
        es_fig.add_trace(go.Bar(
            x=[symbol],
            y=[symbol_data['ES_95'].iloc[0]],
            name=symbol
        ))
    
    var_fig.add_trace(go.Bar(
        x=['Portfolio'],
        y=[portfolio_metrics['VaR_95']],
        name='Portfolio',
        marker_color='red'
    ))
    es_fig.add_trace(go.Bar(
        x=['Portfolio'],
        y=[portfolio_metrics['ES_95']],
        name='Portfolio',
        marker_color='red'
    ))
    
    var_fig.update_layout(
        title='Value at Risk (95% Confidence)',
        xaxis_title='Symbol',
        yaxis_title='VaR ($)',
        barmode='group'
    )
    
    es_fig.update_layout(
        title='Expected Shortfall (95% Confidence)',
        xaxis_title='Symbol',
        yaxis_title='ES ($)',
        barmode='group'
    )
    
    stats_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Metric'),
                html.Th('Value')
            ])
        ),
        html.Tbody([
            html.Tr([html.Td('Total Portfolio Value'), html.Td(f"${portfolio_metrics['TotalValue']:.2f}")]),
            html.Tr([html.Td('Portfolio VaR (95%)'), html.Td(f"${portfolio_metrics['VaR_95']:.2f}")]),
            html.Tr([html.Td('Portfolio ES (95%)'), html.Td(f"${portfolio_metrics['ES_95']:.2f}")]),
            html.Tr([html.Td('Diversification Benefit'), html.Td(f"{portfolio_metrics['DiversificationBenefit']:.2%}")]),
            html.Tr([html.Td('Loss Probability (5%)'), html.Td(f"{portfolio_metrics['Loss_Prob_5pct']:.2%}")]),
            html.Tr([html.Td('Loss Probability (10%)'), html.Td(f"{portfolio_metrics['Loss_Prob_10pct']:.2%}")]),
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse'})
    
    allocation_fig = px.pie(
        trade_risk,
        values='Value',
        names='Symbol',
        title='Portfolio Allocation'
    )
    
    return var_fig, es_fig, stats_table, allocation_fig

@app.callback(
    [Output('symbol-dropdown', 'options'),
     Output('symbol-dropdown', 'value')],
    [Input('tabs', 'value')]
)
def update_symbol_dropdown(tab):
    if tab == 'tab-2':
        market_df, trade_df = get_sample_data()
        symbols = trade_df['Symbol'].unique()
        options = [{'label': symbol, 'value': symbol} for symbol in symbols]
        return options, symbols[0] if len(symbols) > 0 else None
    return [], None

@app.callback(
    [Output('trade-risk-graph', 'figure'),
     Output('trade-risk-table', 'children')],
    [Input('symbol-dropdown', 'value'),
     Input('risk-metric-dropdown', 'value')]
)
def update_trade_risk_tab(symbol, risk_metric):
    if symbol is None:
        return {}, None
    
    market_df, trade_df = get_sample_data()
    risk_calc = RiskCalculator(confidence_level=0.95)
    trade_risk = risk_calc.calculate_trade_risk_metrics(trade_df, market_df)
    
    symbol_data = trade_risk[trade_risk['Symbol'] == symbol]
    
    metric_labels = {
        'VaR_95': 'Value at Risk (95%)',
        'ES_95': 'Expected Shortfall (95%)',
        'Loss_Prob_5pct': 'Loss Probability (5%)',
        'Loss_Prob_10pct': 'Loss Probability (10%)'
    }
    
    risk_fig = go.Figure()
    
    for col in ['VaR_95', 'ES_95', 'Loss_Prob_5pct', 'Loss_Prob_10pct']:
        color = 'red' if col == risk_metric else 'blue'
        risk_fig.add_trace(go.Bar(
            x=[metric_labels[col]],
            y=[symbol_data[col].iloc[0]],
            name=metric_labels[col],
            marker_color=color
        ))
    
    risk_fig.update_layout(
        title=f'Risk Metrics for {symbol}',
        xaxis_title='Metric',
        yaxis_title='Value'
    )
    
    risk_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Metric'),
                html.Th('Value')
            ])
        ),
        html.Tbody([
            html.Tr([html.Td('Symbol'), html.Td(symbol)]),
            html.Tr([html.Td('Position Value'), html.Td(f"${symbol_data['Value'].iloc[0]:.2f}")]),
            html.Tr([html.Td('Value at Risk (95%)'), html.Td(f"${symbol_data['VaR_95'].iloc[0]:.2f}")]),
            html.Tr([html.Td('Expected Shortfall (95%)'), html.Td(f"${symbol_data['ES_95'].iloc[0]:.2f}")]),
            html.Tr([html.Td('Loss Probability (5%)'), html.Td(f"{symbol_data['Loss_Prob_5pct'].iloc[0]:.2%}")]),
            html.Tr([html.Td('Loss Probability (10%)'), html.Td(f"{symbol_data['Loss_Prob_10pct'].iloc[0]:.2%}")]),
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse'})
    
    return risk_fig, risk_table

@app.callback(
    [Output('monte-carlo-graph', 'figure'),
     Output('scenario-stats-table', 'children'),
     Output('var-comparison-graph', 'figure')],
    [Input('scenario-dropdown', 'value'),
     Input('time-horizon-slider', 'value')]
)
def update_stress_test_tab(scenario, time_horizon):
    market_df, trade_df = get_sample_data()
    
    risk_calc = RiskCalculator()
    trade_risk = risk_calc.calculate_trade_risk_metrics(trade_df, market_df)
    
    position_values = []
    for symbol in trade_risk['Symbol'].unique():
        symbol_data = trade_risk[trade_risk['Symbol'] == symbol]
        position_values.append(symbol_data['Value'].iloc[0])
    
    position_values = np.array(position_values)
    
    symbols = trade_risk['Symbol'].unique()
    
    returns_data = []
    for symbol in symbols:
        symbol_market = market_df[market_df['Symbol'] == symbol]
        returns_data.append(symbol_market['DailyReturn'].dropna().values)
    
    means = np.array([np.mean(ret) for ret in returns_data])
    
    cov_matrix = np.zeros((len(symbols), len(symbols)))
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            # Fill with reasonable correlation values
            if i == j:
                cov_matrix[i, j] = np.std(returns_data[i]) ** 2
            else:
                # Assume 0.3 correlation between different assets
                cov_matrix[i, j] = 0.3 * np.std(returns_data[i]) * np.std(returns_data[j])
    
    simulator = MonteCarloSimulator(iterations=1000)
    scenarios = simulator.define_standard_scenarios()
    
    # Run stress test for selected scenario and base case for comparison
    test_scenarios = {'base_case': scenarios['base_case']}
    if scenario != 'base_case':
        test_scenarios[scenario] = scenarios[scenario]
    
    stress_results = simulator.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=cov_matrix,
        scenarios=test_scenarios,
        time_periods=time_horizon
    )
    
    mc_fig = go.Figure()
    
    # Plot mean path and confidence interval
    x_days = np.arange(0, time_horizon + 1)
    
    for sc_name in stress_results.keys():
        sc_results = stress_results[sc_name]
        
        if sc_name == 'base_case':
            line_color = 'blue'
            fill_color = 'rgba(0, 0, 255, 0.1)'
        else:
            line_color = 'red'
            fill_color = 'rgba(255, 0, 0, 0.1)'
        
        # Mean path
        mc_fig.add_trace(go.Scatter(
            x=x_days,
            y=sc_results['mean_path'],
            mode='lines',
            name=f"{sc_name.replace('_', ' ').title()} Mean",
            line=dict(color=line_color)
        ))
        
        # Confidence interval
        mc_fig.add_trace(go.Scatter(
            x=np.concatenate([x_days, x_days[::-1]]),
            y=np.concatenate([sc_results['upper_bound'], sc_results['lower_bound'][::-1]]),
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)'),
            name=f"{sc_name.replace('_', ' ').title()} 90% CI"
        ))
    
    mc_fig.update_layout(
        title=f'Monte Carlo Simulation - {scenario.replace("_", " ").title()}',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)'
    )
    
    # Stats table for selected scenario
    sc_summary = stress_results[scenario]['summary']
    
    stats_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Metric'),
                html.Th('Value')
            ])
        ),
        html.Tbody([
            html.Tr([html.Td('Mean Return'), html.Td(f"{sc_summary['mean_return']:.2%}")]),
            html.Tr([html.Td('Median Return'), html.Td(f"{sc_summary['median_return']:.2%}")]),
            html.Tr([html.Td('Return Std Dev'), html.Td(f"{sc_summary['std_return']:.2%}")]),
            html.Tr([html.Td('VaR (95%)'), html.Td(f"{sc_summary['var_95']:.2%}")]),
            html.Tr([html.Td('ES (95%)'), html.Td(f"{sc_summary['es_95']:.2%}")]),
            html.Tr([html.Td('Max Loss'), html.Td(f"{sc_summary['max_loss']:.2%}")]),
            html.Tr([html.Td('Max Gain'), html.Td(f"{sc_summary['max_gain']:.2%}")]),
            html.Tr([html.Td('Probability of Loss'), html.Td(f"{sc_summary['prob_loss']:.2%}")]),
        ])
    ], style={'width': '100%', 'border-collapse': 'collapse'})
    
    # VaR comparison across scenarios
    var_fig = go.Figure()
    
    all_scenarios = list(scenarios.keys())
    scenario_vars = []
    
    for sc_name in all_scenarios:
        if sc_name in stress_results:
            scenario_vars.append(abs(stress_results[sc_name]['summary']['var_95']))
        else:
            scenario_vars.append(0)  # Placeholder
    
    var_fig.add_trace(go.Bar(
        x=[sc.replace('_', ' ').title() for sc in all_scenarios],
        y=scenario_vars,
        marker_color=['blue' if sc != scenario else 'red' for sc in all_scenarios]
    ))
    
    var_fig.update_layout(
        title='VaR Comparison Across Scenarios',
        xaxis_title='Scenario',
        yaxis_title='VaR (absolute %)',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return mc_fig, stats_table, var_fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
