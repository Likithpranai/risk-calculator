import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from weasyprint import HTML, CSS
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.data_loader import DataLoader
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator

def load_real_market_data(symbols, days=252):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    data_loader = DataLoader()
    
    market_data = data_loader.load_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source='polygon'
    )
    
    return market_data

def generate_trade_data_from_market(market_df):
    latest_dates = market_df.groupby('Symbol')['Date'].max()
    
    trade_data = []
    for symbol, latest_date in latest_dates.items():
        try:
            symbol_data = market_df[(market_df['Symbol'] == symbol) & 
                                 (market_df['Date'] == latest_date)]
            
            if symbol_data.empty:
                continue
                
            latest_price = symbol_data['Close'].iloc[0]
            quantity = np.random.randint(50, 500)
            
            trade_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Price': latest_price,
                'TradeDate': latest_date
            })
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    trade_df = pd.DataFrame(trade_data)
    return trade_df

def calculate_risk_metrics(market_df, trade_df):
    risk_calculator = RiskCalculator(confidence_level=0.95)
    
    trade_risk = risk_calculator.calculate_trade_risk_metrics(trade_df, market_df)
    portfolio_metrics = risk_calculator.calculate_portfolio_metrics(trade_risk, market_df)
    
    return trade_risk, portfolio_metrics

def run_monte_carlo_simulation(market_df, trade_risk):
    simulator = MonteCarloSimulator(iterations=1000)
    
    position_values = []
    symbols = []
    
    for _, row in trade_risk.iterrows():
        symbols.append(row['Symbol'])
        position_values.append(row['Value'])
    
    position_values = np.array(position_values)
    
    dates = sorted(market_df['Date'].unique())
    returns_by_symbol = {}
    returns_data = []
    
    for symbol in symbols:
        symbol_returns = []
        symbol_data = market_df[market_df['Symbol'] == symbol].sort_values('Date')
        returns_by_symbol[symbol] = symbol_data.set_index('Date')['DailyReturn'].to_dict()
        
        for date in dates:
            if date in symbol_data['Date'].values and not pd.isna(returns_by_symbol[symbol].get(date)):
                symbol_returns.append(returns_by_symbol[symbol][date])
        
        if symbol_returns:
            returns_data.append(np.array(symbol_returns))
    
    means = np.array([np.mean(ret) for ret in returns_data])
    
    cov_matrix = np.zeros((len(symbols), len(symbols)))
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if i == j:
                cov_matrix[i, j] = np.std(returns_data[i]) ** 2
            else:
                common_indices = np.logical_and(
                    ~np.isnan(returns_data[i][:min(len(returns_data[i]), len(returns_data[j]))]),
                    ~np.isnan(returns_data[j][:min(len(returns_data[i]), len(returns_data[j]))])
                )
                
                if np.sum(common_indices) > 2:
                    x = returns_data[i][:min(len(returns_data[i]), len(returns_data[j]))][common_indices]
                    y = returns_data[j][:min(len(returns_data[i]), len(returns_data[j]))][common_indices]
                    corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
                else:
                    corr = 0
                    
                cov_matrix[i, j] = corr * np.std(returns_data[i]) * np.std(returns_data[j])
    
    scenarios = simulator.define_standard_scenarios()
    
    stress_results = simulator.run_stress_test(
        position_values=position_values,
        means=means,
        covariance_matrix=cov_matrix,
        scenarios=scenarios,
        time_periods=20
    )
    
    return stress_results

def create_visualizations(market_df, trade_risk, portfolio_metrics, stress_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    symbols = trade_risk['Symbol'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(symbols)))
    
    for i, symbol in enumerate(symbols):
        symbol_data = market_df[market_df['Symbol'] == symbol].sort_values('Date')
        plt.plot(symbol_data['Date'], symbol_data['Close'], label=symbol, color=colors[i])
    
    plt.title('Price History of Portfolio Stocks')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_history.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.bar(trade_risk['Symbol'], trade_risk['VaR_95'], color='red', alpha=0.7)
    plt.title('Value at Risk (95% Confidence) by Stock')
    plt.xlabel('Symbol')
    plt.ylabel('VaR ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'var_by_stock.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    labels = []
    values = []
    colors = []
    
    scenario_colors = {
        'base_case': 'green',
        'market_crash': 'red',
        'volatility_spike': 'orange',
        'correlation_breakdown': 'purple',
        'stagflation': 'brown',
        'liquidity_crisis': 'black'
    }
    
    for scenario, results in stress_results.items():
        labels.append(scenario)
        values.append(abs(results['summary']['var_95']))
        colors.append(scenario_colors.get(scenario, 'blue'))
    
    plt.bar(labels, values, color=colors)
    plt.title('VaR (95%) Under Different Stress Scenarios')
    plt.ylabel('VaR (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stress_test_var.png'))
    plt.close()
    
    plt.figure(figsize=(8, 8))
    labels = trade_risk['Symbol']
    sizes = trade_risk['Value']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10(np.linspace(0, 1, len(labels))))
    plt.axis('equal')
    plt.title('Portfolio Composition by Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_composition.png'))
    plt.close()

def generate_professional_report(market_df, trade_risk, portfolio_metrics, stress_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    create_visualizations(market_df, trade_risk, portfolio_metrics, stress_results, output_dir)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Professional Risk Assessment Report</title>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
            
            :root {{
                --primary-color: #1a3a5f;
                --secondary-color: #3498db;
                --accent-color: #2ecc71;
                --warning-color: #e74c3c;
                --text-color: #2c3e50;
                --light-gray: #f5f5f5;
                --medium-gray: #e0e0e0;
                --border-color: #d1d1d1;
            }}
            
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Roboto', Arial, sans-serif;
                color: var(--text-color);
                line-height: 1.6;
                background-color: #f9f9f9;
                padding: 0;
                margin: 0;
            }}
            
            .report-container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            
            .header {{
                background-color: var(--primary-color);
                color: white;
                padding: 2rem;
                position: relative;
            }}
            
            .header h1 {{
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }}
            
            .header-meta {{
                font-size: 0.9rem;
                opacity: 0.8;
            }}
            
            .logo {{
                position: absolute;
                top: 2rem;
                right: 2rem;
                font-weight: bold;
                font-size: 1.2rem;
                color: white;
            }}
            
            .content {{
                padding: 2rem;
            }}
            
            .section {{
                margin-bottom: 2.5rem;
                border-radius: 4px;
                background-color: white;
                border: 1px solid var(--border-color);
            }}
            
            .section-header {{
                padding: 1rem 1.5rem;
                border-bottom: 1px solid var(--border-color);
                background-color: var(--light-gray);
            }}
            
            .section-header h2 {{
                font-size: 1.5rem;
                color: var(--primary-color);
            }}
            
            .section-content {{
                padding: 1.5rem;
            }}
            
            .executive-summary p {{
                margin-bottom: 1rem;
            }}
            
            .key-metric {{
                display: inline-block;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                padding: 1rem;
                margin: 0.5rem 1rem 0.5rem 0;
                min-width: 200px;
                background-color: var(--light-gray);
            }}
            
            .key-metric-label {{
                font-size: 0.8rem;
                text-transform: uppercase;
                color: var(--text-color);
                opacity: 0.7;
                margin-bottom: 0.3rem;
            }}
            
            .key-metric-value {{
                font-size: 1.8rem;
                font-weight: 500;
                color: var(--primary-color);
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 1.5rem;
                font-size: 0.9rem;
            }}
            
            th {{
                text-align: left;
                background-color: var(--light-gray);
                color: var(--primary-color);
                padding: 0.8rem 1rem;
                border-bottom: 2px solid var(--border-color);
                font-weight: 500;
            }}
            
            td {{
                padding: 0.8rem 1rem;
                border-bottom: 1px solid var(--border-color);
            }}
            
            tr:nth-child(even) {{
                background-color: var(--light-gray);
            }}
            
            .chart {{
                max-width: 100%;
                height: auto;
                margin: 1.5rem 0;
                box-shadow: 0 0 10px rgba(0,0,0,0.05);
                border: 1px solid var(--border-color);
                border-radius: 4px;
            }}
            
            .risk-high {{
                color: var(--warning-color);
                font-weight: 500;
            }}
            
            .risk-medium {{
                color: #f39c12;
                font-weight: 500;
            }}
            
            .risk-low {{
                color: var(--accent-color);
                font-weight: 500;
            }}
            
            .recommendations ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            
            .recommendations li {{
                padding: 0.6rem 0;
                border-bottom: 1px solid var(--medium-gray);
            }}
            
            .recommendations li:last-child {{
                border-bottom: none;
            }}
            
            .recommendations li:before {{
                content: "→";
                color: var(--secondary-color);
                margin-right: 0.5rem;
                font-weight: bold;
            }}
            
            .disclaimer {{
                font-size: 0.8rem;
                background-color: var(--light-gray);
                padding: 1rem;
                border-left: 4px solid var(--primary-color);
                margin-top: 2rem;
            }}
            
            footer {{
                text-align: center;
                padding: 2rem;
                font-size: 0.8rem;
                color: #777;
                border-top: 1px solid var(--border-color);
            }}
            
            @media print {{
                body {{
                    background-color: white;
                }}
                
                .report-container {{
                    box-shadow: none;
                    margin: 0;
                    max-width: 100%;
                }}
                
                .section {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="report-container">
        <div class="header">
            <h1>AI-Driven Trade Risk Assessment Report</h1>
            <div class="header-meta">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div class="logo">RiskAI Analytics</div>
        </div>
        
        <div class="content">
            <div class="section">
                <div class="section-header">
                    <h2>Executive Summary</h2>
                </div>
                <div class="section-content executive-summary">
            <p>This report presents a comprehensive risk assessment for a portfolio of less common stocks
            using real-time market data from Polygon API. The assessment includes Value at Risk (VaR),
            Expected Shortfall (ES), and probability metrics at both individual stock and portfolio levels.</p>
            
            <p>The portfolio has a total value of ${portfolio_metrics['TotalValue']:.2f} with a 
            Value at Risk (95% confidence) of ${portfolio_metrics['VaR_95']:.2f}, representing 
            {portfolio_metrics['VaR_95']/portfolio_metrics['TotalValue']*100:.2f}% of the total portfolio value.</p>
            
            <p>Portfolio diversification benefit: {portfolio_metrics['DiversificationBenefit']:.2f}</p>
            
            <div class="key-metrics-container">
                <div class="key-metric">
                    <div class="key-metric-label">Portfolio Value</div>
                    <div class="key-metric-value">${portfolio_metrics['TotalValue']:.2f}</div>
                </div>
                <div class="key-metric">
                    <div class="key-metric-label">Value at Risk (95%)</div>
                    <div class="key-metric-value">${portfolio_metrics['VaR_95']:.2f}</div>
                </div>
                <div class="key-metric">
                    <div class="key-metric-label">Diversification Benefit</div>
                    <div class="key-metric-value">{portfolio_metrics['DiversificationBenefit']:.2f}</div>
                </div>
            </div>
                </div>
            </div>
        
            <div class="section">
                <div class="section-header">
                    <h2>Portfolio Composition</h2>
                </div>
                <div class="section-content">
                    <img class="chart" src="portfolio_composition.png" alt="Portfolio Composition">
                    
                    <h3>Stock Price History</h3>
                    <img class="chart" src="price_history.png" alt="Price History">
                </div>
            </div>
        
            <div class="section">
                <div class="section-header">
                    <h2>Trade-Level Risk Metrics</h2>
                </div>
                <div class="section-content">
                    <table>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Price ($)</th>
                            <th>Value ($)</th>
                            <th>VaR 95% ($)</th>
                            <th>ES 95% ($)</th>
                            <th>Loss Probability (5%)</th>
                        </tr>
    """
    
    for _, row in trade_risk.iterrows():
        risk_class = ""
        if row['Loss_Prob_5pct'] > 0.1:
            risk_class = "risk-high"
        elif row['Loss_Prob_5pct'] > 0.05:
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
            
        html_content += f"""
                        <tr>
                            <td>{row['Symbol']}</td>
                            <td>{row['Quantity']}</td>
                            <td>{row['Price']:.2f}</td>
                            <td>{row['Value']:.2f}</td>
                            <td>{row['VaR_95']:.2f}</td>
                            <td>{row['ES_95']:.2f}</td>
                            <td class="{risk_class}">{row['Loss_Prob_5pct']:.2f}</td>
                        </tr>
        """
    
    html_content += """
                    </table>
                    
                    <h3>Value at Risk by Stock</h3>
                    <img class="chart" src="var_by_stock.png" alt="VaR by Stock">
                </div>
            </div>
        
            <div class="section">
                <div class="section-header">
                    <h2>Portfolio-Level Risk Metrics</h2>
                </div>
                <div class="section-content">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
    """
    
    for key, value in portfolio_metrics.items():
        html_content += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value:.2f}</td>
                        </tr>
        """
    
    html_content += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <h2>Monte Carlo Stress Test Results</h2>
                </div>
                <div class="section-content">
                    <img class="chart" src="stress_test_var.png" alt="Stress Test VaR">
                    
                    <h3>Scenario Analysis</h3>
                    <table>
                        <tr>
                            <th>Scenario</th>
                            <th>Mean Return (%)</th>
                            <th>VaR 95% (%)</th>
                            <th>ES 95% (%)</th>
                            <th>Probability of Loss</th>
                        </tr>
    """
    
    for scenario, results in stress_results.items():
        risk_class = ""
        if results['summary']['var_95'] > 0.1:
            risk_class = "risk-high"
        elif results['summary']['var_95'] > 0.05:
            risk_class = "risk-medium"
        else:
            risk_class = "risk-low"
            
        html_content += f"""
                        <tr>
                            <td>{scenario}</td>
                            <td>{results['summary']['mean_return']*100:.2f}%</td>
                            <td class="{risk_class}">{results['summary']['var_95']*100:.2f}%</td>
                            <td>{results['summary']['es_95']*100:.2f}%</td>
                            <td>{results['summary']['prob_loss']:.2f}</td>
                        </tr>
        """
    
    html_content += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <h2>Risk Assessment Conclusions</h2>
                </div>
                <div class="section-content">
                    <h3>Key Insights</h3>
                    <ul>
    """
    
    high_risk_stocks = trade_risk[trade_risk['Loss_Prob_5pct'] > 0.1]['Symbol'].tolist()
    worst_scenario = max(stress_results.items(), key=lambda x: x[1]['summary']['var_95'])
    
    html_content += f"""
                        <li>The portfolio shows a diversification benefit of {portfolio_metrics['DiversificationBenefit']:.2f}, indicating 
                        {("good" if portfolio_metrics['DiversificationBenefit'] > 0.1 else "moderate" if portfolio_metrics['DiversificationBenefit'] > 0 else "poor")} 
                        diversification.</li>
                        
                        <li>High-risk stocks: {', '.join(high_risk_stocks) if high_risk_stocks else "None"}</li>
                        
                        <li>The most severe impact would occur under the {worst_scenario[0]} scenario, with a VaR of 
                        {worst_scenario[1]['summary']['var_95']*100:.2f}% and probability of loss of {worst_scenario[1]['summary']['prob_loss']:.2f}.</li>
                        
                        <li>Overall portfolio risk: {("High" if portfolio_metrics['VaR_95']/portfolio_metrics['TotalValue'] > 0.05 else 
                                                 "Medium" if portfolio_metrics['VaR_95']/portfolio_metrics['TotalValue'] > 0.03 else "Low")}</li>
                    </ul>
                    
                    <h3>Recommendations</h3>
                    <div class="recommendations">
                    <ul>
    """
    
    if portfolio_metrics['DiversificationBenefit'] < 0.1:
        html_content += """
                        <li>Consider adding more uncorrelated assets to improve portfolio diversification.</li>
        """
    
    if high_risk_stocks:
        html_content += f"""
                        <li>Consider reducing position sizes in high-risk stocks: {', '.join(high_risk_stocks)}</li>
        """
    
    if worst_scenario[1]['summary']['var_95'] > 0.15:
        html_content += """
                        <li>Implement hedging strategies to protect against severe market downturns.</li>
        """
    
    html_content += """
                    </ul>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <h2>Disclaimer</h2>
                </div>
                <div class="section-content">
                    <div class="disclaimer">
                        <p>This report is for informational purposes only and does not constitute investment advice. 
                        Past performance is not indicative of future results. All investments involve risk, including the 
                        possible loss of principal.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>&copy; 2025 RiskAI Analytics. All rights reserved.</p>
        </footer>
    </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, 'risk_assessment_report.html')
    pdf_path = os.path.join(output_dir, 'risk_assessment_report.pdf')
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        print("Generating PDF report...")
        font_config = FontConfiguration()
        html = HTML(html_path)
        css = CSS(string='''
            @page {
                margin: 1cm;
                @bottom-right {
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 10pt;
                }
                @top-center {
                    content: "Risk Assessment Report";
                    font-size: 10pt;
                }
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
        ''')
        
        html.write_pdf(pdf_path, stylesheets=[css], font_config=font_config)
        
        print(f"Report generated at:\n- HTML: {html_path}\n- PDF: {pdf_path}")
        return pdf_path  # Return PDF path when successful
    except ImportError as e:
        print(f"Warning: Could not generate PDF report due to missing dependencies: {e}")
        print(f"HTML report generated at: {html_path}")
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        print(f"HTML report generated at: {html_path}")

    return html_path  # Return HTML path if PDF generation failed

def run_risk_assessment_and_generate_report(symbols=None):
    if symbols is None:
        symbols = [
            'ZM',    # Zoom Video Communications
            'SNAP',  # Snap Inc.
            'ETSY',  # Etsy Inc.
            'PLTR',  # Palantir Technologies
            'ROKU',  # Roku Inc.
            'DASH',  # DoorDash
            'CHWY',  # Chewy Inc.
            'U',     # Unity Software
            'LMND',  # Lemonade Insurance
            'CRSR'   # Corsair Gaming
        ]
    
    print(f"\n===== Running AI-Driven Trade Risk Assessment with Real Polygon Data =====\n")
    print(f"Using symbols: {', '.join(symbols)}")
    
    market_df = load_real_market_data(symbols)
    
    if market_df.empty:
        print("Failed to load market data from Polygon API.")
        return
    
    print(f"Loaded market data for {len(market_df['Symbol'].unique())} symbols")
    print(f"Date range: {market_df['Date'].min().date()} to {market_df['Date'].max().date()}")
    
    trade_df = generate_trade_data_from_market(market_df)
    
    if trade_df.empty:
        print("Failed to generate trade data.")
        return
    
    trade_risk, portfolio_metrics = calculate_risk_metrics(market_df, trade_df)
    
    print("\nTrade-Level Risk Metrics:")
    print(trade_risk[['Symbol', 'Value', 'VaR_95', 'ES_95', 'Loss_Prob_5pct']].round(2))
    
    print("\nPortfolio-Level Risk Metrics:")
    for key, value in portfolio_metrics.items():
        print(f"{key}: {value:.2f}")
    
    stress_results = run_monte_carlo_simulation(market_df, trade_risk)
    
    print("\nMonte Carlo Stress Test Results:")
    for scenario, results in stress_results.items():
        print(f"\nScenario: {scenario}")
        print(f"Mean Return: {results['summary']['mean_return']:.2%}")
        print(f"VaR (95%): {results['summary']['var_95']:.2%}")
        print(f"ES (95%): {results['summary']['es_95']:.2%}")
        print(f"Probability of Loss: {results['summary']['prob_loss']:.2f}")
    
    output_dir = os.path.join(os.path.dirname(__file__), "../reports/real_data_report")
    
    report_path = generate_html_report(market_df, trade_risk, portfolio_metrics, stress_results, output_dir)
    
    print(f"\n===== Risk Assessment Report Generated Successfully =====")
    print(f"Report available at: {report_path}")
    print(f"Visualizations saved to: {output_dir}")
    
    return report_path

if __name__ == "__main__":
    run_risk_assessment_and_generate_report()
