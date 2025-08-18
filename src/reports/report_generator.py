import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import jinja2
from pathlib import Path

try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.risk_engine.risk_calculator import RiskCalculator
from src.risk_engine.monte_carlo import MonteCarloSimulator
from src.models.risk_predictor import RiskPredictor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

class ReportGenerator:
    def __init__(self, report_dir: Optional[str] = None):
        self.report_dir = report_dir or os.path.join(os.path.dirname(__file__), "../../reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    def generate_risk_report(self, 
                            trade_risk_df: pd.DataFrame, 
                            portfolio_metrics: Dict, 
                            market_data: pd.DataFrame,
                            stress_test_results: Optional[Dict] = None,
                            sentiment_data: Optional[pd.DataFrame] = None,
                            ml_predictions: Optional[Dict] = None,
                            output_format: str = 'html',
                            output_filename: Optional[str] = None) -> str:
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"risk_report_{timestamp}"
        
        if output_filename is None:
            output_filename = f"{report_id}.{output_format}"
            
        output_path = os.path.join(self.report_dir, output_filename)
        
        report_data = {
            "report_id": report_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trade_risk": trade_risk_df.to_dict('records'),
            "portfolio_metrics": portfolio_metrics,
            "has_stress_test": stress_test_results is not None,
            "has_sentiment": sentiment_data is not None and not sentiment_data.empty,
            "has_ml_predictions": ml_predictions is not None
        }
        
        self._add_market_data_summary(report_data, market_data)
        
        if stress_test_results is not None:
            self._add_stress_test_data(report_data, stress_test_results)
            
        if sentiment_data is not None and not sentiment_data.empty:
            self._add_sentiment_data(report_data, sentiment_data, market_data)
            
        if ml_predictions is not None:
            self._add_ml_predictions(report_data, ml_predictions)
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if output_format == 'html':
                self._generate_html_report(report_data, output_path)
            elif output_format == 'pdf':
                self._generate_pdf_report(report_data, output_path)
            elif output_format == 'json':
                self._generate_json_report(report_data, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
            return output_path
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise
    
    def _add_market_data_summary(self, report_data: Dict, market_data: pd.DataFrame) -> None:
        latest_date = market_data['Date'].max()
        earliest_date = market_data['Date'].min()
        
        symbols = market_data['Symbol'].unique() if 'Symbol' in market_data.columns else ['MARKET']
        
        symbol_summaries = []
        for symbol in symbols:
            symbol_data = market_data[market_data['Symbol'] == symbol] if 'Symbol' in market_data.columns else market_data
            
            if symbol_data.empty:
                continue
                
            latest_price = symbol_data.sort_values('Date').iloc[-1]['Close']
            earliest_price = symbol_data.sort_values('Date').iloc[0]['Close']
            period_return = (latest_price / earliest_price) - 1
            
            daily_returns = symbol_data['DailyReturn'].dropna()
            volatility = daily_returns.std() * np.sqrt(252)
            
            symbol_summaries.append({
                "symbol": symbol,
                "latest_price": latest_price,
                "period_return": period_return,
                "volatility": volatility,
                "avg_volume": symbol_data['Volume'].mean() if 'Volume' in symbol_data.columns else None
            })
        
        report_data["market_summary"] = {
            "start_date": earliest_date.strftime("%Y-%m-%d"),
            "end_date": latest_date.strftime("%Y-%m-%d"),
            "symbols": symbol_summaries
        }
    
    def _add_stress_test_data(self, report_data: Dict, stress_test_results: Dict) -> None:
        scenarios_summary = []
        
        for scenario, results in stress_test_results.items():
            summary = results['summary']
            
            scenarios_summary.append({
                "name": scenario,
                "display_name": scenario.replace('_', ' ').title(),
                "mean_return": summary['mean_return'],
                "median_return": summary['median_return'],
                "std_return": summary['std_return'],
                "var_95": summary['var_95'],
                "es_95": summary['es_95'],
                "max_loss": summary['max_loss'],
                "prob_loss": summary['prob_loss']
            })
        
        report_data["stress_test"] = {
            "scenarios": scenarios_summary
        }
    
    def _add_sentiment_data(self, report_data: Dict, sentiment_data: pd.DataFrame, market_data: pd.DataFrame) -> None:
        sentiment_summary = []
        
        symbols = sentiment_data['symbol'].unique() if 'symbol' in sentiment_data.columns else ['MARKET']
        
        for symbol in symbols:
            symbol_sentiment = sentiment_data[sentiment_data['symbol'] == symbol] if 'symbol' in sentiment_data.columns else sentiment_data
            
            if symbol_sentiment.empty:
                continue
                
            latest_sentiment = symbol_sentiment.sort_values('date').iloc[-1]
            avg_compound = symbol_sentiment['compound'].mean()
            
            symbol_market = market_data[market_data['Symbol'] == symbol] if 'Symbol' in market_data.columns else market_data
            
            if not symbol_market.empty and 'Date' in symbol_market.columns and 'date' in symbol_sentiment.columns:
                symbol_sentiment['date'] = pd.to_datetime(symbol_sentiment['date'])
                symbol_market['Date'] = pd.to_datetime(symbol_market['Date'])
                
                merged = pd.merge_asof(
                    symbol_market.sort_values('Date'), 
                    symbol_sentiment.sort_values('date')[['date', 'compound']],
                    left_on='Date', right_on='date', 
                    direction='backward'
                )
                
                correlation = merged['DailyReturn'].corr(merged['compound']) if 'DailyReturn' in merged.columns else None
            else:
                correlation = None
            
            sentiment_summary.append({
                "symbol": symbol,
                "latest_compound": latest_sentiment['compound'],
                "latest_positive": latest_sentiment['positive'] if 'positive' in latest_sentiment else None,
                "latest_negative": latest_sentiment['negative'] if 'negative' in latest_sentiment else None,
                "avg_compound": avg_compound,
                "return_correlation": correlation
            })
        
        report_data["sentiment"] = {
            "summary": sentiment_summary
        }
    
    def _add_ml_predictions(self, report_data: Dict, ml_predictions: Dict) -> None:
        report_data["ml_predictions"] = ml_predictions
    
    def _generate_html_report(self, report_data: Dict, output_path: str) -> None:
        template = self._get_default_html_template() if not self._template_exists("risk_report.html") else self.jinja_env.get_template("risk_report.html")
        
        html_content = template.render(**report_data)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_pdf_report(self, report_data: Dict, output_path: str) -> None:
        html_path = output_path.replace('.pdf', '.html')
        self._generate_html_report(report_data, html_path)
        
        if WEASYPRINT_AVAILABLE:
            html = weasyprint.HTML(filename=html_path)
            html.write_pdf(output_path)
            
            if os.path.exists(html_path):
                os.remove(html_path)
        else:
            print(f"WeasyPrint not available. PDF generation skipped. HTML report saved at: {html_path}")
            return html_path
    
    def _generate_json_report(self, report_data: Dict, output_path: str) -> None:
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=4, default=self._json_serializer)
    
    def _json_serializer(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.float64)):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def _template_exists(self, template_name: str) -> bool:
        template_path = os.path.join(os.path.dirname(__file__), "templates", template_name)
        return os.path.exists(template_path)
    
    def _get_default_html_template(self) -> jinja2.Template:
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Risk Assessment Report - {{ report_id }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #eef1f5;
        }
        .summary-box {
            background-color: #e3f2fd;
            padding: 15px;
            border-left: 5px solid #2196F3;
            margin-bottom: 20px;
        }
        .alert {
            color: #721c24;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        .positive {
            color: #155724;
        }
        .negative {
            color: #721c24;
        }
        .neutral {
            color: #0c5460;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Risk Assessment Report</h1>
        <p>Generated on: {{ timestamp }}</p>
        <p>Report ID: {{ report_id }}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-box">
            <p>Total Portfolio Value: ${{ "%.2f"|format(portfolio_metrics.TotalValue) }}</p>
            <p>Portfolio VaR (95%): ${{ "%.2f"|format(portfolio_metrics.VaR_95) }} ({{ "%.2f"|format(portfolio_metrics.VaR_95 / portfolio_metrics.TotalValue * 100) }}%)</p>
            <p>Portfolio ES (95%): ${{ "%.2f"|format(portfolio_metrics.ES_95) }} ({{ "%.2f"|format(portfolio_metrics.ES_95 / portfolio_metrics.TotalValue * 100) }}%)</p>
            <p>Diversification Benefit: {{ "%.2f"|format(portfolio_metrics.DiversificationBenefit * 100) }}%</p>
        </div>

        {% if portfolio_metrics.VaR_95 / portfolio_metrics.TotalValue > 0.05 %}
        <div class="alert">
            <strong>Warning:</strong> The portfolio VaR is relatively high compared to the total portfolio value. Consider reducing exposure or implementing hedging strategies.
        </div>
        {% endif %}
    </div>

    <div class="section">
        <h2>Market Data Summary</h2>
        <p>Period: {{ market_summary.start_date }} to {{ market_summary.end_date }}</p>
        
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Latest Price</th>
                    <th>Period Return</th>
                    <th>Annualized Volatility</th>
                    <th>Avg Daily Volume</th>
                </tr>
            </thead>
            <tbody>
                {% for symbol in market_summary.symbols %}
                <tr>
                    <td>{{ symbol.symbol }}</td>
                    <td>${{ "%.2f"|format(symbol.latest_price) }}</td>
                    <td>{{ "%.2f"|format(symbol.period_return * 100) }}%</td>
                    <td>{{ "%.2f"|format(symbol.volatility * 100) }}%</td>
                    <td>{{ "{:,.0f}".format(symbol.avg_volume) if symbol.avg_volume else "N/A" }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Portfolio Risk Analysis</h2>
        
        <h3>Trade Level Risk</h3>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Position Value ($)</th>
                    <th>VaR (95%)</th>
                    <th>ES (95%)</th>
                    <th>Loss Prob (5%)</th>
                    <th>Loss Prob (10%)</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in trade_risk %}
                <tr>
                    <td>{{ trade.Symbol }}</td>
                    <td>${{ "%.2f"|format(trade.Value) }}</td>
                    <td>${{ "%.2f"|format(trade.VaR_95) }} ({{ "%.2f"|format(trade.VaR_95 / trade.Value * 100) }}%)</td>
                    <td>${{ "%.2f"|format(trade.ES_95) }} ({{ "%.2f"|format(trade.ES_95 / trade.Value * 100) }}%)</td>
                    <td>{{ "%.2f"|format(trade.Loss_Prob_5pct * 100) }}%</td>
                    <td>{{ "%.2f"|format(trade.Loss_Prob_10pct * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    {% if has_stress_test %}
    <div class="section">
        <h2>Stress Test Results</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Mean Return</th>
                    <th>Median Return</th>
                    <th>Return StdDev</th>
                    <th>VaR (95%)</th>
                    <th>ES (95%)</th>
                    <th>Max Loss</th>
                    <th>Prob of Loss</th>
                </tr>
            </thead>
            <tbody>
                {% for scenario in stress_test.scenarios %}
                <tr>
                    <td>{{ scenario.display_name }}</td>
                    <td>{{ "%.2f"|format(scenario.mean_return * 100) }}%</td>
                    <td>{{ "%.2f"|format(scenario.median_return * 100) }}%</td>
                    <td>{{ "%.2f"|format(scenario.std_return * 100) }}%</td>
                    <td>{{ "%.2f"|format(scenario.var_95 * 100) }}%</td>
                    <td>{{ "%.2f"|format(scenario.es_95 * 100) }}%</td>
                    <td>{{ "%.2f"|format(scenario.max_loss * 100) }}%</td>
                    <td>{{ "%.2f"|format(scenario.prob_loss * 100) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    {% if has_sentiment %}
    <div class="section">
        <h2>Market Sentiment Analysis</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Latest Sentiment</th>
                    <th>Positive Score</th>
                    <th>Negative Score</th>
                    <th>Avg Sentiment</th>
                    <th>Return Correlation</th>
                </tr>
            </thead>
            <tbody>
                {% for item in sentiment.summary %}
                <tr>
                    <td>{{ item.symbol }}</td>
                    <td class="{% if item.latest_compound > 0.1 %}positive{% elif item.latest_compound < -0.1 %}negative{% else %}neutral{% endif %}">
                        {{ "%.2f"|format(item.latest_compound) }}
                    </td>
                    <td>{{ "%.2f"|format(item.latest_positive) if item.latest_positive != None else "N/A" }}</td>
                    <td>{{ "%.2f"|format(item.latest_negative) if item.latest_negative != None else "N/A" }}</td>
                    <td class="{% if item.avg_compound > 0.1 %}positive{% elif item.avg_compound < -0.1 %}negative{% else %}neutral{% endif %}">
                        {{ "%.2f"|format(item.avg_compound) }}
                    </td>
                    <td>{{ "%.2f"|format(item.return_correlation) if item.return_correlation != None else "N/A" }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h3>Sentiment Impact Analysis</h3>
        <p>The correlation between news sentiment and price movements indicates 
        {% if sentiment.summary[0].return_correlation > 0.3 %}
        a strong positive relationship, suggesting the market is sentiment-driven.
        {% elif sentiment.summary[0].return_correlation > 0 %}
        a weak positive relationship.
        {% elif sentiment.summary[0].return_correlation > -0.3 %}
        little or no significant relationship.
        {% else %}
        a negative relationship, which may indicate contrarian market behavior.
        {% endif %}
        </p>
    </div>
    {% endif %}

    {% if has_ml_predictions %}
    <div class="section">
        <h2>Machine Learning Risk Predictions</h2>
        
        <h3>Predicted Risk Metrics (Next 10 Days)</h3>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Predicted Volatility</th>
                    <th>Predicted VaR (95%)</th>
                    <th>Predicted ES (95%)</th>
                </tr>
            </thead>
            <tbody>
                {% for symbol, metrics in ml_predictions.items() %}
                {% if symbol != 'Portfolio' %}
                <tr>
                    <td>{{ symbol }}</td>
                    <td>{{ "%.2f"|format(metrics.PredictedVolatility * 100) }}%</td>
                    <td>${{ "%.2f"|format(metrics.VaR) }} ({{ "%.2f"|format(metrics.VaR / metrics.Value * 100) }}%)</td>
                    <td>${{ "%.2f"|format(metrics.ES) }} ({{ "%.2f"|format(metrics.ES / metrics.Value * 100) }}%)</td>
                </tr>
                {% endif %}
                {% endfor %}
                
                {% if 'Portfolio' in ml_predictions %}
                <tr>
                    <td><strong>Portfolio</strong></td>
                    <td>-</td>
                    <td>${{ "%.2f"|format(ml_predictions.Portfolio.VaR) }} ({{ "%.2f"|format(ml_predictions.Portfolio.VaR / ml_predictions.Portfolio.Value * 100) }}%)</td>
                    <td>${{ "%.2f"|format(ml_predictions.Portfolio.ES) }} ({{ "%.2f"|format(ml_predictions.Portfolio.ES / ml_predictions.Portfolio.Value * 100) }}%)</td>
                </tr>
                {% endif %}
            </tbody>
        </table>
    </div>
    {% endif %}

    <div class="section">
        <h2>Recommendations</h2>
        
        <ul>
            {% if portfolio_metrics.VaR_95 / portfolio_metrics.TotalValue > 0.05 %}
            <li>Consider reducing exposure to higher risk assets or implementing hedging strategies to reduce overall portfolio risk.</li>
            {% endif %}
            
            {% if portfolio_metrics.DiversificationBenefit < 0.2 %}
            <li>Portfolio diversification appears insufficient. Consider adding assets with lower correlation to improve the diversification benefit.</li>
            {% endif %}
            
            {% if has_sentiment and sentiment.summary[0].latest_compound < -0.2 %}
            <li>Negative market sentiment detected. Monitor market conditions closely and consider defensive positioning.</li>
            {% elif has_sentiment and sentiment.summary[0].latest_compound > 0.2 %}
            <li>Positive market sentiment detected. Market conditions appear favorable but remain vigilant for sentiment shifts.</li>
            {% endif %}
            
            {% if has_stress_test and stress_test.scenarios|selectattr('name', 'equalto', 'market_crash')|first %}
            {% set crash_scenario = stress_test.scenarios|selectattr('name', 'equalto', 'market_crash')|first %}
            {% if crash_scenario.var_95 > 0.15 %}
            <li>Portfolio exhibits high vulnerability to market crash scenarios. Consider implementing tail risk hedging strategies.</li>
            {% endif %}
            {% endif %}
        </ul>
    </div>

    <div class="section">
        <h3>Disclaimer</h3>
        <p>This report is for informational purposes only and does not constitute investment advice. Past performance is not indicative of future results. All investments involve risk, including the possible loss of principal.</p>
    </div>
</body>
</html>
        """
        return jinja2.Template(html_template)


def create_report_templates_dir():
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    os.makedirs(templates_dir, exist_ok=True)
