from dash import html

def get_legal_layout():
    return html.Div([
        html.H2("Legal & Disclaimers", style={"marginBottom": 18}),
        html.P("Legal information, disclaimers, and compliance details for StockLens."),
        html.Hr(),
        html.H3("General Disclaimer", style={"color": "#c0392b", "marginTop": 18}),
        html.P(
            "This tool is for educational and informational purposes only. It does not constitute investment advice, recommendation, or solicitation to buy or sell any securities. The creators and contributors of this tool are not SEBI-registered investment advisors. Users are solely responsible for their own investment decisions and should consult a SEBI-registered financial advisor before making any investment decisions. The creators and contributors assume no liability for any losses or damages arising from the use of this tool. Past performance is not indicative of future results. All investments are subject to market risks, including the possible loss of principal. By using this tool, you acknowledge and accept these risks.",
            style={"color": "#555", "fontSize": 15, "marginBottom": 18}
        ),
        html.H3("IPO GMP Disclaimer & Guidance", style={"color": "#1565c0", "marginTop": 18}),
        html.Ul([
            html.Li("Avoid investing in IPOs solely based on GMP — always consider broader fundamentals."),
            html.Li("Analyze company financials, promoter background, and business model before investing."),
            html.Li("Consult a SEBI-registered financial advisor before making any decisions."),
            html.Li("codebynik.live does not associate with any GMP market operators."),
            html.Li("GMP data is based on market perception and publicly available sources only."),
            html.Li("Use your own discretion and risk assessment when making IPO investment decisions.")
        ], style={"color": "#555", "fontSize": 15, "marginBottom": 18, "marginLeft": 24}),
        html.Hr(),
        html.P("For any legal queries or compliance information, contact support@stocklens.com.", style={"color": "#888", "fontSize": 14, "marginTop": 18})
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
