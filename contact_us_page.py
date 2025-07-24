from dash import html

def get_contact_us_layout():
    return html.Div([
        html.H2("Contact Us"),
        html.P("Reach out to the StockLens team for support, feedback, or partnership inquiries."),
        html.Div("Email: support@stocklens.com", style={"marginTop": 12, "fontSize": 16})
    ], style={"maxWidth": 700, "margin": "40px auto", "padding": 24, "background": "#f8f9fa", "borderRadius": 8})
