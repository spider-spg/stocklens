
from dash import html

def get_privacy_policy_layout():
    return html.Div([
        html.H2("Privacy Policy", style={"textAlign": "center", "color": "#1565c0", "marginTop": 30}),
        html.Div([
            html.P("Your privacy is important to us. This Privacy Policy explains how we collect, use, and protect your information when you use StockLens Hub.", style={"fontSize": 16, "color": "#333", "marginBottom": 18}),
            html.H4("Information We Collect", style={"color": "#1565c0", "marginTop": 18}),
            html.Ul([
                html.Li("We do not collect any personal information unless you voluntarily provide it (e.g., via feedback forms or newsletter sign-up)."),
                html.Li("We may collect non-personal, aggregated analytics data to improve the app experience.")
            ], style={"fontSize": 15, "color": "#444"}),
            html.H4("How We Use Information", style={"color": "#1565c0", "marginTop": 18}),
            html.Ul([
                html.Li("To provide and improve the StockLens Hub experience."),
                html.Li("To respond to your feedback or inquiries if you contact us.")
            ], style={"fontSize": 15, "color": "#444"}),
            html.H4("Cookies", style={"color": "#1565c0", "marginTop": 18}),
            html.P("We may use cookies or similar technologies to enhance your experience. You can disable cookies in your browser settings.", style={"fontSize": 15, "color": "#444"}),
            html.H4("Third-Party Services", style={"color": "#1565c0", "marginTop": 18}),
            html.P("We may use third-party analytics or hosting services that may collect non-personal data as per their own privacy policies.", style={"fontSize": 15, "color": "#444"}),
            html.H4("Your Choices", style={"color": "#1565c0", "marginTop": 18}),
            html.P("You may choose not to provide personal information. You can unsubscribe from communications at any time.", style={"fontSize": 15, "color": "#444"}),
            html.H4("Contact Us", style={"color": "#1565c0", "marginTop": 18}),
            html.P("If you have any questions about this Privacy Policy, please contact us via the feedback form.", style={"fontSize": 15, "color": "#444"}),
            html.P("This policy may be updated from time to time. Please review it periodically.", style={"fontSize": 14, "color": "#888", "marginTop": 18})
        ], style={"maxWidth": 700, "margin": "0 auto", "background": "#fafdff", "padding": 30, "borderRadius": 8, "boxShadow": "0 2px 8px #e3e3e3"})
    ])
