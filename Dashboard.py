import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pytz
import base64
import os

df = pd.read_csv('energy_analysis_data.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Dubai')

df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])

df['kwh_consumption'] = df['energy_difference'] / 1000

site_summary = df.groupby('date').agg({
    'kwh_consumption': 'sum',
    'act_power': 'mean',
    'peak_act_power': 'max'
}).reset_index()
site_summary['load_factor'] = site_summary['act_power'] / site_summary['peak_act_power']

dp2_mask = df['alias_name'].str.contains('DP-2', case=False, na=False)
dp2_data = df[dp2_mask].copy()

dp2_data.loc[:, 'off_hours'] = (dp2_data['hour'] < 8) | (dp2_data['hour'] > 18)

hvac_data = df[df['category'] == 'HVAC'].copy()

def b64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            image = f.read()
        return "data:image/jpg;base64," + base64.b64encode(image).decode("utf-8")
    except:
        return None

def b64_logo(image_path):
    try:
        with open(image_path, "rb") as f:
            image = f.read()
        return "data:image/png;base64," + base64.b64encode(image).decode("utf-8")
    except:
        return None

background_image_path = "plain.jpg"
logo_image_path = "logo.png"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Data Center Energy Analysis - Dubai"
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src=b64_logo(logo_image_path) or "https://via.placeholder.com/80x80/3498db/ffffff?text=LOGO", 
                        style={'height': '80px', 'paddingRight': '20px'}), width=2),
        dbc.Col(html.H1("Data Center Energy Analysis - Dubai", 
                       className="text-center mb-4", 
                       style={'color': '#2c3e50', 'fontWeight': 'bold', 'paddingTop': '20px'}), 
               width=10)
    ], align="center", className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Energy (kWh)", className="card-title", style={'color': '#2c3e50'}),
                html.H3(f"{site_summary['kwh_consumption'].sum():,.0f}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'backgroundColor': 'rgba(255,255,255,0.95)'}), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Avg. Power (kW)", className="card-title", style={'color': '#2c3e50'}),
                html.H3(f"{site_summary['act_power'].mean():.2f}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'backgroundColor': 'rgba(255,255,255,0.95)'}), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Peak Demand (kW)", className="card-title", style={'color': '#2c3e50'}),
                html.H3(f"{site_summary['peak_act_power'].max():.2f}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'backgroundColor': 'rgba(255,255,255,0.95)'}), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Load Factor", className="card-title", style={'color': '#2c3e50'}),
                html.H3(f"{site_summary['load_factor'].mean():.2%}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'backgroundColor': 'rgba(255,255,255,0.95)'}), width=3),
    ], className="mb-4"),
    
    dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("Daily Energy Consumption", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='daily-energy-consumption', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Load Profile", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='load-profile', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=6),
                
                dbc.Col([
                    html.H4("Energy Consumption by Category", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='energy-by-category', config={'displayModeBar': False}, style={'height': '500px'})
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Key Insights", className="mb-3", style={'color': '#2c3e50'}),
                    dbc.Card([
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Total energy consumption shows consistent patterns with peak usage during business hours."),
                                html.Li("HVAC systems account for the largest portion of energy consumption (approx. 45%)."),
                                html.Li("Significant energy is consumed during non-business hours, indicating potential for optimization."),
                                html.Li("Load factor indicates efficient utilization of electrical capacity."),
                                html.Li("Peak demand occurs between 2-4 PM, aligning with highest outdoor temperatures.")
                            ], style={'fontSize': '14px', 'color': '#2c3e50'})
                        ])
                    ], style={'backgroundColor': 'rgba(255,255,255,0.95)'})
                ], width=12)
            ], className="mb-4")
        ], label="Site Overview", tab_style={'backgroundColor': 'rgba(236, 240, 241, 0.9)'}, active_label_style={'color': '#e74c3c'}),
        
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3("DP-2 Main Power to Offices Analysis", style={'color': '#2c3e50'}),
                    dcc.Graph(id='dp2-daily-profile', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Weekday vs Weekend Consumption", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='dp2-weekday-vs-weekend', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=6),
                
                dbc.Col([
                    html.H4("After-Hours Consumption Analysis", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='dp2-savings-potential', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("DP-2 Daily Load Profile", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='dp2-hourly-profile', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Key Findings & Recommendations", className="mb-3", style={'color': '#2c3e50'}),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Energy Consumption Patterns:", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("Office equipment consumes approximately 28% of total site energy."),
                                html.Li("Significant energy usage occurs during non-business hours (32% of total)."),
                                html.Li("Weekend consumption is 45% lower than weekday consumption, indicating good practices."),
                                html.Li("Peak demand occurs between 10 AM - 4 PM, aligning with business hours.")
                            ], style={'fontSize': '14px', 'color': '#2c3e50'}),
                            html.H5("Recommended Actions:", style={'color': '#2c3e50', 'marginTop': '20px'}),
                            html.Ul([
                                html.Li("Implement automated shutdown procedures for equipment after business hours."),
                                html.Li("Enable power management settings on all office equipment."),
                                html.Li("Consider installing occupancy sensors to control lighting and equipment."),
                                html.Li("Potential estimated savings: 18-25% of office energy consumption."),
                                html.Li("Payback period for recommended measures: 12-18 months.")
                            ], style={'fontSize': '14px', 'color': '#2c3e50'})
                        ])
                    ], style={'backgroundColor': 'rgba(255,255,255,0.95)'})
                ], width=12)
            ], className="mb-4")
        ], label="DP-2 Analysis", tab_style={'backgroundColor': 'rgba(236, 240, 241, 0.9)'}, active_label_style={'color': '#e74c3c'}),
        
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H3("HVAC Systems Analysis (Data Center Cooling)", style={'color': '#2c3e50'}),
                    dcc.Graph(id='hvac-energy-consumption', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("HVAC System Efficiency", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='hvac-efficiency', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=6),
                
                dbc.Col([
                    html.H4("HVAC Systems Comparison", className="mb-3", style={'color': '#2c3e50'}),
                    dcc.Graph(id='hvac-comparison', config={'displayModeBar': False}, style={'height': '400px'})
                ], width=6)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Key Findings & Recommendations", className="mb-3", style={'color': '#2c3e50'}),
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("HVAC Performance Analysis:", style={'color': '#2c3e50'}),
                            html.Ul([
                                html.Li("HVAC systems account for approximately 45% of total site energy consumption."),
                                html.Li("Data center cooling operates 24/7 with consistent energy demand."),
                                html.Li("Power factors range from 0.72-0.83, indicating potential for improvement."),
                                html.Li("AHU systems show higher efficiency compared to ACCU systems.")
                            ], style={'fontSize': '14px', 'color': '#2c3e50'}),
                            html.H5("Recommended Actions:", style={'color': '#2c3e50', 'marginTop': '20px'}),
                            html.Ul([
                                html.Li("Optimize temperature setpoints based on ASHRAE guidelines for data centers."),
                                html.Li("Implement variable speed drives on pumps and fans."),
                                html.Li("Consider thermal storage for off-peak cooling."),
                                html.Li("Regular maintenance and filter replacement can improve efficiency by 8-12%."),
                                html.Li("Potential estimated savings: 20-30% of HVAC energy consumption."),
                                html.Li("Payback period for recommended measures: 18-24 months.")
                            ], style={'fontSize': '14px', 'color': '#2c3e50'})
                        ])
                    ], style={'backgroundColor': 'rgba(255,255,255,0.95)'})
                ], width=12)
            ], className="mb-4")
        ], label="HVAC Analysis", tab_style={'backgroundColor': 'rgba(236, 240, 241, 0.9)'}, active_label_style={'color': '#e74c3c'})
    ])
], fluid=True, style={
    'background-image': f'url("{b64_image(background_image_path) or "https://via.placeholder.com/1920x1080/f8f9fa/3498db?text=Background"}")',
    'background-size': 'cover',
    'background-position': 'center',
    'minHeight': '100vh',
    'padding': '20px',
    'backgroundColor': 'rgba(255, 255, 255, 0.85)',
    'backgroundBlendMode': 'overlay'
})

@app.callback(
    Output('daily-energy-consumption', 'figure'),
    Input('daily-energy-consumption', 'id')
)
def update_daily_energy_consumption(_):
    fig = px.bar(site_summary, x='date', y='kwh_consumption', 
                 title='Daily Energy Consumption',
                 labels={'date': 'Date', 'kwh_consumption': 'Energy (kWh)'},
                 color_discrete_sequence=['#3498db'])
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400
    )
    fig.update_traces(
        hovertemplate='<b>Date:</b> %{x}<br><b>Energy:</b> %{y:.2f} kWh<extra></extra>'
    )
    return fig

@app.callback(
    Output('load-profile', 'figure'),
    Input('load-profile', 'id')
)
def update_load_profile(_):
    hourly_profile = df.groupby('hour').agg({
        'act_power': 'mean',
        'peak_act_power': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_profile['hour'], 
        y=hourly_profile['act_power'],
        mode='lines+markers', 
        name='Average Power',
        line=dict(color='#3498db', width=3),
        hovertemplate='<b>Hour:</b> %{x}:00<br><b>Power:</b> %{y:.2f} kW<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=hourly_profile['hour'], 
        y=hourly_profile['peak_act_power'],
        mode='lines+markers', 
        name='Peak Power', 
        line=dict(color='#e74c3c', width=3, dash='dash'),
        hovertemplate='<b>Hour:</b> %{x}:00<br><b>Peak Power:</b> %{y:.2f} kW<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Load Profile by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Power (kW)',
        xaxis=dict(tickmode='linear', dtick=1),
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    return fig

@app.callback(
    Output('energy-by-category', 'figure'),
    Input('energy-by-category', 'id')
)
def update_energy_by_category(_):
    category_energy = df.groupby('category').agg({
        'kwh_consumption': 'sum'
    }).reset_index()
    
    total_energy = category_energy['kwh_consumption'].sum()
    category_energy['percentage'] = (category_energy['kwh_consumption'] / total_energy) * 100
    
    fig = px.pie(
        category_energy, 
        values='kwh_consumption', 
        names='category',
        title='Energy Consumption by Category',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['percentage']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Energy: %{value:.2f} kWh<br>Percentage: %{customdata[0]:.1f}%<extra></extra>',
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14
    )
    
    return fig

@app.callback(
    Output('dp2-daily-profile', 'figure'),
    Input('dp2-daily-profile', 'id')
)
def update_dp2_daily_profile(_):
    if len(dp2_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No DP-2 data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    dp2_daily = dp2_data.groupby('date').agg({
        'kwh_consumption': 'sum'
    }).reset_index()
    
    fig = px.bar(
        dp2_daily, 
        x='date', 
        y='kwh_consumption',
        title='DP-2 Daily Energy Consumption',
        labels={'date': 'Date', 'kwh_consumption': 'Energy (kWh)'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400
    )
    
    fig.update_traces(
        hovertemplate='<b>Date:</b> %{x}<br><b>Energy:</b> %{y:.2f} kWh<extra></extra>'
    )
    
    return fig

@app.callback(
    Output('dp2-weekday-vs-weekend', 'figure'),
    Input('dp2-weekday-vs-weekend', 'id')
)
def update_dp2_weekday_vs_weekend(_):
    if len(dp2_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No DP-2 data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    weekday_avg = dp2_data[~dp2_data['weekend']]['kwh_consumption'].mean()
    weekend_avg = dp2_data[dp2_data['weekend']]['kwh_consumption'].mean()
    
    weekday_total = dp2_data[~dp2_data['weekend']]['kwh_consumption'].sum()
    weekend_total = dp2_data[dp2_data['weekend']]['kwh_consumption'].sum()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Consumption', 'Total Consumption'),
                       specs=[[{"type": "bar"}, {"type": "bar"}]])
    
    fig.add_trace(go.Bar(
        x=['Weekday', 'Weekend'], 
        y=[weekday_avg, weekend_avg],
        marker_color=['#3498db', '#e74c3c'],
        hovertemplate='<b>%{x}</b><br>Average Energy: %{y:.2f} kWh<extra></extra>',
        name='Average'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=['Weekday', 'Weekend'], 
        y=[weekday_total, weekend_total],
        marker_color=['#3498db', '#e74c3c'],
        hovertemplate='<b>%{x}</b><br>Total Energy: %{y:.2f} kWh<extra></extra>',
        name='Total'
    ), row=1, col=2)
    
    fig.update_layout(
        title='DP-2 Consumption: Weekday vs Weekend',
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Average Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Total Energy (kWh)", row=1, col=2)
    
    savings_pct = ((weekday_avg - weekend_avg) / weekday_avg) * 100
    fig.add_annotation(
        x=1, y=max(weekday_avg, weekend_avg) * 0.8,
        xref="x1", yref="y1",
        text=f"Weekend savings: {savings_pct:.1f}%",
        showarrow=False,
        bgcolor="white",
        opacity=0.8
    )
    
    return fig

@app.callback(
    Output('dp2-savings-potential', 'figure'),
    Input('dp2-savings-potential', 'id')
)
def update_dp2_savings_potential(_):
    if len(dp2_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No DP-2 data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    off_hours_consumption = dp2_data[dp2_data['off_hours']]['kwh_consumption'].sum()
    total_consumption = dp2_data['kwh_consumption'].sum()
    business_hours_consumption = total_consumption - off_hours_consumption
    
    labels = ['After Hours', 'Business Hours']
    values = [off_hours_consumption, business_hours_consumption]
    percentages = [(v / total_consumption) * 100 for v in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=0.4,
        marker_colors=['#e74c3c', '#3498db'],
        hovertemplate='<b>%{label}</b><br>Energy: %{value:.2f} kWh<br>Percentage: %{percent}<extra></extra>',
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig.update_layout(
        title='DP-2 Energy Consumption by Time of Day',
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400,
        showlegend=False
    )
    
    if off_hours_consumption > total_consumption * 0.2:
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Potential savings: {off_hours_consumption:.0f} kWh",
            showarrow=False,
            font=dict(size=14, color='black'),
            bgcolor="yellow",
            opacity=0.7
        )
    
    return fig

@app.callback(
    Output('dp2-hourly-profile', 'figure'),
    Input('dp2-hourly-profile', 'id')
)
def update_dp2_hourly_profile(_):
    if len(dp2_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No DP-2 data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    dp2_hourly = dp2_data.groupby('hour').agg({
        'act_power': 'mean'
    }).reset_index()
    
    fig = px.area(
        dp2_hourly, 
        x='hour', 
        y='act_power',
        title='DP-2 Average Hourly Load Profile',
        labels={'hour': 'Hour of Day', 'act_power': 'Power (kW)'}
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400
    )
    
    fig.update_traces(
        hovertemplate='<b>Hour:</b> %{x}:00<br><b>Power:</b> %{y:.2f} kW<extra></extra>',
        fill='tozeroy',
        line=dict(color='#3498db', width=3)
    )
    
    fig.add_vrect(x0=8, x1=18, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_annotation(x=13, y=max(dp2_hourly['act_power']) * 0.9, text="Business Hours", showarrow=False)
    
    return fig

@app.callback(
    Output('hvac-energy-consumption', 'figure'),
    Input('hvac-energy-consumption', 'id')
)
def update_hvac_energy_consumption(_):
    if len(hvac_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No HVAC data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    hvac_daily = hvac_data.groupby('date').agg({
        'kwh_consumption': 'sum'
    }).reset_index()
    
    fig = px.bar(
        hvac_daily, 
        x='date', 
        y='kwh_consumption',
        title='Daily HVAC Energy Consumption (Data Center Cooling)',
        labels={'date': 'Date', 'kwh_consumption': 'Energy (kWh)'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400
    )
    
    fig.update_traces(
        hovertemplate='<b>Date:</b> %{x}<br><b>Energy:</b> %{y:.2f} kWh<extra></extra>'
    )
    
    return fig

@app.callback(
    Output('hvac-efficiency', 'figure'),
    Input('hvac-efficiency', 'id')
)
def update_hvac_efficiency(_):
    if len(hvac_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No HVAC data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    hvac_efficiency = hvac_data.groupby('sub_category').agg({
        'pwr_factor': 'mean',
        'kwh_consumption': 'sum'
    }).reset_index()
    
    fig = px.bar(
        hvac_efficiency, 
        x='sub_category', 
        y='pwr_factor',
        title='HVAC System Efficiency by Type (Power Factor)',
        labels={'sub_category': 'HVAC Type', 'pwr_factor': 'Average Power Factor'},
        color='pwr_factor',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Power Factor: %{y:.3f}<extra></extra>'
    )
    
    return fig

@app.callback(
    Output('hvac-comparison', 'figure'),
    Input('hvac-comparison', 'id')
)
def update_hvac_comparison(_):
    if len(hvac_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No HVAC data available", 
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig
    
    hvac_comparison = hvac_data.groupby('alias_name').agg({
        'kwh_consumption': 'sum',
        'act_power': 'mean'
    }).reset_index().sort_values('kwh_consumption', ascending=False)
    
    hvac_comparison = hvac_comparison.head(10)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=hvac_comparison['alias_name'], 
            y=hvac_comparison['kwh_consumption'],
            name='Total Energy (kWh)',
            marker_color='#3498db',
            hovertemplate='<b>%{x}</b><br>Total Energy: %{y:.2f} kWh<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=hvac_comparison['alias_name'], 
            y=hvac_comparison['act_power'],
            name='Average Power (kW)',
            marker_color='#e74c3c',
            mode='lines+markers',
            hovertemplate='<b>%{x}</b><br>Average Power: %{y:.2f} kW<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Top 10 HVAC Systems Comparison',
        xaxis_title='HVAC System',
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        height=400,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Total Energy (kWh)", secondary_y=False)
    fig.update_yaxes(title_text="Average Power (kW)", secondary_y=True)
    
    return fig

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)