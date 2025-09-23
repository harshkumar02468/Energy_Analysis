import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
from datetime import datetime
import base64
import os

# Load data with error handling
try:
    df = pd.read_csv('python_assignment.csv')
except FileNotFoundError:
    # Create dummy data for testing if file not found
    print("CSV file not found, creating sample data for testing")
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    dummy_data = {
        'timestamp': dates,
        'group_id': ['group_1'] * len(dates),
        'group_name': ['Test Group'] * len(dates),
        'category': ['HVAC'] * len(dates),
        'sub_category': ['AC Unit'] * len(dates),
        'asset_name': ['Test Asset'] * len(dates),
        'energy_difference_calculated': np.random.uniform(0, 10, len(dates)),
        'start_time_1': ['08:00'] * len(dates),
        'end_time_1': ['18:00'] * len(dates),
        'start_time_2': [''] * len(dates),
        'end_time_2': [''] * len(dates)
    }
    df = pd.DataFrame(dummy_data)

df = df.dropna(subset=['group_id'])
df['energy_difference_calculated'] = pd.to_numeric(df['energy_difference_calculated'], errors='coerce').fillna(0)

# FIXED: Proper timestamp conversion without errors parameter
try:
    # First try converting from seconds (Unix timestamp)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    
    # If that fails (produces NaT), try direct datetime conversion
    nat_mask = df['timestamp'].isna()
    if nat_mask.any():
        df.loc[nat_mask, 'timestamp'] = pd.to_datetime(df.loc[nat_mask, 'timestamp'], errors='coerce')
    
    # Now handle timezone conversion safely
    if df['timestamp'].dt.tz is None:
        # Localize to UTC if timezone-naive
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Dubai')
    
except Exception as e:
    print(f"Error in timestamp conversion: {e}")
    # Fallback: create simple datetime without timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

df['date'] = df['timestamp'].dt.date
df['time'] = df['timestamp'].dt.time
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute

df['kwh_consumption'] = df['energy_difference_calculated'].clip(lower=0)

time_columns = ['start_time_1', 'end_time_1', 'start_time_2', 'end_time_2']
for col in time_columns:
    df[col] = df[col].fillna('')

def time_to_minutes(time_str):
    if pd.isna(time_str) or time_str == '' or time_str is None or time_str == '00:00':
        return None
    try:
        if ':' in time_str:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        else:
            return None
    except:
        return None

def is_within_operation_hours(row_time, start1, end1, start2, end2):
    time_minutes = row_time.hour * 60 + row_time.minute
    
    if start1 is None and start2 is None:
        return True
    
    if start1 is not None and end1 is not None and end1 != 0:
        if start1 <= end1:
            if start1 <= time_minutes < end1:
                return True
        else:
            if time_minutes >= start1 or time_minutes < end1:
                return True
    
    if start2 is not None and end2 is not None and end2 != 0:
        if start2 <= end2:
            if start2 <= time_minutes < end2:
                return True
        else:
            if time_minutes >= start2 or time_minutes < end2:
                return True
    
    return False

df['within_operation'] = df.apply(
    lambda row: is_within_operation_hours(
        row['time'],
        time_to_minutes(row['start_time_1']),
        time_to_minutes(row['end_time_1']),
        time_to_minutes(row['start_time_2']),
        time_to_minutes(row['end_time_2'])
    ), 
    axis=1
)

group_summary_data = []

for group_id in df['group_id'].unique():
    group_data = df[df['group_id'] == group_id]
    if len(group_data) == 0:
        continue
        
    first_row = group_data.iloc[0]
    
    total_consumption = group_data['kwh_consumption'].sum()
    inside_consumption = group_data[group_data['within_operation']]['kwh_consumption'].sum()
    outside_consumption = group_data[~group_data['within_operation']]['kwh_consumption'].sum()
    
    if total_consumption > 0:
        savings_potential_percentage = (outside_consumption / total_consumption * 100)
        savings_potential_percentage = min(savings_potential_percentage, 100)
    else:
        savings_potential_percentage = 0
    
    unique_dates = group_data['date'].unique()
    daily_savings = outside_consumption / len(unique_dates) if len(unique_dates) > 0 else 0
    
    group_summary_data.append({
        'group_id': group_id,
        'group_name': first_row.get('group_name', 'Unknown'),
        'category': first_row.get('category', 'Unknown'),
        'sub_category': first_row.get('sub_category', 'Unknown'),
        'asset_name': first_row.get('asset_name', 'Unknown'),
        'total_consumption': total_consumption,
        'inside_operation_consumption': inside_consumption,
        'outside_operation_consumption': outside_consumption,
        'savings_potential_percentage': savings_potential_percentage,
        'daily_savings_potential_kwh': daily_savings,
        'data_points': len(group_data),
        'days_analyzed': len(unique_dates)
    })

group_summary = pd.DataFrame(group_summary_data)

if len(group_summary) > 0:
    total_consumption = group_summary['total_consumption'].sum()
    total_outside_consumption = group_summary['outside_operation_consumption'].sum()

    if total_consumption > 0:
        overall_savings_potential = (total_outside_consumption / total_consumption * 100)
    else:
        overall_savings_potential = 0

    site_summary = {
        'total_assets': len(group_summary),
        'total_energy_consumption_kwh': total_consumption,
        'total_inside_operation_consumption_kwh': group_summary['inside_operation_consumption'].sum(),
        'total_outside_operation_consumption_kwh': total_outside_consumption,
        'overall_savings_potential_percentage': overall_savings_potential,
        'average_savings_potential_percentage': group_summary['savings_potential_percentage'].mean(),
        'assets_with_high_savings_potential': len(group_summary[group_summary['savings_potential_percentage'] > 50]),
        'top_savings_asset': group_summary.loc[group_summary['savings_potential_percentage'].idxmax()]['asset_name'] if len(group_summary) > 0 else 'N/A',
        'top_savings_percentage': group_summary['savings_potential_percentage'].max() if len(group_summary) > 0 else 0,
        'analysis_period_days': df['date'].nunique(),
        'total_data_points': len(df)
    }
else:
    # Default values if no data
    site_summary = {
        'total_assets': 0,
        'total_energy_consumption_kwh': 0,
        'total_inside_operation_consumption_kwh': 0,
        'total_outside_operation_consumption_kwh': 0,
        'overall_savings_potential_percentage': 0,
        'average_savings_potential_percentage': 0,
        'assets_with_high_savings_potential': 0,
        'top_savings_asset': 'N/A',
        'top_savings_percentage': 0,
        'analysis_period_days': 0,
        'total_data_points': 0
    }

# Prepare dashboard summary data
if len(group_summary) > 0:
    dashboard_summary = group_summary[[
        'group_id', 'group_name', 'category', 'sub_category', 'asset_name',
        'total_consumption', 'inside_operation_consumption', 'outside_operation_consumption',
        'savings_potential_percentage', 'daily_savings_potential_kwh', 'days_analyzed'
    ]].copy()

    for col in ['total_consumption', 'inside_operation_consumption', 'outside_operation_consumption', 'daily_savings_potential_kwh']:
        dashboard_summary[col] = dashboard_summary[col].round(3)

    dashboard_summary['savings_potential_percentage'] = dashboard_summary['savings_potential_percentage'].round(2)
    table_data = dashboard_summary.to_dict('records')
else:
    table_data = []

def encode_image(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except:
        return None

logo_encoded = encode_image("logo.png") if os.path.exists("logo.png") else None
background_encoded = encode_image("plain.jpg") if os.path.exists("plain.jpg") else None

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Energy Savings Potential Analysis - Dubai"

# Server configuration for Render
server = app.server

GRAPH_HEIGHT = 400
SMALL_GRAPH_HEIGHT = 350

background_style = {
    'padding': '20px', 
    'minHeight': '100vh'
}

if background_encoded:
    background_style.update({
        'background-image': f'url("{background_encoded}")',
        'background-repeat': 'no-repeat',
        'background-size': 'cover',
        'background-attachment': 'fixed',
        'background-color': 'rgba(248, 249, 250, 0.9)',
        'background-blend-mode': 'overlay'
    })
else:
    background_style.update({
        'backgroundColor': '#f8f9fa'
    })

# Modified layout with logo on left and header centered
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Img(
                    src=logo_encoded if logo_encoded else "https://via.placeholder.com/50x50/3498db/ffffff?text=Logo",
                    style={
                        'height': '70px',
                        'width': 'auto',
                        'display': 'inline-block',
                        'verticalAlign': 'middle',
                        'marginRight': '7px',
                        'float': 'left'
                    }
                ),
                html.Div([
                    html.H1("Energy Savings Potential Analysis - Dubai", 
                           style={
                               'color': '#2c3e50', 
                               'fontWeight': 'bold',
                               'marginBottom': '0',
                               'textAlign': 'center',
                               'marginLeft': '70px'  # Adjust based on logo width
                           })
                ], style={'textAlign': 'center', 'width': '100%'})
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-start'})
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Total Energy (kWh)", className="card-title", style={'color': '#2c3e50', 'fontSize': '14px'}),
                html.H3(f"{site_summary['total_energy_consumption_kwh']:,.0f}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '24px'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'height': '100px'}), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Savings Potential (kWh)", className="card-title", style={'color': '#2c3e50', 'fontSize': '14px'}),
                html.H3(f"{site_summary['total_outside_operation_consumption_kwh']:,.0f}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '24px'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'height': '100px'}), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Savings Potential (%)", className="card-title", style={'color': '#2c3e50', 'fontSize': '14px'}),
                html.H3(f"{site_summary['overall_savings_potential_percentage']:.1f}%", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '24px'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'height': '100px'}), width=3),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Assets Analyzed", className="card-title", style={'color': '#2c3e50', 'fontSize': '14px'}),
                html.H3(f"{site_summary['total_assets']}", 
                       className="card-text", 
                       style={'color': '#e74c3c', 'fontWeight': 'bold', 'fontSize': '24px'})
            ])
        ], color="light", outline=False, style={'border-left': '5px solid #3498db', 'height': '100px'}), width=3),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button("📊 Download Detailed Excel Report", 
                          id="download-btn",
                          color="success", 
                          size="lg"),
                dcc.Download(id="download-excel")
            ], className="d-grid gap-2")
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Energy Consumption by Operation Status", 
                             style={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'fontSize': '16px'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='operation-status-chart', 
                        config={'displayModeBar': False},
                        style={'height': f'{GRAPH_HEIGHT}px', 'width': '100%'}
                    )
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Savings Potential by Category", 
                             style={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'fontSize': '16px'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='savings-by-category', 
                        config={'displayModeBar': False},
                        style={'height': f'{GRAPH_HEIGHT}px', 'width': '100%'}
                    )
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Daily Energy Consumption Pattern", 
                             style={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'fontSize': '16px'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='daily-pattern', 
                        config={'displayModeBar': False},
                        style={'height': f'{SMALL_GRAPH_HEIGHT}px', 'width': '100%'}
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Energy Consumption by Category (Inside vs Outside Operation Hours)", 
                             style={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'fontSize': '16px'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='category-consumption', 
                        config={'displayModeBar': False},
                        style={'height': f'{SMALL_GRAPH_HEIGHT}px', 'width': '100%'}
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Energy Savings Summary - All Assets", 
                             style={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'fontSize': '16px'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='savings-table',
                        columns=[
                            {"name": "Asset Name", "id": "asset_name", "type": "text"},
                            {"name": "Category", "id": "category", "type": "text"},
                            {"name": "Total (kWh)", "id": "total_consumption", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                            {"name": "Inside (kWh)", "id": "inside_operation_consumption", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                            {"name": "Outside (kWh)", "id": "outside_operation_consumption", "type": "numeric", "format": Format(precision=3, scheme=Scheme.fixed)},
                            {"name": "Savings (%)", "id": "savings_potential_percentage", "type": "numeric", "format": Format(precision=2, scheme=Scheme.percentage)},
                        ],
                        data=table_data,
                        page_size=15,
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'fontFamily': 'Arial',
                            'fontSize': '11px',
                            'border': '1px solid #dee2e6'
                        },
                        style_header={
                            'backgroundColor': '#3498db',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'fontSize': '12px'
                        },
                        style_data={
                            'backgroundColor': 'white',
                            'color': '#2c3e50'
                        },
                        style_table={
                            'overflowX': 'auto', 
                            'height': '400px', 
                            'overflowY': 'auto',
                            'border': '1px solid #dee2e6'
                        },
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Key Insights and Results", 
                             style={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa', 'fontSize': '16px', 'color': '#2c3e50'}),
                dbc.CardBody([
                    html.Ul([
                        html.Li(f"Total energy consumption analyzed: {site_summary['total_energy_consumption_kwh']:,.2f} kWh"),
                        html.Li(f"Overall savings potential: {site_summary['overall_savings_potential_percentage']:.2f}% ({site_summary['total_outside_operation_consumption_kwh']:,.2f} kWh)"),
                        html.Li(f"{site_summary['assets_with_high_savings_potential']} assets have more than 50% savings potential"),
                        html.Li(f"Top savings opportunity: {site_summary['top_savings_asset']} with {site_summary['top_savings_percentage']:.2f}% potential savings"),
                        html.Li(f"Analysis period: {site_summary['analysis_period_days']} days with {site_summary['total_data_points']} data points"),
                        html.Li("Savings potential represents energy consumed outside of defined operation hours"),
                        html.Li("Implementing operational controls could significantly reduce energy waste")
                    ], style={'fontSize': '14px', 'lineHeight': '1.6'})
                ])
            ], style={'border': '1px solid #3498db'})
        ], width=12)
    ])
    
], fluid=True, style=background_style)

@app.callback(
    Output("download-excel", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_excel(n_clicks):
    # Create Excel file in memory for download
    df_excel = df.copy()
    # Remove timezone info for Excel compatibility
    if df_excel['timestamp'].dt.tz is not None:
        df_excel['timestamp'] = df_excel['timestamp'].dt.tz_localize(None)
    
    # Create a temporary file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        with pd.ExcelWriter(tmp.name, engine='xlsxwriter') as writer:
            if len(group_summary) > 0:
                group_summary.to_excel(writer, sheet_name='Asset Summary', index=False)
                
                site_overview = pd.DataFrame([site_summary])
                site_overview.to_excel(writer, sheet_name='Site Overview', index=False)
                
                category_summary = group_summary.groupby('category').agg({
                    'total_consumption': 'sum',
                    'inside_operation_consumption': 'sum',
                    'outside_operation_consumption': 'sum',
                    'group_id': 'count'
                }).reset_index()
                category_summary['savings_potential_percentage'] = (category_summary['outside_operation_consumption'] / category_summary['total_consumption'] * 100)
                category_summary = category_summary.rename(columns={'group_id': 'asset_count'})
                category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
        
        return dcc.send_file(tmp.name)

@app.callback(
    Output('operation-status-chart', 'figure'),
    Input('operation-status-chart', 'id')
)
def update_operation_status_chart(_):
    inside = df[df['within_operation']]['kwh_consumption'].sum()
    outside = df[~df['within_operation']]['kwh_consumption'].sum()
    
    fig = px.pie(
        values=[inside, outside],
        names=['Inside Operation Hours', 'Outside Operation Hours'],
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        height=GRAPH_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=80, l=50, r=50)
    )
    
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Energy: %{value:.2f} kWh<br>Percentage: %{percent}<extra></extra>',
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

@app.callback(
    Output('savings-by-category', 'figure'),
    Input('savings-by-category', 'id')
)
def update_savings_by_category(_):
    if len(group_summary) > 0:
        category_summary = group_summary.groupby('category').agg({
            'total_consumption': 'sum',
            'outside_operation_consumption': 'sum'
        }).reset_index()
        
        category_summary['savings_potential_percentage'] = (category_summary['outside_operation_consumption'] / category_summary['total_consumption'] * 100)
        
        category_summary['savings_potential_percentage'] = category_summary['savings_potential_percentage'].fillna(0)
        category_summary['savings_potential_percentage'] = category_summary['savings_potential_percentage'].clip(upper=100)
        
        fig = px.bar(
            category_summary,
            x='category',
            y='savings_potential_percentage',
            color='category',
            labels={'savings_potential_percentage': 'Savings Potential (%)', 'category': 'Category'}
        )
    else:
        fig = px.bar(labels={'x': 'Category', 'y': 'Savings Potential (%)'})
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        height=GRAPH_HEIGHT,
        showlegend=False,
        xaxis_title="Category",
        yaxis_title="Savings Potential (%)",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

@app.callback(
    Output('daily-pattern', 'figure'),
    Input('daily-pattern', 'id')
)
def update_daily_pattern(_):
    hourly_consumption = df.groupby('hour').agg({
        'kwh_consumption': 'mean'
    }).reset_index()
    
    hourly_consumption['operation_hours'] = (hourly_consumption['hour'] >= 8) & (hourly_consumption['hour'] <= 18)
    hourly_consumption['period'] = hourly_consumption['operation_hours'].map({True: 'Operation Hours (8:00-18:00)', False: 'Non-Operation Hours'})
    
    fig = px.bar(
        hourly_consumption,
        x='hour',
        y='kwh_consumption',
        color='period',
        labels={'hour': 'Hour of Day', 'kwh_consumption': 'Average Energy Consumption (kWh)'},
        color_discrete_map={'Operation Hours (8:00-18:00)': '#3498db', 'Non-Operation Hours': '#e74c3c'}
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=2),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        height=SMALL_GRAPH_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=80, l=50, r=50)
    )
    
    return fig

@app.callback(
    Output('category-consumption', 'figure'),
    Input('category-consumption', 'id')
)
def update_category_consumption(_):
    if len(group_summary) > 0:
        category_consumption = group_summary.groupby('category').agg({
            'inside_operation_consumption': 'sum',
            'outside_operation_consumption': 'sum'
        }).reset_index()
        
        fig = go.Figure(data=[
            go.Bar(name='Inside Operation Hours', x=category_consumption['category'], 
                   y=category_consumption['inside_operation_consumption'], marker_color='#3498db'),
            go.Bar(name='Outside Operation Hours', x=category_consumption['category'], 
                   y=category_consumption['outside_operation_consumption'], marker_color='#e74c3c')
        ])
    else:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(
        barmode='stack',
        title='',
        xaxis_title="Category",
        yaxis_title="Energy Consumption (kWh)",
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=12),
        height=SMALL_GRAPH_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=80, l=50, r=50)
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)