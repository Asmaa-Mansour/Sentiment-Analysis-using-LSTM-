# import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
from helper import predict, get_model
import warnings
warnings.filterwarnings("ignore")



model = get_model()


plot_data = pd.DataFrame()
plot_data['Sentiment'] = ['POSITIVE', 'NEGATIVE']
plot_data['Confidence'] = [0, 0]
            
plot1 = px.bar(plot_data, x='Sentiment', y='Confidence', range_y=[0, 100])


app = dash.Dash(name='Sentiment Analysis')

app.layout = html.Div(children=[
    html.H1(
        'Sentiment Analysis',
        style={'textAlign':'center', 'color': '#7161ef'}
    ),

    html.Div([
        # text box
        html.Div([
            dcc.Textarea(id='input-text',
            placeholder='Enter you review', style={'width': '50%', 'height': 150}),
            
            html.Div([
            html.H2(id='output')
            ], style={'margin': 'auto', 'color': '#7161ef', 'textAlign':'center'})
            
            
        ], style={'display': 'flex'}),
        
        html.Div([dcc.Graph(figure=plot1, )], id='plot1', 
        style={'margin': 'auto','width': '50%'})      

    ])

], style={'backgroundColor':'#FFFFFF'})


old_text = ''
@app.callback(
                [Output(component_id='output', component_property='children'),
                Output(component_id='plot1', component_property='children')],
                [Input(component_id='input-text', component_property='value')]

)
def update_output(value):
    # if n_clicks > 0 and value != old_text:
    if value != old_text:
        result, confidence = predict(model, value, 200)
        sentiment = 'POSITIVE' if result else 'NEGATIVE'
        
        confidence = confidence * 100
        #plot_data = pd.DataFrame()
        #plot_data['Sentiment'] = ['POSITIVE', 'NEGATIVE']
        if result == 1:
            plot_data['Confidence'] = [confidence, 100-confidence]
        else:
            plot_data['Confidence'] = [100-confidence, confidence]
            
        plot2 = px.bar(plot_data, x='Sentiment', y='Confidence', range_y=[0, 100], color='Sentiment')
    


        res = 'Your review was: {}, \nwith confidence: {:.2f}%'.format(sentiment, confidence)
        
        return res, dcc.Graph(figure=plot2)
    else:
        return None, dcc.Graph(figure=plot1)

if __name__ == '__main__':
    print('Serving on: http://localhost:1111/')
    app.run_server(host='172.17.0.2', port=1111, debug=False, dev_tools_ui=False, dev_tools_props_check=False)

