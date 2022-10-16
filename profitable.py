
# This is a Web server with next features:
# 1. View Portfolio structure in Pies
# 2. Portfolio view in details - Histogram view for every instrument
# 3. Today trades
# 4. Estimate trading efficiently

import os
import time
import dash
import datetime as dt
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from dash import callback_context
import plotly.graph_objects as go

import asyncio
import aiomoex
import aiohttp

import pandas as pd # sql operations
import pandas as pdp # for portfolio.csv
import sqlite3 as db

from styles import *

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(prevent_initial_callbacks=True)

Work_Folder = "C:/Trade/"
Models_Folder = "Models/"
Index = 'IMOEX' # as compare component

# returns: 
#         Volume of portfolio in %,
#         Profit in %,
#         Profit in $
#         Estimate in $
#         Deadline in months
def get_ticker_dashboard_info(ticker):
    vols = dfp.iloc[:, 0]    # % of Portfolio
    pr_perc = dfp.iloc[:, 4] # Profit %
    pr_money = dfp.iloc[:, 5]# Profit $
    estimate = dfp.iloc[:, 3]# Estimate in Money

    indx = 0
    for each in dfp['Instrument']:
        if str(ticker) in str(each):
            if ticker != str('Money'):
                ddln, t0 = get_deadline_in_months(ticker)
                div = (100.0*get_ticker_dividend(ticker))/float(t0)
                return str(vols[indx]).replace(',', '.'), str(pr_perc[indx]).replace(',', '.'), str(pr_money[indx]).replace(',', '.'), str(estimate[indx]).replace(',', '.'), ddln, div
            else: # special Money return
                return str(vols[indx]).replace(',', '.'), str('0'), str('0'), str(estimate[indx]).replace(',', '.'), str('0'), str('0')
        else:
            indx += 1

    return str('0'), str('0'), str('0'), str('0'), str('0'), str('0')

# Generates nav links for every quote
def generate_navs(quote_name):
    title = str(quote_name)
    if quote_name in archived:
        title = "<"+title+">"
        return html.Div(children=[dbc.NavLink(title, id=str(quote_name), href="/"+str(quote_name), active="exact", style={"display":"inline-block"}), 
                                  dbc.Button("+/-", id=str(quote_name+str("-add")), outline=True, size="sm")])
    return dbc.NavLink(title, id=str(quote_name), href="/"+str(quote_name), active="exact")

# APP entry point
app = dash.Dash(external_stylesheets=[dbc.themes.LUMEN])
app.title = 'Profitable'

cnxx = db.connect(Work_Folder + "Models\\" + 'QuotesSettings.sqlite3')
df = pd.read_sql_query("SELECT Quote FROM Parameters", cnxx)
cnxx.close()
# reading portfolio.csv table
dfp = pdp.read_csv(Work_Folder + "Reports\\" + "portfolio.csv", delimiter= ';', encoding='ISO-8859-1')
file_mod_time = os.stat(Work_Folder + "Reports\\" + "portfolio.csv").st_mtime

# discover archived instruments, from the folder
archived = [] # list of archived tickers
add_to_histogramm = []  # archived ticker add to compare histogramm

for file in os.listdir(Work_Folder + Models_Folder):
    if file.endswith(".sqlite3"):
        file = os.path.join(Work_Folder + Models_Folder, file)            
        head, tail = os.path.split(file)
        file = tail[0:tail.index('.'):None]
        file = file.upper()
        if file != "" and file != str('QUOTESSETTINGS'):
            bExist = False
            for voc in df['Quote']:
                if voc == file:
                    bExist = True
                    
            if bExist == False:
                archived.append(file)      

q_buttons = pd.DataFrame(df)['Quote'].tolist()

# full list of the instruments
all_instruments = q_buttons + archived

# loading progressbar
progress = html.Div( 
    [
        dcc.Loading(id="loading", type="circle", children=html.Div(id="loading-output")), 
    ]
)
# progressbar value during loading
iDwnld = int(0)

#logo graph at start app
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])

# sidebar
sidebar = html.Div(
    [
        html.H2("Instruments", className="display-5", style={'text-align' : 'center', 'margin-bottom' : '20px'}),
        progress,
        dbc.Nav([dbc.NavLink("DASHBOARD", id = "dashboard-btn", href="/DASHBOARD", active="exact"),
                 dbc.NavLink("PORTFOLIO", id = "portfolio-btn", href="/PORTFOLIO", active="exact"), 
                 dbc.NavLink("TODAY", id = "today-btn", href="/TODAY", active="exact")], vertical=True, pills=True,),
        html.Hr(),
        dbc.Nav(
            [generate_navs(i) for i in q_buttons],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        dbc.Nav(
            [generate_navs(i) for i in archived],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# main view
content = html.Div([dbc.Table.from_dataframe(dfp, id = 'portfolio-tbl', striped=True, bordered=True, hover=True)], id = 'content', style=CONTENT_STYLE)
current_btn_id = 'portfolio-btn' # last user button clicked id
app.layout = html.Div([dcc.Location(id="url"), sidebar, content], id = 'outer-container', style=FULL_WINDOW)

@app.callback(Output("loading-output", "children"), 
    [Input(str(i), "n_clicks") for i in all_instruments],
    [Input('portfolio-btn', "n_clicks")])
def input_triggers_spinner(*args):
    global iDwnld
    iDwnld = 5
    while iDwnld > 0:
        time.sleep(1)
    return ""

@app.callback(
    Output('content', 'children'),
    [Input(str(i), "n_clicks") for i in all_instruments],
    [Input('portfolio-btn', "n_clicks")],
    [Input('dashboard-btn', "n_clicks")],
    [Input('today-btn', "n_clicks")],
    [Input(str(i)+str("-add"), "n_clicks") for i in archived]
)
def b_clicked_callback(*args):
    global iDwnld
    global content
    global q_buttons
    global file_mod_time
    global current_btn_id
    url = "/"
    to_ret = content
    if callback_context.triggered[0] is not None:
        trigger = callback_context.triggered[0] 
        if trigger['prop_id'] != '.':
            ticker = str(trigger['prop_id'].split('.')[0])
            if ticker is not None:
                global dfp
                current_btn_id = ticker
                if ticker.find("-add") >= 0:
                    ticker = ticker.replace("-add", "")
                    if ticker not in add_to_histogramm:
                        add_to_histogramm.append(ticker)
                    else:
                        add_to_histogramm.remove(ticker)
                    ticker = "dashboard-btn" # goto dashboard mode
                    
                if ticker == "today-btn":
                    trades = get_today_trades(q_buttons)
                    to_ret = dbc.Table.from_dataframe(trades, id = 'portfolio-tbl', striped=True, bordered=True, hover=True)
                elif ticker == "portfolio-btn":
                    dfp = pdp.read_csv(Work_Folder + "Reports\\" + "portfolio.csv", delimiter= ';', encoding='ISO-8859-1')
                    to_ret = dbc.Table.from_dataframe(dfp, id = 'portfolio-tbl', striped=True, bordered=True, hover=True)
                elif ticker == "dashboard-btn":
                    vols = []
                    aps = []
                    apm = []
                    aestim = []
                    ddlns = []
                    divs = []
                    with_money = q_buttons.copy()
                    with_money.append('Money')

                    for quote in with_money:
                        vol, pp, pm, estim, ddln, div = get_ticker_dashboard_info(quote)
                        vols.append(float(vol))
                        apm.append(float(pm))
                        aestim.append(float(estim))
                        if(quote != 'Money'):
                            ddlns.append(float(ddln))
                            divs.append(float(div))
                            aps.append(float(pp))
                        
                    if len(add_to_histogramm) > 0:
                        for arch in add_to_histogramm:
                            ddln, t0 = get_deadline_in_months(arch)
                            div = (100.0*get_ticker_dividend(arch))/float(t0)
                            pp = get_archived_ticker_profit(arch)
                            ddlns.append(float(ddln))
                            divs.append(float(div))     
                            aps.append(float(pp))                            
                            
                    perc1 = go.Bar(x = q_buttons+add_to_histogramm, y = ddlns, name='Time, months')
                    dvdnd = go.Bar(x = q_buttons+add_to_histogramm, y = divs, name='Dividends, %')
                    perc2 = go.Bar(x = q_buttons+add_to_histogramm, y = aps,  name='Profit, %')
                    # draw pie
                    pie = go.Pie(labels = with_money, values = aestim)
                    figPie = go.Figure(data = [pie])
                    figPie.update_layout(
                        title={
                            'text': "Portfolio content",
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})                            
                    # draw bars
                    to_ret = html.Div(children=[ dcc.Graph(id='pie_plot', figure=figPie),
                                                 dcc.Graph(id='bar_plot', figure=go.Figure(data=[perc1, dvdnd, perc2], layout=go.Layout({"title":"Intrument profit, %"})))] ,     style=GRAPH_STYLE  )        
                else:
                    asyncio.run(download_imoex(ticker))
                    to_ret = dcc.Graph(id='live-update-graph', figure = fig, style=GRAPH_STYLE)

    iDwnld = 0
    return to_ret#, url

# returns: 
#         time of the instrument, in months
#         T0 money flow
def get_deadline_in_months(Ticker):
    dbfile = 'file:///' + Work_Folder + Models_Folder + Ticker + '.sqlite3' + '?mode=ro'
    # Dataframe from trading DB of the Ticker
    cnx = db.connect(dbfile, uri=True)
    df = pd.read_sql_query("SELECT Time, BudgetQuantity, BudgetMoney, BudgetBalance, Price, ChangeT0, Comission, BudgetSyntheticProfit FROM Market", cnx)
    count_row = df.shape[0]  # gives number of row count   
    begin = pd.to_datetime(df.loc[0]['Time'])
    last = pd.to_datetime(df.loc[count_row-1]['Time'])
    deadline = (last - begin).total_seconds()/3600/24/30
    cnx.close()
    return str(deadline), str(df.loc[0]['BudgetMoney'])

# returns:
#         sum of dividends for all time 
def get_ticker_dividend(Ticker):
    rup = str('\'rup')+Ticker+str('\'')
    rdv = str('\'rdv')+Ticker+str('\'')
    adv = str('\'adv')+Ticker+str('\'')
    hdv = str('\'hdv')+Ticker+str('\'')
    dbfile = Work_Folder + Models_Folder + 'QuotesSettings' + '.sqlite3'

    # Dataframe from trading DB of the Ticker
    cnx = db.connect(dbfile)
    query = ""
    if Ticker in q_buttons:
        query = "SELECT Sum(ProfitLoss) FROM History WHERE Quote = " + adv + " OR Quote = " + rdv + " OR Quote = " + rup
    else:
        query = "SELECT Sum(ProfitLoss) FROM History WHERE Quote = " + hdv + " OR Quote = " + rdv + " OR Quote = " + rup
    divs = pd.read_sql_query(query, cnx)
    cnx.close()
    
    ret_val = 0
    if divs.iloc[0][0] is not None:
        ret_val = divs.iloc[0][0].astype(float)

    return ret_val

# return:
#         sum of dividends for all time, for archived tickers only
def get_archived_ticker_profit(Ticker):
    dbfile = Work_Folder + Models_Folder + 'QuotesSettings' + '.sqlite3'
    # Dataframe from trading DB of the Ticker
    cnx = db.connect(dbfile)
    query = "SELECT Percent FROM History WHERE Quote = \'" + Ticker + "\'"
    profit = pd.read_sql_query(query, cnx)    
    return profit.iloc[0][0].astype(float)

#return:
    # list of trades today of the ticker
def get_today_trades(Tickers):
    
    # calc date for the query
    to = dt.datetime.now() # 
    fr = dt.datetime.now() - dt.timedelta(hours=to.hour, minutes=to.minute, seconds=to.second, microseconds= to.microsecond) # clocks to midnight current day
    
    today = '\'' + fr.isoformat(sep=' ') + '\'' 
    #today = '\'2020-12-23 23:55:00\''
    # Dataframe from trading DB of the Ticker
    df = pd.DataFrame()
    for ticker in Tickers:
        dbfile = 'file:///' + Work_Folder + Models_Folder + ticker + '.sqlite3' + '?mode=ro'
        cnx = db.connect(dbfile, uri=True)
        query = pd.read_sql_query("SELECT Time, Price, ChangeT0, Change, BudgetQuantity, BudgetBalance, Comission FROM Market WHERE Time > " + today, cnx)
        query['Quote'] = ticker
        query['Action'] = 0
        new_cols = query[['Quote', 'Time', 'Price', 'ChangeT0', 'Change', 'Action', 'BudgetQuantity', 'BudgetBalance', 'Comission']]
        # get record before for dtermine real order volume
        top1 = pd.read_sql_query("SELECT BudgetQuantity FROM Market WHERE Time < " + today + "ORDER BY Time DESC LIMIT 1", cnx)
        if len(top1) > 0:
            for i in range(0, len(new_cols)):
                if i == 0:
                    new_cols.iloc[0, 5] = str(query.iloc[0, 4].astype(int) - int(top1.iloc[0][0]))
                else:
                    new_cols.iloc[i, 5] = str(query.iloc[i, 4].astype(int) - query.iloc[i-1, 4].astype(int))
        #
        df = df.append(new_cols, ignore_index=True)
        cnx.close()

    return df#.sort_values('Time')
    
# Download and Build picture
async def download_imoex(Ticker):
    Quantity0 = 0
    global progress, iDwnld
    dbfile = 'file:///' + Work_Folder + Models_Folder + Ticker + '.sqlite3' + '?mode=ro'
    # Dataframe from trading DB of the Ticker
    cnx = db.connect(dbfile, uri=True)
    df = pd.read_sql_query("SELECT Time, BudgetQuantity, BudgetMoney, BudgetBalance, Price, ChangeT0, Comission, BudgetSyntheticProfit FROM Market", cnx)
    
    iDwnld = 5
    df['Time'] = pd.to_datetime(df['Time'])
    df['Price'] = df['Price'].astype(float)
    df['ChangeT0'] = df['ChangeT0'].astype(float)
    df['BudgetQuantity'] = df['BudgetQuantity'].astype(int)
    df['BudgetMoney'] = df['BudgetMoney'].astype(float)
    df['BudgetBalance'] = df['BudgetBalance'].astype(float)
    df['Comission'] = df['Comission'].astype(float)
    
    # Checking closed Quote or working
    iLI = df.last_valid_index()
    # if instrument is archived, its last BudgetQuantity is Zero
    # because decrease last index to first not Zero
    while df.loc[iLI]['BudgetQuantity'] == 0:
        iLI -= 1
       
    Quantity0 = df.loc[0]['BudgetQuantity']
    iPrice0 = df.loc[0]['Price']
    iLotSz0 = int(.5 + (df.loc[0]['BudgetMoney'] / (iPrice0*Quantity0)))
    EstimateT0Pure = Quantity0 * iLotSz0 * iPrice0
    
    # determine lotsize koeff after changes if it happens at MOEX market 
    real_lot_koeff = 1
    Estimate = float(df.loc[0]['BudgetQuantity']) * iLotSz0 * float(df.loc[0]['Price'])     
    Balance = Estimate + df.loc[0]['BudgetMoney']

    # Scan for lot changing point, if it was
    iPt = df.first_valid_index()
    while iPt < iLI-1:
        fM = df.loc[iPt+1]['BudgetQuantity']/df.loc[iPt]['BudgetQuantity']
        fB = df.loc[iPt+1]['BudgetBalance']/df.loc[iPt]['BudgetBalance']
        # lot increassed at 10
        if(fM > 9.0 and fM < 90.0 and fB < 2.0 and df.loc[iPt]['BudgetQuantity'] > 1 and df.loc[iPt+1]['BudgetQuantity'] > 1): 
            real_lot_koeff = 10
            break
        iPt+=1

    #print(real_lot_koeff, iLotSz, iLotSz0)
    count_row = df.shape[0]  # gives number of row count   
    cnx.close()
    col = [] # profit of the algorithm
    ind_pr = [] #profit of the index    

    # calc profit of instrument
    iDwnld = 50
    for i in range(0, iLI + 1, 1):
        Estimate = float(df.loc[i]['BudgetQuantity']) * iLotSz0 * float(df.loc[i]['Price'])
        if i > iPt:
            Estimate = Estimate / real_lot_koeff # apply lotsize koeff if was lotsize changes by MOEX 
        Balance = Estimate + df.loc[i]['BudgetMoney']

        EstimateT0 = (EstimateT0Pure + (EstimateT0Pure - df.loc[i]['BudgetMoney'] )) + df.loc[i]['Comission']
        dPar = ((Estimate - EstimateT0) / EstimateT0Pure ) * 100.0
        col.append(dPar)
    
    iDwnld = 90
    begin = df.loc[0]['Time']
    last = df.loc[count_row-1]['Time'] 
    if begin >= last:
        return  go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 1, 2])])  
    data = pd.DataFrame()   

    # Downloading quote from MOEX and draw plotly Dash picture
    async with aiohttp.ClientSession() as session:
        try:    
            data = await aiomoex.get_market_candles(session, Index, interval = 24, start = str(begin) , end = str(last), market = "index")
        except Exception as e:
            print ("An error is: ", str(e)) 
        
        di = pd.DataFrame(data)
        del di['value']; del di['end']; del di['low']; del di['high']; del di['volume']
        di = di[['begin', 'close']]
        di['begin'] = pd.to_datetime(di['begin'])

        # Calculating perentage of the Index
        for j in range(0, di.shape[0], 1):
            ind_pr.append((di.loc[j]['close']/di.loc[0]['close'] -1.0) * 100.0)    
    
        legendaIndex = ind_pr[len(ind_pr)-1]  # get last value
        legendaAlgo  = col[len(col) - 1]  # get last value
        legendaQuote = df.loc[df.last_valid_index()]['ChangeT0']# get last value   
        # print(legendaIndex, legendaAlgo, legendaQuote)
        # web
        global fig
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = di['begin'], y = ind_pr, mode='lines', name= "IMOEX: " + "{:6.1f}".format(legendaIndex), line=dict(dash='dash', color='gray', width=1)))
        fig.add_trace(go.Scatter(x = df['Time'], y = col, mode='lines', name= "Profit: " + "{:6.1f}".format(legendaAlgo), line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x = df['Time'], y = df['ChangeT0'], mode='lines', name= Ticker + ": " + "{:6.1f}".format(legendaQuote), line=dict(color='blue', width=2)))
        fig.update_layout(
            title={
                'text': Ticker,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})        
                           
    # Download complete
    iDwnld = 98
    return fig

#if __name__ == "__main__":
#    app.run_server(debug=True)
# PUBLISH LOCAL NETWORK
if __name__ == '__main__':
    app.run_server(debug=False,port=8080,host= '0.0.0.0')    
 
