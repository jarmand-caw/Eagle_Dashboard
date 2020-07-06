import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_auth
import numpy as np
import warnings
warnings.simplefilter('ignore')

def PERT(low,mode,high,size):
    return np.random.triangular(low,mode,high,size)

stats = {
    'adressable_market':130000,
    'base_line_gc':np.array([.01,.05,.15,.45,.75,1]),
    'initial_fixed_cost':750,
    'upfront_investment':7500
}

class Distribution:
    """
    Note: Stats is a list. Important that items are placed in correct order, otherwise its possible an error will be
    thrown and results will certainly be off
    Note: Order is always: low, mode, high or mean, std
    """
    def __init__(self, name, stats):
        if name=='triangular':
            assert len(stats)==3, 'Wrong number of arguments for triangular dist'
            self.name = name
            self.lower = stats[0]
            self.mode = stats[1]
            self.upper = stats[2]
        elif name=='pert':
            assert len(stats) == 3, 'Wrong number of arguments for pert dist'
            self.name = name
            self.lower = stats[0]
            self.mode = stats[1]
            self.upper = stats[2]
        elif name=='normal':
            assert len(stats)==2, 'Wrong number of arguments for normal dist'
            self.name = name
            self.mean = stats[0]
            self.std = stats[1]
        elif name=='uniform':
            assert len(stats)==2, 'Wrong number of arguments for uniform dist'
            self.name = name
            self.lower = stats[0]
            self.upper = stats[1]
    def get_list(self):
        if self.name=='triangular' or self.name=='pert':
            return [self.lower,self.mode,self.upper]
        elif self.name=='normal':
            return [self.mean,self.std]
        elif self.name=='uniform':
            return [self.lower,self.upper]
    def generate(self, size):
        if self.name == 'triangular':
            return np.random.triangular(*self.get_list(), size)
        elif self.name == 'pert':
            return PERT(*self.get_list(),size)
        elif self.name == 'normal':
            return np.random.normal(*self.get_list(), size)
        elif self.name == 'uniform':
            return np.random.uniform(*self.get_list(), size)

#step1: simulate year 5 customers
distribution_dict = {
    'mkt_growth':['pert',[.01,.02,.03]],
    'mkt_takeup':['pert',[.4,.5,.6]],
    'share':['pert',[.2,.3,.6]],
    'growth_factor':['triangular',[.9,1,1.1]],
    'ref_rates':['pert',[.25,.4,.65]],
    'churn_rates':['pert',[.3,.5,.7]],
    'margins':['pert',[3.5,4.5,5.5]],
    'aquisition_expenses':['normal',[1.75,.15]],
    'aquisition_capital':['normal',[1.75,.15]],
    'fixed_cost_increases':['pert',[.06,.10,.16]]
}
def create_dist_dict_from_list(list_of_values):
    d = {
    'mkt_growth':['pert',list_of_values[:3]],
    'mkt_takeup':['pert',list_of_values[3:6]],
    'share':['pert',list_of_values[6:9]],
    'growth_factor':['triangular',list_of_values[9:12]],
    'ref_rates':['pert',list_of_values[12:15]],
    'churn_rates':['pert',list_of_values[15:18]],
    'margins':['pert',list_of_values[18:21]],
    'aquisition_expenses':['normal',list_of_values[21:23]],
    'aquisition_capital':['normal',list_of_values[23:25]],
    'fixed_cost_increases':['pert',list_of_values[25:28]]
    }
    return d
list_of_values = [.01,.02,.03, .4,.5,.6, .2,.3,.6, .9,1,1.1, .25,.4,.65, .3,.5,.7, 3.5,4.5,5.5, 1.75,.15, 1.75,.15, .06,.10,.16]
distribution_dict = create_dist_dict_from_list(list_of_values)
dists = []
for statistics in list(distribution_dict.values()):
    dists.append(Distribution(*statistics))

manager_estimates = dict(zip(list(distribution_dict.keys()),dists))

class MonteCarlo:
    def __init__(self):
        pass
    def simulate_npv(self, manager_estimates, stats):
        target_market_size = stats['adressable_market']

        mkt_growths = manager_estimates['mkt_growth'].generate(6)

        mkt_growths[0] = ((1 + mkt_growths[0]) ** .5) - 1

        target_market = target_market_size
        for year in range(6):
            target_market = target_market * (1 + mkt_growths[year])

        mkt_takeup = manager_estimates['mkt_takeup'].generate(1)[0]

        share = manager_estimates['share'].generate(1)[0]

        customers = target_market * mkt_takeup * share

        baseline = stats['base_line_gc']
        factor = manager_estimates['growth_factor'].generate(1)[0]

        curve = baseline*factor

        curve[5] = 1

        net_per_year = curve*customers

        yearly_ref_rates = manager_estimates['ref_rates'].generate(5)

        yearly_churn_rates = manager_estimates['churn_rates'].generate(6)

        yearly_churns = [] #len should be 6 at end
        yearly_refs = [] #len should be 6 at end
        yearly_refs.append(0)
        for idx in range(5):
            yearly_refs.append(yearly_ref_rates[idx]*net_per_year[idx])

        yearly_churns.append(-1*(yearly_churn_rates[0] / 2) * net_per_year[0])
        for idx in range(5):
            yearly_churns.append(-1*yearly_churn_rates[idx+1]*net_per_year[idx])

        yearly_churns = np.array(yearly_churns)
        yearly_refs = np.array(yearly_refs)
        prev_year = list(net_per_year[:-1])
        prev_year.insert(0,0)
        net_per_year = np.array(net_per_year)
        prev_year = np.array(prev_year)

        gross_additions = net_per_year-prev_year-yearly_churns-yearly_refs

        margins = manager_estimates['margins'].generate(6)
        margins[0] = margins[0]/2

        units_for_margin = list(net_per_year)
        units_for_margin.insert(0,gross_additions[0])
        gross_margins = []
        for i in range(6):
            avg_u = (units_for_margin[i]+units_for_margin[i+1])/2
            gross_margins.append(avg_u*margins[i])

        aq_exps = manager_estimates['aquisition_expenses'].generate(6)

        total_aq_exps = aq_exps*gross_additions

        aq_caps = manager_estimates['aquisition_capital'].generate(6)

        total_aq_caps = aq_caps * gross_additions

        oi = np.array(gross_margins)-total_aq_exps
        self.oi = oi
        self.total_aq_exps = total_aq_exps
        self.gross_margins = np.array(gross_margins)

        taxes = oi*.2

        fcf_part1 = oi-taxes-total_aq_caps

        fixed_cost_increases = manager_estimates['fixed_cost_increases'].generate(5)
        fixed_costs = []
        f1 = stats['initial_fixed_cost']
        for inc in fixed_cost_increases:
            f1 = f1*(1+inc)
            fixed_costs.append(f1)
        fixed_costs.insert(0,stats['initial_fixed_cost'])

        fcf = fcf_part1-np.array(fixed_costs)
        self.fcf = fcf

        npv = 0
        for i in range(6):
            npv+=fcf[i]/((1.2)**i)
        npv = npv-7500

        cumulative_investment = []
        running_total = 0
        for idx in range(6):
            one = fcf[idx]
            running_total+=one
            cumulative_investment.append(running_total)
        cumulative_investment = np.array(cumulative_investment)

        required_investment = cumulative_investment.min()

        total_investment = required_investment-stats['upfront_investment']
        self.total_investment = total_investment

        self.multiple = -1*(fcf[-1]/self.total_investment)
        return npv

def run_simulation(manager_estimates):
    npvs = []
    fcfs = []
    ois = []
    exps = []
    margins = []
    total_investment = []
    multiples = []

    mc1 = MonteCarlo()
    for i in range(10000):
        npv = mc1.simulate_npv(manager_estimates, stats)
        fcf = mc1.fcf
        oi = mc1.oi
        ti = mc1.total_investment
        margin = mc1.gross_margins
        aq_exp = mc1.total_aq_exps
        mult = mc1.multiple

        multiples.append(mult)
        total_investment.append(ti)
        margins.append(margin)
        exps.append(aq_exp)
        ois.append(oi)
        npvs.append(npv)
        fcfs.append(fcf)
    npvs = np.array(npvs)
    print('Mean:',npvs.mean())
    print('Std:',npvs.std())

    columns = ['year0','year1','year2','year3','year4','year5']
    df = pd.DataFrame(np.array(fcfs))
    df.columns = columns
    df['npv'] = npvs

    df = df.sort_values('npv')
    bottom_25 = df[columns].iloc[2499]
    median = df[columns].iloc[4999]
    top_75 = df[columns].iloc[7499]

    multiples = np.array(multiples)
    mult = multiples.mean()
    mult_color="danger"
    if mult>3:
        mult_color="success"
    mult =  'Average: '+str(round(mult,2))
    multiples = list(multiples)
    multiples.sort()
    def locate_mult3(multiples):
        for i,x in enumerate(multiples):
            if x<3:
                continue
            else:
                if i == 0:
                    return 0
                else:
                    return i - 1
    idx = locate_mult3(multiples)
    percent_above_3 = (idx/len(multiples))*100
    multiple_string = str(round(100-percent_above_3,4))+'%'
    multiple_string = 'Percent over 3.0: '+multiple_string

    investment = np.array(total_investment)
    invest = investment.mean()
    invest_color = "danger"

    if invest>-13500:
        invest_color="success"

    invest = 'Average: '+str(round(-1*invest,0))
    investment = list(investment)
    investment.sort()

    def locate_over_maxinv(investment):
        for i,x in enumerate(investment):
            if x<-13500:
                continue
            else:
                if i == 0:
                    return 0
                else:
                    return i - 1
    idx = locate_over_maxinv(investment)
    percent_over = (idx/len(investment))*100
    investment_string = str(round(100-percent_over,4))+'%'
    investment_string = 'Percent under 13.5M: '+investment_string


    fcf_year3 = np.array(fcfs)[:,3]
    year3 = fcf_year3.mean()
    year3_color = "danger"
    if year3>0:
        year3_color="success"
    year3 = 'Average: ' + str(round(year3,0))

    fcf_year3 = list(fcf_year3)
    fcf_year3.sort()

    def locate_neg(fcfs):
        for i,x in enumerate(fcfs):
            if x <0:
                continue
            else:
                if i == 0:
                    return 0
                else:
                    return i - 1
    idx = locate_neg(fcf_year3)
    percent_under = (idx/len(fcf_year3))*100
    year3_string = str(round(100-percent_under,4))+'%'
    year3_string = 'Percent positive: ' + year3_string


    npv_mean = np.array(npvs).mean()
    npv_color="danger"
    if npv_mean>0:
        npv_color="success"
    npv_mean = 'Average: '+str(round(npv_mean,0))
    npvs.sort()
    def locate_zero(npvs):
        for i,x in enumerate(npvs):
            if x<0:
                continue
            else:
                if i==0:
                    return 0
                else:
                    return i-1
    idx = locate_zero(npvs)
    percent_above_zero = (1-(idx/len(npvs)))*100
    npv_string = str(round(percent_above_zero,4))+'%'
    npv_string = 'Percent positive: ' + npv_string

    columns = ['year0', 'year1', 'year2', 'year3', 'year4', 'year5']

    hist = go.Figure(go.Histogram(x=npvs))
    hist.update_layout(title='Simulated NPV Histogram')
    hist.update_xaxes(title='NPV')

    scatter = go.Figure(go.Scatter(x=columns, y=median, name='Median NPV'))
    scatter.add_trace(go.Scatter(x=columns, y=bottom_25, name='25% NPV'))
    scatter.add_trace(go.Scatter(x=columns, y=top_75, name='75% NPV'))
    scatter.update_layout(title='FCF Over Time for NPV Quartiles')
    scatter.update_yaxes(title='FCF')
    return mult, multiple_string, invest, investment_string, year3, year3_string, npv_mean, npv_string, hist, scatter, mult_color, invest_color, year3_color, npv_color

mult, multiple_string, invest, investment_string, year3, year3_string, npv_mean, npv_string, hist, scatter, mult_color, invest_color, year3_color, npv_color = run_simulation(manager_estimates)

VALID_USERNAME_PASSWORD_PAIRS = {
    '1all': 'jbk'
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server


def input_component(label, id, value, step):
    components = dbc.FormGroup([
            dbc.Label(label, width=5,align='center'),
        dbc.Col(
            dbc.Input(
            type='number',
            id = id,
            value = value,
            max=1e10,
            min=-1e10,
            step=step,
            persistence=True,
            persistence_type='session'
    ), width=7)], row=True)
    return components
app.layout = dbc.Tabs([
    dbc.Tab([
    dbc.Container([
        dbc.Row([html.H1('Monte Carlo Simulation: 1All Media')], justify='center'),
        dbc.Row([
            dbc.Col([
                   dbc.Card(
                        id="first-card",
                        children=[
                        dbc.CardHeader(['Yearly Market Growth'],style={'fontWeight':'bold'}),
                        dbc.Container([
                                    input_component('min','mkt_growth_min',distribution_dict['mkt_growth'][1][0],.0001),
                                    input_component('mode','mkt_growth_mode',distribution_dict['mkt_growth'][1][1],.0001),
                                    input_component('max','mkt_growth_max',distribution_dict['mkt_growth'][1][2],.0001)
                                   ])])
                ]),
            dbc.Col([
                    dbc.Card([
                            dbc.CardHeader(['Year 5 Market Takeup'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'mkt_takeup_min', distribution_dict['mkt_takeup'][1][0], .001),
                            input_component('mode', 'mkt_takeup_mode', distribution_dict['mkt_takeup'][1][1], .001),
                            input_component('max', 'mkt_takeup_max', distribution_dict['mkt_takeup'][1][2], .001)
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                            dbc.CardHeader(['Year 5 1All Mkt Share'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'mkt_share_min', distribution_dict['share'][1][0], .001),
                            input_component('mode', 'mkt_share_mode', distribution_dict['share'][1][1], .001),
                            input_component('max', 'mkt_share_max', distribution_dict['share'][1][2], .001)
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                            dbc.CardHeader(['Growth Factor'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'gf_min', distribution_dict['growth_factor'][1][0], .001),
                            input_component('mode', 'gf_mode', distribution_dict['growth_factor'][1][1], .001),
                            input_component('max', 'gf_max', distribution_dict['growth_factor'][1][2], .001)
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                            dbc.CardHeader(['Referral Rate'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'ref_rate_min', distribution_dict['ref_rates'][1][0], .001),
                            input_component('mode', 'ref_rate_mode', distribution_dict['ref_rates'][1][1], .001),
                            input_component('max', 'ref_rate_max', distribution_dict['ref_rates'][1][2], .001)
                        ])])
                ]),
            ],justify='center'),
        dbc.Row([
            dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(['Churn Rate'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'churn_rate_min', distribution_dict['churn_rates'][1][0], .001),
                            input_component('mode', 'churn_rate_mode', distribution_dict['churn_rates'][1][1], .001),
                            input_component('max', 'churn_rate_max', distribution_dict['churn_rates'][1][2], .001)
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(['Average Margin per User'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'margin_min', distribution_dict['margins'][1][0], .001),
                            input_component('mode', 'margin_mode', distribution_dict['margins'][1][1], .001),
                            input_component('max', 'margin_max', distribution_dict['margins'][1][2], .001)
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(['Aquisition Expenses'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('mean', 'aq_exp_mean', distribution_dict['aquisition_expenses'][1][0], .001),
                            input_component('std', 'aq_exp_std', distribution_dict['aquisition_expenses'][1][1], .001),
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(['Aquisition Capital'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('mean', 'aq_cap_mean', distribution_dict['aquisition_capital'][1][0], .001),
                            input_component('std', 'aq_cap_std', distribution_dict['aquisition_capital'][1][1], .001),
                        ])])
                ]),
            dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(['Fixed Cost Increase'],style={'fontWeight':'bold'}),
                        dbc.Container([
                            input_component('min', 'fc_min', distribution_dict['fixed_cost_increases'][1][0], .001),
                            input_component('mode', 'fc_mode', distribution_dict['fixed_cost_increases'][1][1], .001),
                            input_component('max', 'fc_max', distribution_dict['fixed_cost_increases'][1][2], .001)
                        ])])
                ]),
            ]),
        dbc.Row([
            dbc.Col([
                            dbc.Button(
                                "Submit",
                                block=True,
                                id="submit_button",
                            ),
                    ])]),
    ])],label='Input', tab_id='tab-1'),
    dbc.Tab([
        dbc.Row([
            html.H1('Results')
        ],justify='center'),
        dbc.Row([
            html.H3('All NPV, Investment and FCF numbers in thousands (000)')
        ],justify='center'),
        dbc.Container([
            dbc.Row([
                dbc.Col(children=[
                    dcc.Graph(
                        id="hist",
                        animate=False,
                        config={'responsive':True},
                        figure=hist
                    )
                ]),
                dbc.Col(children=[
                    dcc.Graph(
                        id="scatter",
                        animate=False,
                        figure=scatter
                    )
                ])
            ])]),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card(id="mult_card",children=[
                dbc.CardBody([
                        html.H3("Five Year Multiple of Investment",  style={'fontWeight':'bold'}),
                        html.H5(id='mult', children=[
                           mult]
                        ),
                        html.H5(id='multiple_string', children=[
                           multiple_string]
                        )
                    ])
                ],color=mult_color, inverse=True)
            ]),
            dbc.Col([
                dbc.Card(id="npv_card", children=[
                    dbc.CardBody(
                        [
                            html.H3("Average NPV",  style={'fontWeight':'bold'}),
                            html.H5(id='npv_mean',children=[
                                npv_mean]
                            ),
                            html.H5(id='npv_string',children=[
                                npv_string]
                            )
                        ])
                    ], color=npv_color, inverse=True),
                ]),
            dbc.Col([
                dbc.Card(id="invest_card", children=[
                    dbc.CardBody(
                        [
                            html.H3("Total Investment", style={'fontWeight':'bold'}),
                            html.H5(id='invest', children=[
                                invest]
                            ),
                            html.H5(id='investment_string', children=[
                                investment_string]
                            )
                        ])
                    ],color=invest_color, inverse=True)
            ]),
            dbc.Col([
                dbc.Card(id="year3_card", children=[
                    dbc.CardBody(
                        [
                            html.H3("Year 3 FCF", style={'fontWeight': 'bold'}),
                            html.H5(id = 'year3', children=[
                                year3]
                            ),
                            html.H5(id = 'year3_string', children=[
                                year3_string]
                            )
                        ])
                ], color=year3_color, inverse=True),
            ]),
        ])])

    ], label='Output', tab_id='tab-2')], id='tabs', active_tab='tab-1')


@app.callback(
    [
        Output('mult', 'children'),
        Output('multiple_string', 'children'),
        Output('invest', 'children'),
        Output('investment_string', 'children'),
        Output('year3', 'children'),
        Output('year3_string', 'children'),
        Output('npv_mean', 'children'),
        Output('npv_string', 'children'),
        Output('hist', 'figure'),
        Output('scatter', 'figure'),
        Output('tabs','active_tab'),
        Output('mult_card','color'),
        Output('invest_card','color'),
        Output('year3_card','color'),
        Output('npv_card','color')
    ],
              [Input('submit_button', 'n_clicks')],
              [
                  State('mkt_growth_min', 'value'),
                  State('mkt_growth_mode', 'value'),
                  State('mkt_growth_max', 'value'),
                  State('mkt_takeup_min', 'value'),
                  State('mkt_takeup_mode', 'value'),
                  State('mkt_takeup_max', 'value'),
                  State('mkt_share_min', 'value'),
                  State('mkt_share_mode', 'value'),
                  State('mkt_share_max', 'value'),
                  State('gf_min', 'value'),
                  State('gf_mode', 'value'),
                  State('gf_max', 'value'),
                  State('ref_rate_min', 'value'),
                  State('ref_rate_mode', 'value'),
                  State('ref_rate_max', 'value'),
                  State('churn_rate_min', 'value'),
                  State('churn_rate_mode', 'value'),
                  State('churn_rate_max', 'value'),
                  State('margin_min', 'value'),
                  State('margin_mode', 'value'),
                  State('margin_max', 'value'),
                  State('aq_exp_mean', 'value'),
                  State('aq_exp_std', 'value'),
                  State('aq_cap_mean', 'value'),
                  State('aq_cap_std', 'value'),
                  State('fc_min', 'value'),
                  State('fc_mode', 'value'),
                  State('fc_max', 'value')
              ]
            )
def update_output(xx,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb):
    list_of_values = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb]
    distribution_dict = create_dist_dict_from_list(list_of_values)
    print(distribution_dict)

    dists = []
    for statistics in list(distribution_dict.values()):
        dists.append(Distribution(*statistics))

    manager_estimates = dict(zip(list(distribution_dict.keys()), dists))

    mult, multiple_string, invest, investment_string, year3, year3_string, npv_mean, npv_string, hist, scatter, mult_color, invest_color, year3_color, npv_color = run_simulation(manager_estimates)
    return mult, multiple_string, invest, investment_string, year3, year3_string, npv_mean, npv_string, hist, scatter, 'tab-2', mult_color, invest_color, year3_color, npv_color

if __name__ == '__main__':
    app.run_server(debug=True)