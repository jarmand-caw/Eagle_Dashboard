import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import dash_table
import dash_table.FormatTemplate as FormatTemplate
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_auth
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

warnings.simplefilter('ignore')


def find_target_rate(power_capacity, growth_curve='conservative'):
    assert growth_curve in ['standard', 'conservative'], 'Invalid growth curve option'
    if growth_curve == 'conservative':
        if power_capacity < 2200:
            return 0.49
        if power_capacity > 20000:
            return 0.4
        else:
            return 0.5011 - (.09 / 17800) * power_capacity
    if growth_curve == 'standard':
        if power_capacity < 2200:
            return 0.65
        if power_capacity > 20000:
            return 0.51
        else:
            return 0.6673 - (.14 / 17800) * power_capacity


class DataCenter:
    def __init__(self, power_capacity, sqft, estimate_percent=True, year_5_percent=None, growth_curve='conservative',
                 starting_capacity=0, offset=0, utility_kw_hr_rate=0.13, rent_per_sqft=31, usage_rate=0.7, monthly_revenue_per_kw=160):
        if estimate_percent == True:
            assert growth_curve in ['conservative', 'standard'], 'Invalid growth curve choice'
            self.goal_percent = find_target_rate(power_capacity, growth_curve)
        else:
            assert year_5_percent is not None, 'Must enter year_5_percent if using estimate_percent=False'
            self.goal_percent = year_5_percent

        self.offset = offset

        self.power_capacity = power_capacity
        self.square_footage = sqft
        self.starting_capacity = starting_capacity
        self.estimate_percent = estimate_percent

        # Revenue
        self.monthly_rate_per_cabinet = 600
        self.monthly_kw_per_cabinet = 5

        self.cabinet_capacity = self.power_capacity / self.monthly_kw_per_cabinet

        self.monthly_per_kw_rate = monthly_revenue_per_kw
        self.monthly_cross_connect_per_cab = 250
        self.revenue_per_install = 2500
        self.professional_percent_colo = 0.02

        # Costs
        self.rent_per_sqft = rent_per_sqft
        self.sqft_per_kw = 9.6
        self.utilities_kw_hr_rate = utility_kw_hr_rate
        self.hours_in_quarter = 24 * 365 / 4
        self.usage_percent = usage_rate
        self.monthly_maintenance = 1000
        self.monthly_rm = 2000
        self.property_tax_percent_of_rent = 0.011
        self.cross_connect_monthly_cost = 50
        self.conduit_monthly_cost = 50
        self.sga_minimum = 167490

    def log(self):
        if self.starting_capacity > 0:
            if self.estimate_percent == True:
                sum = 0
                for x in np.arange(1, 22):
                    sum += np.log(x)
                coef = (self.goal_percent - 21 * 8 / 600) / sum
                self.coef = coef
                news = []
                for x in np.arange(1, 22):
                    news.append(self.coef * np.log(x) + 8 / 600)
                cum_percent = 0
                for i in range(len(news)):
                    cum_percent += news[i]
                    if cum_percent > self.starting_capacity:
                        break
                shift_x = i
                final_news = []
                for x in np.arange(1, 22):
                    final_news.append(self.coef * np.log(x + shift_x) + 8 / 600)
                self.quarterly_new_percents = np.array(final_news)
            elif self.estimate_percent == False:
                sum = 0
                for x in np.arange(1, 22):
                    sum += np.log(x)
                goal_percent = self.goal_percent - self.starting_capacity
                coef = (goal_percent - 21 * 8 / 600) / sum
                self.coef = coef
                news = []
                for x in np.arange(1, 22):
                    news.append(self.coef * np.log(x) + 8 / 600)
                self.quarterly_new_percents = np.array(news)
        else:
            sum = 0
            for x in np.arange(1, 22):
                sum += np.log(x)
            goal_percent = self.goal_percent - self.starting_capacity
            coef = (goal_percent - 21 * 8 / 600) / sum
            self.coef = coef
            news = []
            for x in np.arange(1, 22):
                news.append(self.coef * np.log(x) + 8 / 600)
            self.quarterly_new_percents = np.array(news)
        self.quarterly_new_cabinets = self.quarterly_new_percents * self.cabinet_capacity
        return self.quarterly_new_percents

    def quarterly_effective(self):
        change_cabs = self.quarterly_new_cabinets
        print(change_cabs)
        for x in [5, 9, 13, 17]:
            change_cabs[x] = change_cabs[x] - 1
        print(change_cabs)
        cum = np.cumsum(change_cabs)
        skip = np.cumsum(change_cabs)
        skip = list(skip)
        skip.insert(0, 0)
        skip = skip[:-1]
        skip = np.array(skip)
        qe = (skip + cum) / 2
        qe = np.array([round(x, 0) for x in list(qe)])
        if self.starting_capacity > 0:
            qe = qe + self.cabinet_capacity * self.starting_capacity
        self.quarterly_effective_cabinets = qe
        return self.quarterly_effective_cabinets

    def get_financial_information(self):
        # all revenues quarterly
        self.space_revenue = self.monthly_rate_per_cabinet * 3 * self.quarterly_effective_cabinets
        self.power_revenue = self.monthly_kw_per_cabinet * self.monthly_per_kw_rate * 3 * self.quarterly_effective_cabinets
        self.cross_connect_revenue = self.monthly_cross_connect_per_cab * 3 * self.quarterly_effective_cabinets
        self.install_revenue = self.revenue_per_install * np.array(
            [round(x, 0) for x in list(self.quarterly_new_cabinets)])
        self.professional_service_revenue = self.professional_percent_colo * self.space_revenue
        self.total_revenue = self.space_revenue + self.power_revenue + self.cross_connect_revenue + self.install_revenue + \
                             self.professional_service_revenue

        # all costs quarterly
        self.rent_cost = self.square_footage * self.rent_per_sqft / 4
        self.utilities = self.utilities_kw_hr_rate * self.hours_in_quarter * self.usage_percent * self.quarterly_effective_cabinets
        self.maintenance = self.monthly_maintenance * 3
        self.rm = self.monthly_rm * 3
        self.property_tax = self.property_tax_percent_of_rent * self.rent_cost
        self.cross_connect_expense = self.cross_connect_monthly_cost * 3 * self.quarterly_effective_cabinets
        self.conduit_expense = self.conduit_monthly_cost * 3 * self.quarterly_effective_cabinets
        self.total_direct_costs = self.rent_cost + self.utilities + self.maintenance + self.rm + self.property_tax + self.cross_connect_expense + \
                                  self.conduit_expense

        self.gross_profit = self.total_revenue - self.total_direct_costs
        opex = []
        for x in self.total_revenue:
            if x * 0.2 > self.sga_minimum:
                opex.append(x * 0.2)
            else:
                opex.append(self.sga_minimum)
        self.operating_expenses = np.array(opex)
        self.ebitda = self.gross_profit - self.operating_expenses

        self.df = pd.DataFrame({
            'New Cabinets': [round(x, 0) for x in list(self.quarterly_new_cabinets)],
            'Quarterly Effective Cabinets': list(self.quarterly_effective_cabinets),
            'Total Revenue': list(self.total_revenue),
            'Total Costs': list(self.total_direct_costs),
            'OPEX': list(self.operating_expenses),
            'EBITDA': list(self.ebitda)
        }).transpose()
        return self.df

    def offset_for_addition(self):
        offset = self.offset
        a = np.zeros([len(self.df), 21])
        projected = self.df.values
        projected = projected[:, :21 - offset]
        a[:, offset:] = projected
        self.offset_array = a
        return self.offset_array

    def get_all_for_addition(self):
        self.log()
        self.quarterly_effective()
        self.get_financial_information()
        self.offset_for_addition()
        return self.offset_array


dc1 = DataCenter(3000, 27871, estimate_percent=True, starting_capacity=0)
dc1.log()
dc1.quarterly_effective()
dc1.get_financial_information()
pd.DataFrame(dc1.offset_for_addition())
final_df = dc1.get_financial_information()

VALID_USERNAME_PASSWORD_PAIRS = {
    'HawkWoodGroup': 'project_eagle'
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server
quarters = []
for x in range(20):
    if x % 4 == 0:
        quarter = 'Q1'
    if x % 4 == 1:
        quarter = 'Q2'
    if x % 4 == 2:
        quarter = 'Q3'
    if x % 4 == 3:
        quarter = 'Q4'
    year = str(x // 4 + 2021)
    quarters.append(year + '-' + quarter)
quarters.insert(0, '2020-Q4')

final_df.columns = quarters

quarter_dropdown_options = []
for x in range(len(quarters)):
    quarter_dropdown_options.append({"label": quarters[x], "value": x})
quarter_dropdown_options.insert(0,{"label":'None selected', "value":None})

data_centers = dict(zip(list(np.arange(1,11)),[None for x in range(10)]))
data_centers[1] = dc1



def create_input_group(number, disabled,visible):
    a = html.Div([
        dbc.Card([
        dbc.CardHeader('Datacenter ' + str(number + 1)),
        dbc.Col([
            dbc.Container([
                dbc.FormGroup([
                    dbc.Label('Select Quarter'),
                    dbc.Select(options=quarter_dropdown_options, value=None, disabled=disabled, id='select_'+str(number))
                ])
            ])
        ]),
        dbc.Col([
            dbc.Container([
                dbc.FormGroup([
                    dbc.Label('Power Capacity (kw)'),
                    dbc.Input(
                        placeholder='3000',
                        type='number',
                        id='power_capacity_' + str(number),
                        persistence=True,
                        persistence_type='session',
                        disabled=disabled,
                        value=3000
                    )
                ])
            ])
        ]),
        dbc.Col([
            dbc.Container([
                dbc.FormGroup([
                    dbc.Label('Square Footage'),
                    dbc.Input(
                        placeholder='27,871',
                        type='number',
                        id='sqft_' + str(number),
                        persistence=True,
                        persistence_type='session',
                        disabled=disabled,
                        value=27971
                    )
                ])
            ])
        ]),
        dbc.Col([
            dbc.Container([
                dbc.FormGroup([
                    dbc.Label('Percent of Capacity Filled Currently'),
                    dbc.Input(
                        placeholder='0%',
                        type='number',
                        id='current_percent_' + str(number),
                        persistence=True,
                        persistence_type='session',
                        disabled=disabled,
                        value=0
                    )
                ])
            ])
        ]),
        dbc.Col([
            dbc.Container([
                dbc.FormGroup([
                    dbc.Label('Growth Curve'),
                    dbc.RadioItems(
                        options=[
                            {"label": "Conservative", "value": 'conservative', "disabled": disabled},
                            {"label": "Standard", "value": 'standard', "disabled": disabled}
                        ],
                        value='conservative',
                        id='growth_curve_' + str(number),
                        persistence=True,
                        persistence_type='session',
                    )
                ])
            ])
        ]),
        dbc.Col([
            dbc.Container([
                dbc.Button('Advanced Options', color='secondary', disabled=disabled, block=True, id='advanced_'+str(number))
            ]),
            dbc.Collapse([
                dbc.Col([
                    dbc.Container([
                        dbc.FormGroup([
                            dbc.Label('Utility cost per kw/hr'),
                            dbc.Input(
                                placeholder='$0.13',
                                type='number',
                                id='utility_' + str(number),
                                persistence=True,
                                persistence_type='session',
                                disabled=disabled,
                                value=0.13
                            )
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Container([
                        dbc.FormGroup([
                            dbc.Label('Rent per sqft'),
                            dbc.Input(
                                placeholder='31',
                                type='number',
                                id='rent_' + str(number),
                                persistence=True,
                                persistence_type='session',
                                disabled=disabled,
                                value=31
                            )
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Container([
                        dbc.FormGroup([
                            dbc.Label('Usage Rate'),
                            dbc.Input(
                                placeholder='0.7',
                                type='number',
                                id='usage_' + str(number),
                                persistence=True,
                                persistence_type='session',
                                disabled=disabled,
                                value=0.7
                            )
                        ])
                    ])
                ]),
                dbc.Col([
                    dbc.Container([
                        dbc.FormGroup([
                            dbc.Label('Monthly revenue per kw'),
                            dbc.Input(
                                placeholder='160',
                                type='number',
                                id='kw_revenue_' + str(number),
                                persistence=True,
                                persistence_type='session',
                                disabled=disabled,
                                value=160
                            )
                        ])
                    ])
                ]),
            ], id='collapse_'+str(number))
        ]),
        dbc.Col([
            dbc.Container([
                dbc.Button('OK', color='primary', disabled=disabled, block=True, id='submit_'+str(number))
            ])
        ]),
        dbc.Col([
            dbc.Container([
                dbc.Button('Remove', color='warning', disabled=disabled, block=True, id='remove_' + str(number))
            ])
        ])
    ])
    ], style={'display':visible}, id='div'+str(i))
    return a


visible_list = ['block'] * 2 + ['none'] * 8
disabled_list = [False]*1+[True]*9

inputs = []
for i, (x, y) in enumerate(zip(visible_list, disabled_list)):
    inputs.append(create_input_group(i, disabled=y, visible=x))

def create_outputs_change_disabled(number):
    outputs = []
    ids = ['select', 'power_capacity', 'sqft', 'current_percent', 'utility', 'rent', 'usage',
           'kw_revenue','advanced','submit']
    new_ids = []
    for x in ids:
        new_ids.append(x+'_'+str(number))
    for x in new_ids:
        outputs.append(Output(x, 'disabled'))
    outputs.append(Output('growth_curve_'+str(number), 'options'))
    return outputs

def create_states(number):
    states = []
    ids = ['select', 'power_capacity', 'sqft', 'current_percent', 'utility', 'rent', 'usage',
           'kw_revenue','growth_curve']
    new_ids = []
    for x in ids:
        new_ids.append(x+'_'+str(number))
    for x in new_ids:
        states.append(State(x, 'value'))
    return states

def create_outputs_change_visibility(number):
    outputs = []
    name = 'div'+str(number)
    outputs.append(Output(name, 'style'))
    return outputs

app.layout = dbc.Tabs([
    dbc.Tab([
        dbc.Container([
            dbc.Row([dbc.Col([html.H1('Welcome')])], justify='center'),
            dbc.Row([dbc.Col([html.H1('Project Eagle Dashboard')])], justify='center'),
        ])
    ], label='Welcome', tab_id='welcome'),
    dbc.Tab([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2('Enter Projected Data Centers'),
                    html.Div([],id='card_tracking', style={'display':'none'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5(
                        'You can key metrics and add data centers here. For more advanced options, please see the advanced tab')
                ])
            ]),
            dbc.Row(dbc.Col(dbc.Button('Submit', id='big_sub', block=True, color='primary'))),
            dbc.Row([dbc.Col(dbc.Button('Add Data Center', id='plus', block=True, color='success'))]),
            dbc.Row(html.P('')),
            dbc.Row(
                inputs, id='input_row')
        ]),
        html.Div(children=final_df.to_json(),id='intermediate-value', style={'display': 'none'})
    ], label='Input', tab_id='input'),
    dbc.Tab([
        dbc.Row([dbc.Col([dbc.Container([], id='proj')])])
    ], label='Projected Financials', tab_id='proj')
], id='tabs', active_tab='input')

all_states = []
for x in range(10):
    all_states = all_states+create_states(x)
big_sub_input = [Input('big_sub', 'n_clicks')]

all_vis_outputs = []
for x in range(10):
    all_vis_outputs+=create_outputs_change_visibility(x)

#Change the graph

@app.callback(Output('card_tracking', 'children'),
              [Input('plus', 'n_clicks')],
              [State('card_tracking', 'children')])
def change(n,current):
    if n:
        current = list(current)
        current.sort()
        all = list(np.arange(10))
        avail = all-current
        next = avail[0]
        current.append(next)
        return current
    else:
        return current

@app.callback(all_vis_outputs,
              [Input('card_tracking', 'children')],
              [State('input_row', 'children')])

def update_visibility(tracking, current_inp):
    for idx in tracking:
        current_inp[idx] = create_input_group(idx,False,'block')
    return current_inp

@app.callback(Output('intermediate-value', 'children'),
              big_sub_input,
              all_states)
def update_df(*args):
    input_names = [item.component_id for item in big_sub_input + all_states]
    kwargs = dict(zip(input_names, args))
    array = np.zeros([6,21])
    count = 0
    for x in range(10):
        name = 'select_'+str(x)
        if (kwargs[name] is not None) & (kwargs[name]!='None selected'):
            print(kwargs[name])
            find = [i for i in input_names if str(x) in i]
            find.remove(name)
            dc = DataCenter(kwargs[find[0]], kwargs[find[1]], growth_curve=kwargs[find[-1]], starting_capacity=kwargs[find[2]],
                            offset=int(kwargs[name]), utility_kw_hr_rate=kwargs[find[3]], rent_per_sqft= kwargs[find[4]],
                            usage_rate=kwargs[find[5]], monthly_revenue_per_kw=kwargs[find[6]])
            a = dc.get_all_for_addition()
            array = array+a
            count+=1
    if count>0:
        df = pd.DataFrame(array, columns=quarters, index=list(dc.df.index))
        json = df.to_json()
        return json
    else:
        return final_df.to_json()
@app.callback(Output('proj', 'children'),
              [Input('intermediate-value','children')]
)
def run_graph(json):
    current_df = pd.read_json(json, convert_dates=False, convert_axes=False)
    current_df = current_df.round(0)
    #ind = list(current_df.index)[2:]

    dt = dash_table.DataTable(
        id='table',
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        columns = [{"name":i, "id":i} for i in list(current_df.reset_index().columns)[:1]]+[{"name":i, "id":i, "type":'numeric', 'format':FormatTemplate.money(0)} for i in list(current_df.reset_index().columns)[1:]],
        data = current_df.reset_index().to_dict(orient='records'),

    )
    return dt

@app.callback(
    Output("collapse_0", "is_open"),
    [Input("advanced_0", "n_clicks")],
    [State("collapse_0", "is_open")],
)
def toggle_collapse(n,is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse_1", "is_open"),
    [Input("advanced_1", "n_clicks")],
    [State("collapse_1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse_2", "is_open"),
    [Input("advanced_2", "n_clicks")],
    [State("collapse_2", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("collapse_3", "is_open"),
    [Input("advanced_3", "n_clicks")],
    [State("collapse_3", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(
    Output("collapse_4", "is_open"),
    [Input("advanced_4", "n_clicks")],
    [State("collapse_4", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(
    Output("collapse_5", "is_open"),
    [Input("advanced_5", "n_clicks")],
    [State("collapse_5", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(
    Output("collapse_6", "is_open"),
    [Input("advanced_6", "n_clicks")],
    [State("collapse_6", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(
    Output("collapse_7", "is_open"),
    [Input("advanced_7", "n_clicks")],
    [State("collapse_7", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(
    Output("collapse_8", "is_open"),
    [Input("advanced_8", "n_clicks")],
    [State("collapse_8", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


@app.callback(
    Output("collapse_9", "is_open"),
    [Input("advanced_9", "n_clicks")],
    [State("collapse_9", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open

    return is_open


#Remove button submit reset
@app.callback(Output('submit_0', 'n_clicks'),[Input('remove_0', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_1', 'n_clicks'),[Input('remove_1', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_2', 'n_clicks'),[Input('remove_2', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_3', 'n_clicks'),[Input('remove_3', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_4', 'n_clicks'),[Input('remove_4', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_5', 'n_clicks'),[Input('remove_5', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_6', 'n_clicks'),[Input('remove_6', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_7', 'n_clicks'),[Input('remove_7', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_8', 'n_clicks'),[Input('remove_8', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None
@app.callback(Output('submit_9', 'n_clicks'),[Input('remove_9', 'n_clicks')])
def reset_remove_clicks(n):
    if n:
        return None
    else:
        return None

#Submit button add more entries
@app.callback(create_outputs_change_disabled(1)+create_outputs_change_visibility(2)+[Output('remove_0', 'disabled')],
              [Input('submit_0','n_clicks'),Input('remove_0','n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(2)+create_outputs_change_visibility(3)+[Output('remove_1', 'disabled')],
              [Input('submit_1','n_clicks'),Input('remove_1', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(3)+create_outputs_change_visibility(4)+[Output('remove_2', 'disabled')],
              [Input('submit_2','n_clicks'),Input('remove_2', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l

@app.callback(create_outputs_change_disabled(4)+create_outputs_change_visibility(5)+[Output('remove_3', 'disabled')],
              [Input('submit_3','n_clicks'),Input('remove_3', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(5)+create_outputs_change_visibility(6)+[Output('remove_4', 'disabled')],
              [Input('submit_4','n_clicks'),Input('remove_4', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(6)+create_outputs_change_visibility(7)+[Output('remove_5', 'disabled')],
              [Input('submit_5','n_clicks'),Input('remove_5', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(7)+create_outputs_change_visibility(8)+[Output('remove_6', 'disabled')],
              [Input('submit_6','n_clicks'),Input('remove_6', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(8)+create_outputs_change_visibility(9)+[Output('remove_7', 'disabled')],
              [Input('submit_7','n_clicks'),Input('remove_7', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[{'display':'block'}]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[{'display':'none'}]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [{'display': 'none'}]
        l = l + [True]
        return l
@app.callback(create_outputs_change_disabled(9)+[Output('remove_8', 'disabled')],
              [Input('submit_8','n_clicks'),Input('remove_8', 'n_clicks')])
def incremental_change(n1,n2):
    if n1:
        l = [False]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": False},
            {"label": "Standard", "value": 'standard', "disabled": False}
        ]
        l = l+[options]
        l = l+[False]
        return l
    elif n2:
        l = [True]*10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l+[options]
        l = l+[True]
        return l
    else:
        l = [True] * 10
        options = [
            {"label": "Conservative", "value": 'conservative', "disabled": True},
            {"label": "Standard", "value": 'standard', "disabled": True}
        ]
        l = l + [options]
        l = l + [True]
        return l

#Submit button add DC objects


if __name__ == '__main__':
    app.run_server(debug=True)
