import sys
import numpy as np
import pandas as pd
import ipywidgets as widgets
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ipywidgets import interact
from scipy.stats import norm
from plotly.subplots import make_subplots
from sklearn.gaussian_process.kernels import RBF

display(sys.version)


def hide_code_in_slideshow():   
    from IPython import display
    import binascii
    import os
    uid = binascii.hexlify(os.urandom(8)).decode()    
    html = """<div id="%s"></div>
    <script type="text/javascript">
        $(function(){
            var p = $("#%s");
            if (p.length==0) return;
            while (!p.hasClass("cell")) {
                p=p.parent();
                if (p.prop("tagName") =="body") return;
            }
            var cell = p;
            cell.find(".input").addClass("hide-in-slideshow")
        });
    </script>""" % (uid, uid)
    display.display_html(html, raw=True)


# note that this helper function does three different things:
# (i) plots the observed data;
# (ii) plots the predictions from the learned GP after conditioning on data;
# (iii) plots samples from the GP prior (with no conditioning on observed data)

def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):

    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'kx')
    if plot_predictions:
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    plt.xlim(-0.5, 5.5)
    

def fig_rbf_kernel(only_trace: bool=True):
    from sklearn.gaussian_process.kernels import RBF
    kernel = RBF(length_scale=0.5)
    x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    x_exp = np.expand_dims(x, axis=1)
    surf_data = kernel(x_exp, x_exp)
    trace = go.Surface(x=x, y=x, z=surf_data, showscale=False)
    if only_trace:
        return trace
    else:
        fig = go.Figure([trace])
        fig.update_layout(scene = dict(xaxis_title='xi',
                                       yaxis_title='xj',
                                       zaxis_title='RBF Kernel Value'),
                                       margin=dict(r=10, b=10, l=10, t=10))
        return fig


def fig_dotprod_kernel(only_trace: bool=True):
    from sklearn.gaussian_process.kernels import DotProduct
    kernel = DotProduct()
    x = np.linspace(-1, 1, 100)
    x_exp = np.expand_dims(x, axis=1)
    surf_data = kernel(x_exp, x_exp)
    trace = go.Surface(x=x, y=x, z=surf_data, showscale=False)
    if only_trace:
        return trace
    else:
        fig.update_layout(scene = dict(xaxis_title='xi',
                                       yaxis_title='xj',
                                       zaxis_title='Linear Kernel Value'),
                                       margin=dict(r=10, b=10, l=10, t=10))
        return fig


def fig_matern_kernel(only_trace: bool=True):
    from sklearn.gaussian_process.kernels import Matern
    kernel = Matern(length_scale=2, nu=15)
    x = np.linspace(-5, 5, 100)
    x_exp = np.expand_dims(x, axis=1)
    surf_data = kernel(x_exp, x_exp)
    trace = go.Surface(x=x, y=x, z=surf_data, showscale=False)
    if only_trace:
        return trace
    else:
        fig.update_layout(scene = dict(xaxis_title='xi',
                                       yaxis_title='xj',
                                       zaxis_title='Matern Kernel Value'),
                                       margin=dict(r=10, b=10, l=10, t=10))
        return fig


def fig_sin_kernel(only_trace: bool=True):
    from sklearn.gaussian_process.kernels import ExpSineSquared
    kernel = ExpSineSquared()
    x = np.linspace(-1, 1, 100)
    x_exp = np.expand_dims(x, axis=1)
    surf_data = kernel(x_exp, x_exp)
    trace = go.Surface(x=x, y=x, z=surf_data, showscale=False)
    if only_trace:
        return trace
    else:
        fig.update_layout(scene = dict(xaxis_title='xi',
                                       yaxis_title='xj',
                                       zaxis_title='Linear Kernel Value'),
                                       margin=dict(r=10, b=10, l=10, t=10))
        return fig


def plot_kernels():
    fig = make_subplots(rows=2, cols=2, #column_widths=[0.2, 0.8], row_heights=[0.5, 0.5], 
                        horizontal_spacing = 0.0,
                        vertical_spacing = 0.1,
                        specs=[[{"type": "scene"}, {"type": "scene"}],
                               [{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=('RBF Kernel', 'DotProd Kernel', 'ExpSineSquared Kernel', 'Matern Kernel'))

    fig.add_trace(fig_rbf_kernel(), row=1, col=1)
    fig.add_trace(fig_dotprod_kernel(), row=1, col=2)
    fig.add_trace(fig_sin_kernel(), row=2, col=1)
    fig.add_trace(fig_matern_kernel(), row=2, col=2)

    fig.update_layout(height=700,  
                      margin=dict(l=20, r=20, t=20, b=20),
                      xaxis=dict(title='x1',  # '$\Large x_1$',
                                 showgrid=True),
                      yaxis=dict(title='x2',  # '$\Large x_2$',
                                 showgrid=True))
    return fig

    
def plot_multivariate(mu: np.ndarray, sigma: np.ndarray, x1: float):
    # Compute conditional expectation and variance
    exp_x2_cond_x1 = mu[1] + sigma[1,0] * (1.0 / sigma[0,0]) * (x1 - mu[0])
    var_x2_cond_x1 = sigma[1,1] - sigma[1,0] * (1.0 / sigma[0,0]) * sigma[0,1]
    assert var_x2_cond_x1 > 0.0  # Test: variance can only be positive

    # Plot ranges
    x1_min = mu[0] - 3*sigma[0,0]
    x1_max = mu[0] + 3*sigma[0,0]
    x2_min = mu[1] - 3*sigma[1,1]
    x2_max = mu[1] + 3*sigma[1,1]

    # Plot grid 
    N = 100
    x1_arr = np.linspace(x1_min, x1_max, N)
    x2_arr = np.linspace(x2_min, x2_max, N)
    x1_grid, x2_grid = np.meshgrid(x1_arr, x2_arr)

    pos = np.empty(x1_grid.shape + (2,))
    pos[:, :, 0] = x1_grid
    pos[:, :, 1] = x2_grid

    def multivariate_gaussian(x, mu, sigma):
        """Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/"""
        n = mu.shape[0]
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2*np.pi)**n * sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, sigma_inv, x-mu)
        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    z = multivariate_gaussian(pos, mu, sigma)

    x2_min_plot = x2_min+0.5
    x2_max_plot = x2_max-0.5
    x2_arr_cond = np.linspace(x2_min_plot, x2_max_plot, 100)
    #x2_arr_cond = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
    x2_cond_pdf = norm.pdf(x2_arr_cond, loc=exp_x2_cond_x1, scale=var_x2_cond_x1)

    x2_cond_pdf_plot = x2_cond_pdf + x1

    # Plotly plot
    fig = go.Figure(data = [go.Contour(x=x1_arr,
                                       y=x2_arr,
                                       z=z,
                                       showlegend=False,
                                       showscale=False
                                   ),
                            go.Scatter(x=[x1, x1], 
                                       y=[x2_min_plot, x2_max_plot],
                                       mode='lines',
                                       name=f'Conditional value x1={x1}',
                                       line=dict(color='rgb(179,205,227)',
                                                 width=2),
                                       showlegend=True),
                            go.Scatter(x=x2_cond_pdf_plot, 
                                       y=x2_arr_cond,
                                       mode='lines',
                                       name=f'PDF, $f(X_2|X_1={x1}) \sim \mathcal{{N}}({exp_x2_cond_x1}, {var_x2_cond_x1})$',
                                       # name=f'PDF, f(X2|X1={x1}) = N({exp_x2_cond_x1}, {var_x2_cond_x1})',
                                       line=dict(color='rgb(204,235,197)',
                                                 width=2),
                                       showlegend=True)
        ])

    fig.update_layout(title_text="Conditional Distribution of 2D Normal Distribution",
                      title_font_size=30,
                      xaxis=dict(title='x1',  # '$\Large x_1$',
                                 showgrid=True),
                      yaxis=dict(title='x2',  # '$\Large x_2$',
                                 showgrid=True))
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig
    
    
def get_plot_cov_and_samples(length_scale: float = 0.1, num_samples: int = 10, number_of_variables: int = 50):
    # Define kernel
    kernel = RBF(length_scale=length_scale)

    # Generate covariance matrix and mean vector
    x = np.linspace(0.0, 1.0, number_of_variables)
    x = np.expand_dims(x, axis=1)
    sigma = kernel(x, x)
    mu = np.zeros(sigma.shape[0])

    # Sample multivariate the normal process (defined by mean and covariance)
    samples = np.random.multivariate_normal(mu, sigma, size=num_samples)
    samples = samples.transpose()
    samples_df = pd.DataFrame(samples, 
                              columns=[f'Sample func. {i}' for i in range(samples.shape[1])],
                              index=x.squeeze())

    # Plot covariance matrix as heatmap
    fig = make_subplots(rows=2, cols=2, column_widths=[0.2, 0.8], row_heights=[0.5, 0.5], 
                        specs=[[{"type": "heatmap"}, {"type": "scatter", "rowspan": 2}],
                               [None, None]],
                        subplot_titles=('Covariance Matrix', 'Function samples'))

    fig.add_trace(go.Heatmap(z=sigma, showlegend=False, showscale=False, colorscale='Greys', reversescale=True), row=1, col=1)
    fig.update_layout(xaxis=dict(title='$\mathbf{x}$'),
                      yaxis=dict(title='$\mathbf{x}$',
                                 autorange='reversed'))

    # Plot sample functions
    for col_name in samples_df:
        col = samples_df[col_name]
        trace = go.Scatter(x=col.index.values, 
                           y=col.values, 
                           name=col_name, 
                           mode='markers+lines', 
                           showlegend=True,
                           line=dict(width=1.0),
                           marker=dict(size=3))
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(xaxis2=dict(title='$\mathbf{x}$'),
                      yaxis2=dict(title='$f(\mathbf{x})$'))    
    
    # Define dragmode, newshape parameters, amd add modebar buttons
    fig.update_layout(dragmode='drawrect',
                      newshape=dict(line_color='cyan'),
        )
    return fig


def get_dash_app_prior_demo():
    # Build Dash Application for Demo
    import dash_core_components as dcc
    import dash_html_components as html
    from jupyter_dash import JupyterDash
    from dash.dependencies import Input, Output

    # Add support for latex in Dash
    MATHJAX_CDN = '''https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'''
    external_scripts = [
                        {'type': 'text/javascript',
                         'id': 'MathJax-script',
                         'src': MATHJAX_CDN,
                         },
                        ]
    app = JupyterDash(__name__, external_scripts=external_scripts)

    app.layout = html.Div([
        dcc.Graph(id='graph',
#                 config={'modeBarButtonsToAdd':['drawline',
#                                                'drawopenpath',
#                                                'drawclosedpath',
#                                                'drawcircle',
#                                                'drawrect',
#                                                'eraseshape'
#                                               ]}
                 ),
        html.Label([
            "Number of samples",
            dcc.Slider(
                id='slider-sample-number',
                min=1,
                max=100,
                step=1,
                marks={i: str(i) for i in [1,10,20,30,40,50,60,70,80,90,100]},
                value=10,
            ),        
        ]),
        html.Label([
            "Length scale",
            dcc.Slider(
                id='slider-length-scale',
                min=0.01,
                max=1.0,
                step=0.01,
                marks={(i): str(i) for i in [0.01, 0.05, 0.1, 0.5, 1.0]},
                value=0.1,
            ),        
        ]),
        html.Label([
            "Number of variables",
            dcc.Slider(
                id='number-of-variables',
                min=2,
                max=100,
                step=1,
                marks={i: str(i) for i in [2,10,20,30,40,50,60,70,80,90,100]},
                value=50,
            ),        
        ]),
    ])

    # Define callback to update graph
    @app.callback(
        Output('graph', 'figure'),
        [Input("slider-sample-number", "value"),
         Input("slider-length-scale", "value"),
         Input("number-of-variables", "value"),
        ]
    )
    def update_figure(num_samples, length_scale, number_of_variables):
        fig = get_plot_cov_and_samples(length_scale=length_scale, 
                                       num_samples=num_samples, 
                                       number_of_variables=number_of_variables)
        return fig
    
    return app


def get_dash_app_prior_demo_ipywidgets():
    layout = widgets.Layout(width='auto')  # set slider label width
    disp = interact(lambda num_samples,length_scale,number_of_variables: 
                         get_plot_cov_and_samples(length_scale=length_scale, 
                                                  num_samples=num_samples, 
                                                  number_of_variables=number_of_variables), 
                     num_samples=widgets.IntSlider(min=1, 
                                                   max=100, 
                                                   step=10, 
                                                   value=10, 
                                                   description='Number of samples',
                                                   layout=layout,
                                                   style = {'description_width': 'initial'}),
                     length_scale=widgets.FloatLogSlider(base=10, 
                                                         min=-2, 
                                                         max=0, 
                                                         step=0.2, 
                                                         value=0.2, 
                                                         description='Length scale', 
                                                         readout_format='.2f',
                                                         layout=layout,
                                                         style = {'description_width': 'initial'}),
                     number_of_variables=widgets.IntSlider(min=2, 
                                                           max=100, 
                                                           step=10, 
                                                           value=20, 
                                                           description='Number of variables',
                                                           layout=layout,
                                                           style = {'description_width': 'initial'}))


def plot_gaussian_process(mean: np.ndarray, cov: np.ndarray, x1: np.ndarray, f1: np.ndarray, x_plot: np.ndarray):
    std = np.sqrt(np.diag(cov))
    
    num_samples = 5

    # Sample multivariate the normal process (defined by mean and covariance)
    samples = np.random.multivariate_normal(mean.squeeze(), cov, size=num_samples)
    samples = samples.transpose()
    samples_df = pd.DataFrame(samples, 
                              columns=[f'Sample func. {i}' for i in range(samples.shape[1])],
                              index=x_plot.squeeze())

    # Plot covariance matrix as heatmap
    fig = make_subplots(rows=2, cols=2, column_widths=[0.2, 0.8], row_heights=[0.5, 0.5], 
                        specs=[[{"type": "heatmap"}, {"type": "scatter", "rowspan": 2}],
                               [None, None]],
                        subplot_titles=('Covariance Matrix', 'Function samples'))
    fig.add_trace(go.Heatmap(z=cov, showlegend=False, showscale=False, colorscale='Greys', reversescale=True), row=1, col=1)
    fig.update_layout(xaxis=dict(title='$\mathbf{x}$'),
                      yaxis=dict(title='$\mathbf{x}$',
                                 autorange='reversed'))

    # Plot sample functions
    for col_name in samples_df:
        col = samples_df[col_name]
        trace = go.Scatter(x=col.index.values, 
                           y=col.values, 
                           name=col_name, 
                           mode='markers+lines', 
                           showlegend=True,
                           line=dict(width=1.0),
                           marker=dict(size=3))
        fig.add_trace(trace, row=1, col=2)
    fig.update_layout(xaxis2=dict(title='$\mathbf{x}$'),
                      yaxis2=dict(title='$f(\mathbf{x})$'))

    # Plot mean and standard deviation band
    y_upper = mean.squeeze() + 2*std  # 2 sigma range
    y_lower = mean.squeeze() - 2*std  # 2 sigma range
    # Plot mean
    fig.add_trace(go.Scatter(
        x=x_plot.squeeze(),
        y=mean.squeeze(),
        name="Mean"
    ), row=1, col=2)
    # Plot std band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_plot.squeeze(), x_plot.squeeze()[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        hoveron='points',
        name="Std. Deviation Band",
        line=dict(width=0.5,
                  color='mediumseagreen'),
        opacity=0.4
    ), row=1, col=2)

    # Plot train datapoints
    trace = go.Scatter(x=x1.squeeze(), 
                       y=f1.squeeze(), 
                       name='Train Datapoints', 
                       mode='markers', 
                       showlegend=True,
                       line=dict(width=2.0),
                       marker=dict(#symbol='x-thin',
                                   size=8,
                                   color='black'))
    fig.add_trace(trace, row=1, col=2)

    # Define dragmode, newshape parameters, amd add modebar buttons
    fig.update_layout(dragmode='drawrect',
                      newshape=dict(line_color='cyan'))
    return fig
