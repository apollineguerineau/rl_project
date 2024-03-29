"""
Loads data from the yfinance package
"""
import yfinance as yf
from plotly import tools
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
class DataLoader:

    def __init__(self, start_date, freq, train_test_split, end_date) -> None:
        """Constructor

        Parameters
        ----------
        start_date: str
            Date starting which data sould be downloaded
        freq: str
            Frequence at which data should be downloaded
        train_test_split: str
            Before is training data, after is test data
        """
        self.start_date = start_date
        self.freq = freq
        self.train_test_split = train_test_split
        self.end_date = end_date

    def load(self, asset):
        """Loads data and splits into train and test
        
        Parameter
        ---------
        asset: str
            Asset name (see yfinance doc for possible values)

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            Train and test datasets
        """

        # load data
        data = yf.download([asset], 
                           start=self.start_date,
                           interval=self.freq)
        
        # split train test
        train = data[:self.train_test_split]
        test = data[self.train_test_split:self.end_date]

        return train, test

    def plot_train_test(self,asset):
        train, test = self.load(asset)
        data = [
            Candlestick(x=train.index, open=train['Open'], high=train['High'], low=train['Low'], close=train['Close'], name='train'),
            Candlestick(x=test.index, open=test['Open'], high=test['High'], low=test['Low'], close=test['Close'], name='test')
        ]
        layout = {
            'shapes': [
                {'x0': self.train_test_split, 'x1': self.train_test_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper', 'line': {'color': 'rgb(0,0,0)', 'width': 1}}
            ],
            'annotations': [
                {'x': self.train_test_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left', 'text': ' test data'},
                {'x': self.train_test_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right', 'text': 'train data '}
            ]
        }
        figure = Figure(data=data, layout=layout)
        return figure
        

    