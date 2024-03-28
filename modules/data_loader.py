"""
Loads data from the yfinance package
"""
import yfinance as yf

class DataLoader:

    def __init__(self, start_date, freq, train_test_split) -> None:
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
        test = data[self.train_test_split:]

        return train, test
        

    