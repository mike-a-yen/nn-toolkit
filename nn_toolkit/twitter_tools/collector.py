from datetime import datetime
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import tweepy

from .tweet import Tweet


class TweetCollector:
    """Collect tweets using the tweepy cursor.

    I found that the cursor is faster at collecting tweets than the streaming api.
    
    Parameters
    ----------
    path : Path
        path to the directory to save collected tweets. Tweets are saved in a dataframe.
    api : tweepy.API
    print_at : int
        print one of every `print_at` tweet collected, by default 1000
    flush_at : int, optional
        save every chunk of `flush_at` tweets to `path` and empty the collection, by default 50000
    """
    def __init__(self, path: Path, api: tweepy.API, flush_at: int = 50000, print_at: int = 1000) -> None:
        self.path = path
        self.api = api
        self.DATA = []
        self.pbar = tqdm()
        self.flush_at = flush_at
        self.print_at = print_at

    def search(self, term: str, count: int = 100, **kwargs) -> None:
        """Search for tweets.
        
        Parameters
        ----------
        term : str
            filter tweets by this term
        count : int, optional
            how many tweets to collect, max allowed is 100, by default 100
        """
        query = f'{term} -filter:retweets'
        cursor = tweepy.Cursor(self.api.search, q=query, lang='en', count=count, **kwargs)
        i = 0
        for data in cursor.items():
            tweet = Tweet(data._json)
            if not tweet.retweeted:
                self.DATA.append(tweet)
                self.pbar.update(1)
                i += 1
                time.sleep(0.05)
                if len(self.DATA) % self.print_at == 0:
                    tqdm.write(tweet)
                if len(self.DATA) >= self.flush_at:
                    self.save_df()
                    self.DATA = []
            if i >= count:
                break

    def to_df(self) -> pd.DataFrame:
        """Dump collected tweets to a dataframe."""
        df = pd.DataFrame([tweet.dump() for tweet in self.DATA])
        df = df.drop_duplicates(subset='text')
        return df

    def save_df(self) -> None:
        now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        filename = self.path / f'{now}.csv.gz'
        df = self.to_df()
        df.to_csv(filename, index=False)
        print('*** ', end='')
        print(f'Saved {df.shape[0]} samples to {filename}', end=' ')
        print('***')
