from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import tweepy

from .tweet import Tweet


def init_tweepy_stream(api: tweepy.API) -> tweepy.StreamListener:
    listener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=listener)
    return stream


class StreamListener(tweepy.StreamListener):
    delay = 1

    def __init__(self) -> None:
        super().__init__()
        self.latest_tweet: Tweet = None

    def on_data(self, raw_data: str) -> Tweet:
        data = json.loads(raw_data)
        if self.check_data(data):
            self.latest_tweet = Tweet(data)
        return True

    def on_status_(self, status) -> Tweet:
        if self.check_data(status._json):
            self.latest_tweet = Tweet(status._json)
        return True

    def check_data(self, data: dict) -> bool:
        if data.get('limit') is not None:
            # rate limiting
            time.sleep(self.delay)
            self.delay *= 2
            return False
        self.delay = 1
        if data.get('id') is None:  # not a tweet
            return False
        return True


class CollectionStreamListener(tweepy.StreamListener):
    def __init__(self, path: Path, flush_at: int = 50000, print_at: int = 100) -> None:
        super().__init__()
        self.path = path
        self.DATA = []
        self.pbar = tqdm()
        self.flush_at = flush_at
        self.print_at = print_at

    def on_status(self, status) -> bool:
        data = status._json
        if data.get('id') is None:
            return True
        tweet = Tweet(data)
        if not tweet.retweeted:
            self.DATA.append(tweet)
            self.pbar.update(1)
            if len(self.DATA) % self.print_at == 0:
                tqdm.write(str(tweet))
            if len(self.DATA) >= self.flush_at:
                self.save_df()
        return True

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
        self.DATA = []
