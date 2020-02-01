from pathlib import Path

import toml
import tweepy


_CREDENTIALS = Path().home() / '.twitter' / 'auth.toml'


def load_credentials(credentials_path: Path = _CREDENTIALS) -> dict:
    """Load Tweepy credentials.
    
    Parameters
    ----------
    credentials_path : Path, optional
        path to credentials, by default '~/.twitter.auth.toml'
    
    Returns
    -------
    dict
    
    Raises
    ------
    OSError
        if file does not exist
    """
    if credentials_path is None:
        credentials_path = _CREDENTIALS
    if credentials_path.exists():
        with open(credentials_path) as fo:
            credentials = toml.load(fo)
        return credentials
    raise OSError(f'Could not locate {credentials_path}.')


def init_tweepy_api(credentials: dict) -> tweepy.API:
    auth = tweepy.OAuthHandler(
        credentials['api_key'],
        credentials['api_secret_key']
    )
    auth.set_access_token(
        credentials['access_token'],
        credentials['access_token_secret']
    )
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api
