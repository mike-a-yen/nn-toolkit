from datetime import datetime

_TWITTER_DATE_FORMAT = '%a %b %d %H:%M:%S %z %Y'


class Tweet:
    """Object for storing twitter data."""

    def __init__(self, data: dict) -> None:
        self.text = data.get('text', data.get('full_text'))
        self.id = data['id']
        self.created_at = self.convert_date(data['created_at'])
        self.lang = data['lang']
        self.keyword = None

        self.retweeted = data['retweeted'] or self.text.startswith('RT @')
        self.truncated = data['truncated']

        self.set_user(data['user'])
        self.set_coordinates(data['coordinates'])
        self.set_place(data['place'])
        self.geo = data['geo']

    def set_user(self, user: dict) -> None:
        self.user_id = user['id']
        self.user_name = user['name']
        self.user_screen_name = user['screen_name']
        self.user_location = user['location']

    def set_coordinates(self, coordinates: dict) -> None:
        if coordinates is None:
            self.lat = None
            self.long = None
        else:
            self.lat = coordinates.get('lat')
            self.long = coordinates.get('long')
    
    def set_place(self, place: dict) -> None:
        if place is None:
            self.country = None
            self.country_code = None
            self.location = None
        else:
            self.country = place.get('country')
            self.country_code = place.get('country_code')
            self.location = place.get('full_name')

    def convert_date(self, date_str: str) -> datetime:
        return datetime.strptime(date_str, _TWITTER_DATE_FORMAT)

    def dump(self):
        return self.__dict__

    def __repr__(self) -> str:
        return f'>>> {self.text}'
