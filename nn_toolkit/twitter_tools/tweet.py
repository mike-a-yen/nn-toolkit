class Tweet:
    """Object for storing twitter data."""

    def __init__(self, data: dict) -> None:
        self.text = data.get('text', data.get('full_text'))
        self.id = data['id']
        self.lang = data['lang']
        self.keyword = None
        self.retweeted = data['retweeted'] or self.text.startswith('RT @')
        self.truncated = data['truncated']
        self.user_id = data['user']['id']
        self.user_name = data['user']['name']
        self.screen_nmae = data['user']['screen_name']
        place = data.get('place')
        if place is None:
            place = dict()
        self.country = place.get('country')
        self.country_code = place.get('country_code')
        self.location = place.get('full_name')

    def dump(self):
        return self.__dict__

    def __repr__(self) -> str:
        return f'>>> {self.text}'
