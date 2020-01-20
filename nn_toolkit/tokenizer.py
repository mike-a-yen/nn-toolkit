from typing import List

from nltk.tokenize import word_tokenize



def check_nltk_download():
    try:
        word_tokenize('checking download')
    except LookupError:
        import nltk  # pylint: disable=import-outside-toplevel
        nltk.download('punkt')


class SimpleTokenizer:
    def __init__(self) -> None:
        check_nltk_download()
        self.tokenizer = word_tokenize
    
    def __call__(self, text: str) -> List[str]:
        text = self.preprocessing(text)
        tokens = self.tokenizer(text)
        tokens = self.postprocessing(tokens)
        return tokens
    
    def preprocessing(self, text: str) -> str:
        return text.lower()
    
    def postprocessing(self, tokens: List[str]) -> List[str]:
        return tokens
