from typing import List
import re

# Ukrainian stopwords (core set — extend as needed)
UK_STOPWORDS = {
    "і", "в", "у", "на", "з", "із", "зі", "до", "за", "від", "по",
    "для", "при", "про", "що", "як", "це", "той", "та", "але", "або",
    "чи", "не", "ні", "так", "він", "вона", "воно", "вони", "ми",
    "ви", "я", "ти", "його", "її", "їх", "мій", "наш", "ваш", "свій",
    "цей", "ця", "цього", "тому", "також", "ще", "вже", "якщо", "коли",
    "де", "є", "був", "була", "було", "були", "бути", "може", "треба",
    "можна", "потрібно", "після", "перед", "між", "через", "під", "над",
    "без", "лише", "тільки", "ж", "б", "би", "хоча", "проте", "однак",
    "себе", "собі", "нас", "вас", "їм", "нам", "вам", "мене", "тебе",
}

# Tokenizer regex: preserves Ukrainian apostrophes, handles Latin tokens
UK_TOKEN_RE = re.compile(
    r"[а-яіїєґ][а-яіїєґ'ʼ\u2019]*"  # Cyrillic words (with apostrophe)
    r"|[a-z][a-z0-9]*",               # Latin words (medical terms)
    re.UNICODE
)

# Medical abbreviation expansion (add your domain terms here)
MEDICAL_ABBREVS = {
    "екг": ["електрокардіограма"],
    "узд": ["ультразвукове", "дослідження"],
    "мрт": ["магнітно", "резонансна", "томографія"],
    "ат":  ["артеріальний", "тиск"],
    "чсс": ["частота", "серцевих", "скорочень"],
    "лфк": ["лікувальна", "фізична", "культура"],
}


class UkrainianPreprocessor:
    """Tokenize, remove stopwords, and lemmatize Ukrainian text for BM25."""

    def __init__(self, use_lemmatization: bool = True, expand_abbrevs: bool = True):
        self.use_lemmatization = use_lemmatization
        self.expand_abbrevs = expand_abbrevs
        self._morph = None

    @property
    def morph(self):
        if self._morph is None:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer(lang='uk')
        return self._morph

    def __call__(self, text: str) -> List[str]:
        text = text.lower()
        tokens = UK_TOKEN_RE.findall(text)
        tokens = [t for t in tokens if t not in UK_STOPWORDS and len(t) > 1]

        # Expand medical abbreviations
        if self.expand_abbrevs:
            expanded = []
            for t in tokens:
                if t in MEDICAL_ABBREVS:
                    expanded.extend(MEDICAL_ABBREVS[t])
                expanded.append(t)
            tokens = expanded

        # Lemmatize
        if self.use_lemmatization:
            tokens = [self.morph.parse(t)[0].normal_form for t in tokens]

        return tokens
