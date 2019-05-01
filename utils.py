import nltk
import pandas as pd
import spacy
from collections import namedtuple, Counter
from gensim.models.phrases import Phrases
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pymagnitude import *

VECS = Magnitude(MagnitudeUtils.download_model('word2vec/medium/GoogleNews-vectors-negative300'))
STOPWORDS = set(stopwords.words('english'))
SPACY_TAGGER = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


def dict2obj(d):
    return namedtuple('Struct', d.keys())(*d.values())


def args2obj(**kwargs):
    return dict2obj(kwargs)


pos_whitelist = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]
lemmatizer = WordNetLemmatizer()


class Term:
    def __init__(self, literal):
        self.forms = Counter()
        self.forms[literal] += 1
        self.count = 1

    def add_instance(self, literal):
        self.forms[literal] += 1
        self.count += 1

    def unique_forms(self):
        uniques = []
        for (form, count) in self.forms.items():
            uniques.append(form)
        return uniques

    def common_form(self):
        return self.forms.most_common(1)[0][0]


class TermBook:
    def __init__(self):
        self.terms = {}
        self.all_term_count = 0

    def add_term(self, lemma, literal):
        if lemma in self.terms:
            self.terms[lemma].add_instance(literal)
        else:
            self.terms[lemma] = Term(literal)
        self.all_term_count += 1

    def common_term(self, lemma):
        if lemma in self.terms:
            return self.terms[lemma].common_form()
        else:
            return lemma


def choose_most_common_pos(pos_counts):
    POS_Count = namedtuple("POS_Count", ["pos", "count"])

    if not pos_counts:
        return None
    elif len(pos_counts) == 1:
        return POS_Count(*pos_counts.popitem()).pos
    else:
        first, second = [POS_Count(*pos_count) for
                         pos_count in pos_counts.most_common(2)]
        if first.count > second.count:
            return first.pos
        else:
            return None


def most_common_pos(word):
    pos_counts = Counter()
    for pos in pos_whitelist:
        root = wordnet.morphy(word, pos)
        if root:
            pos_counts[pos] = sum(
                [lemma.count() for lemma in wordnet.lemmas(root, pos)])

        return choose_most_common_pos(pos_counts)


def lemmatize(word, lemma_map):
    if word in lemma_map:
        return lemma_map[word]

    pos = most_common_pos(word)
    lemma = wordnet.morphy(word, pos)
    if lemma:
        lemma_map[word] = lemma
        return lemma
    else:
        lemma_map[word] = word
        return word


def tokenize(data):
    tokenized_data = []
    tokenizer = nltk.tokenize.RegexpTokenizer("\w+'\w+|[\w]+").tokenize
    apostrophe = nltk.tokenize.RegexpTokenizer("'[sS]$", gaps=True).tokenize
    docs_tokenized = 0
    for document in data:
        tokenized_data.append([apostrophe(token)[0] for token in tokenizer(document.lower())])
        docs_tokenized += 1
    return tokenized_data


def transform_ngrams(documents):
    transformed = documents
    for _ in range(1):
        transformed = Phrases(transformed, delimiter=b" ")[transformed]
    return transformed


def process_documents(verbatims):
    lemma_map = {}
    terms = TermBook()
    verbatims_processed = 0
    tokenized_verbatims = tokenize(verbatims)
    lemmatized_docs = []

    for verbatim in tokenized_verbatims:
        lemmatized_doc = []
        for token in verbatim:
            lemma = lemmatize(token, lemma_map)
            terms.add_term(lemma, token)
            lemmatized_doc.append(lemma)

        lemmatized_docs.append(lemmatized_doc)
        verbatims_processed += 1

    return lemmatized_docs, terms


def learn_topics(lemmatized_input):
    def get_embeddings(vocab, *args, **kwargs):
        embs = VECS.query(list(vocab.keys()))
        return pd.DataFrame(dict(zip(vocab.keys(), embs)))

    emb_type_dict = {
        'en': 'word2vec',
    }

    lang = 'en'
    params = {
        'mode': 'train',
        'domain': 'autism_speech_analysis',
        'epochs': 5,
        'seed': 3,
        'lang': lang,
        'emb_type': emb_type_dict[lang]
    }

    model_args = aspect_extraction.utils.ModelArguments(params)
    aspect_extraction.utils.print_args(model_args)
    aspect_extraction.extract_ngrams(model_args, lemmatized_input)
    aspect_extraction.continue_train_and_save_embeddings(model_args,
                                                         vocab_emb_callback=get_embeddings)
    aspect_extraction.train(model_args, word_embedding_callback=None)
    ranked_results = aspect_extraction.rank_topics(model_args,
                                                   stem_emb_callback=get_embeddings,
                                                   save_output=False)

    return ranked_results
