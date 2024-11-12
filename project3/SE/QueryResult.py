#%%
import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords


from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def tagToPos(tag):
    if 'JJ' in tag:
        return 'a'
    elif 'NN' in tag:
        return 'n'
    elif 'VB' in tag:
        return 'v'
    elif 'RB' in tag:
        return 'r'
    else:
        return None

def importantTag(tag):
    if 'JJ' in tag:
        return True
    elif 'NN' in tag:
        return True
    elif 'VB' in tag:
        return True
    elif 'RB' in tag:
        if tag == 'WRB':
            return False
        return True
    return False



def getSearchEngineResult(query_dict):
    result_dict = {}
    ix = index.open_dir("./index")
    retokenize = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    #with ix.searcher(weighting=scoring.BM25F()) as searcher:
    with ix.searcher(weighting=scoring.ScoringFunction(K1=3, B=0.7, eps=0)) as searcher:

        # TODO - Define your own query parser
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.9))

        stopWords = set(stopwords.words('english'))
        stopWords.update(['patients', 'treatment', 'p', 'clinical', 'study', 'patient', 'results'])
        stopWords.update(['case','associated', 'diagnosis', 'may', 'methods', 'group', 'diseases'])
        stopWords.update(['year', 'years', 'cases', 'pain', 'therapy', 'background', 'two', 'age'])
        stopWords.update(['significant', 'conclusion', 'old', '95', 'one', 'care', 'management', 'common'])
        stopWords.update(['hospital', 'non', 'af', 'factors', 'also', 'present', 'use'])
        stopWords.update(['compared', 'used', 'however', 'severe', 'performed', 'rate', 'data', 'using',])
        stopWords.update(['report', 'studies', 'rare', 'showed', 'days', 'conclusions', 'significantly', 'cause', 'higher', 'analysis'])
        stopWords.update(['without', 'due', 'well', 'reported', 'presented', 'history', 'mean', 'treated', 'time'])
        stopWords.update(['failure', 'outcome', 'type', 'based', 'presentation', 'n', 'medical', 'total', 'groups'])
        stopWords.update(['review', 'outcomes', 'including', 'findings', 'among', 'small', 'related', 'found', 'positive', 'revealed'])
        stopWords.update(['months', 'day', 'low', 'injury', 'included', 'lower', 'vs', 'primary', 'incidence'])
        stopWords.update(['weeks', 'follow', 'normal', 'within', 'new', 'term', 'underwent'])
        stopWords.update(['period', 'population', 'test', 'although', 'effects', 'function', 'right', 'score'])
        stopWords.update(['literature', 'major', 'events', 'multiple', 'rates', 'condition', 'number', 'considered'])

      

        for qid, q in query_dict.items():
            new_q = ''

            sentence = q.lower()
            tagged_list = pos_tag(retokenize.tokenize(sentence))
            
            for (word, tag) in tagged_list:
                if not importantTag(tag):
                    continue
                if word in stopWords:
                    continue
                word = lemmatizer.lemmatize(word, pos=tagToPos(tag))
                if '-' in word:
                    for w in word.split('-'):
                        new_q += w + ' '
                else: new_q += word + ' '
            query = parser.parse(new_q.lower())

            results = searcher.search(query, limit=None)
            result_dict[qid] = [result.fields()['docID'] for result in results]

    return result_dict



#%%




