#%%
import os.path
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC


from whoosh.analysis import StemmingAnalyzer
from nltk.corpus import stopwords

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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


schema = Schema(docID=NUMERIC(stored=True),
                contents=TEXT(analyzer=StemmingAnalyzer(stoplist=stopWords))
                )
                
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)
writer = ix.writer()

lm = WordNetLemmatizer()


with open('./doc/document.txt', 'r', encoding='UTF-8') as f:
    text = f.read()
    docs = text.split('////\n')[:-1]

    for doc in docs:
        br = doc.find('\n')
        docID = int(doc[:br])
        doc_text = doc[br+1:]

        tagged_list = pos_tag(word_tokenize(doc_text))
        new_doc_text = ''
        for (word, tag) in tagged_list:
            if not importantTag(tag):
                new_doc_text += '//////////' + ' '
                continue
            word = lm.lemmatize(word, pos=tagToPos(tag))
            if '-' in word:
                for w in word.split('-'):
                    new_doc_text += w + ' '
            else: 
                new_doc_text += word + ' '


        writer.add_document(docID=docID, contents=new_doc_text)

writer.commit()
print("make index!")

# %%
