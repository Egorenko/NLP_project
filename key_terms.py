import nltk
from lxml import etree
from nltk.corpus import stopwords
import string
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


lemma = WordNetLemmatizer()
xml_path = 'news.xml'
tree = etree.parse(xml_path)
root = tree.getroot()
news = root[0]
titles = []
all_words = []
for new in news:
    title = new[0]
    titles.append(title.text)
    text = new[1]
    words = nltk.tokenize.word_tokenize(text.text.lower())
    words = [lemma.lemmatize(i) for i in words]
    words = [i for i in words if i not in stopwords.words('english') and i not in list(string.punctuation)]
    pos_tag_words = []
    for word in words:
        pos_tag_word = nltk.pos_tag([word])
        if pos_tag_word[0][1] == 'NN':
            pos_tag_words.append(pos_tag_word[0][0])
    all_words.append(' '.join(pos_tag_words))
vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True,
                             analyzer='word', ngram_range=(1, 1), stop_words=None,
                             vocabulary=None, min_df=0.01, max_df=0.60)
tfidf_matrix = vectorizer.fit_transform(all_words)
terms = vectorizer.get_feature_names_out()
tfidf_matrix = tfidf_matrix.toarray()
shapes = tfidf_matrix.shape
row = 0
while row < shapes[0]:
    print(titles[row], end=':\n')
    column = 0
    word_rate = []
    while column < shapes[1]:
        if tfidf_matrix[row][column] != 0:
            word_rate.append((terms[column], tfidf_matrix[row][column]))
        column += 1
    sorted_words = sorted(word_rate, key=lambda x: (x[1], x[0]), reverse=True)[:5]
    sorted_words = [i[0] for i in sorted_words]
    print(*sorted_words, sep=' ')
    row += 1
