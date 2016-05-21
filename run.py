# coding: utf-8
from flask import Flask, request, jsonify, render_template
from flask_frozen import Freezer
import urllib.request as urllib2
import json
from datetime import datetime
import codecs
from multiprocessing import Process
from tom_lib.nlp.topic_model import NonNegativeMatrixFactorization, LatentDirichletAllocation
from tom_lib.structure.corpus import Corpus as TOM_Corpus
from tom_lib import utils
from mabed_lib.corpus import Corpus as MABED_Corpus
from mabed_lib.mabed import MABED
import os
import time
from nltk import FreqDist, word_tokenize
import re
# from unidecode import unidecode

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

web_service_app = Flask(__name__)
topic_model_browser_app = Flask(__name__, static_folder='static', template_folder='templates_tom')
event_browser_app = Flask(__name__, static_folder='static', template_folder='templates_mabed')

token = None
model = None
k = None
json_corpus = None
topic_model = None
min_tf = 3
max_tf = 0.6
lang = 'english'
theta = 0.6
sigma = 0.5
tsl = 30
mabed = None


@web_service_app.route('/detect_events/init', methods=['GET', 'POST'])
def detect_events_init():
    get_token()
    if token is not None:
        get_corpus_mabed()
        t_results = Process(target=transmit_events, args=(token, k, tsl, theta, sigma, lang))
        t_results.start()
        return jsonify({'success': 'success'}), 200


@web_service_app.route('/infer_topics/init', methods=['GET', 'POST'])
def infer_topics_init():
    get_token()
    if token is not None:
        get_corpus_tom()
        t_results = Process(target=transmit_topic_model, args=(token, model, k, min_tf, max_tf, lang))
        t_results.start()
        return jsonify({'success': 'success'}), 200


@web_service_app.route('/vocabulary/init', methods=['GET', 'POST'])
def vocabulary_init():
    get_token()
    if token is not None:
        get_corpus_mabed()
        t_results = Process(target=transmit_vocabulary, args=(token, ))
        t_results.start()
        return jsonify({'success': 'success'}), 200


def get_token():
    if request.method == 'POST':
        str_data = request.data.decode('utf-8')
        post_dict = json.loads(str_data)
        global token, k, model, min_tf, max_tf, theta, sigma, tsl, lang
        token = post_dict.get('token')
        parameters = post_dict.get('params')
        if parameters.get('k') is not None:
            k = parameters.get('k')
        if parameters.get('model') is not None:
            model = parameters.get('model')
        if parameters.get('min_tf') is not None:
            min_tf = parameters.get('min_tf')
        if parameters.get('max_tf') is not None:
            max_tf = parameters.get('max_tf')
        if parameters.get('theta') is not None:
            theta = parameters.get('theta')
        if parameters.get('sigma') is not None:
            sigma = parameters.get('sigma')
        if parameters.get('tsl') is not None:
            tsl = parameters.get('tsl')
        if parameters.get('lang') is not None:
            lang = parameters.get('lang')
    elif request.method == 'GET':
        return 'Operation not allowed'


def get_corpus_tom():
    print('http://mediamining.univ-lyon2.fr:8080/cats/api/', token)
    corpus_request = urllib2.Request('http://mediamining.univ-lyon2.fr:8080/cats/api')
    corpus_request.add_header('token', token)
    response = urllib2.urlopen(corpus_request)
    global json_corpus
    content = response.read()
    str_content = content.decode('utf-8')
    json_corpus = json.loads(str_content)
    output_file = codecs.open('csv/'+token+'.csv', 'w', 'utf-8')
    output_file.write('id\tshort_content\tfull_content\tauthors\tdate\n')
    for i in range(0, len(json_corpus)):
        parsed_date = datetime.fromtimestamp(json_corpus[i]['date']/1000.0)
        cleaned_text = json_corpus[i]['text'].replace('\n', '').replace('\r', '').replace('\t', ' ')
        cleaned_text = cleaned_text.replace('"', '')
        cleaned_text0 = re.sub(r'(?:https?\://)\S+', 'URL', cleaned_text)
        cleaned_text0 = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z_]+[A-Za-z0-9_]+)', 'USERNAME', cleaned_text0)
        name = 'N/A'
        if json_corpus[i].get('name') is not None:
            name = json_corpus[i]['name'].replace('\n', '').replace('\r', '').replace('\t', ' ').replace('"', '')
        csv_line = str(json_corpus[i]['id'])+'\t'+cleaned_text+'\t'+cleaned_text0+'\t'+name+'\t'+str(parsed_date.strftime('%Y-%m-%d'))
        output_file.write(csv_line+'\n')
    return None


def get_corpus_mabed():
    print('http://mediamining.univ-lyon2.fr:8080/cats/api/', token)
    corpus_request = urllib2.Request('http://mediamining.univ-lyon2.fr:8080/cats/api')
    corpus_request.add_header('token', token)
    response = urllib2.urlopen(corpus_request)
    global json_corpus
    content = response.read()
    str_content = content.decode('utf-8')
    json_corpus = json.loads(str_content)
    output_file = codecs.open('csv/'+str(token)+'.csv', 'w', 'utf-8')
    output_file.write('date\ttext\n')
    for i in range(0, len(json_corpus)):
        parsed_date = datetime.fromtimestamp(json_corpus[i]['date']/1000.0)
        cleaned_text = json_corpus[i]['text'].replace('\n', '').replace('\r', '').replace('\t', ' ')
        cleaned_text = cleaned_text.replace('"', '')
        cleaned_text0 = re.sub(r'(?:https?\://)\S+', 'URL', cleaned_text)
        csv_line = str(parsed_date.strftime('%Y-%m-%d %H:%M:%S'))+'\t'+cleaned_text0
        output_file.write(csv_line+'\n')
    return None


def transmit_events(t_token, t_k, t_tsl, t_theta, t_sigma, t_lang):
    my_corpus = MABED_Corpus('csv/'+t_token+'.csv')
    my_corpus.discretize(t_tsl)
    global mabed
    mabed = MABED(my_corpus)
    mabed.run(k=t_k, theta=t_theta, sigma=t_sigma)
    result_data = {'token': t_token, 'result': '<a href="http://mediamining.univ-lyon2.fr/people/guille/cats/mabed/' +
                                               t_token+'" target="_blank">Open the event browser in a new window</a>'}
    freeze_event_browser()
    json_data = json.dumps(result_data)
    results_request = urllib2.Request('http://mediamining.univ-lyon2.fr:8080/cats/module/result')
    results_request.add_header('Content-Type', 'application/json')
    results_request.data = json_data.encode('utf-8')
    urllib2.urlopen(results_request)
    print('Transmitted events for token '+t_token)
    os.remove('csv/' + t_token + '.csv')


def transmit_vocabulary(t_token):
    i_f = codecs.open('csv/'+t_token+'.csv', 'r', 'utf-8')
    lines = i_f.readlines()
    all_tweets = []
    corpus_size = 0
    for line in lines:
        row = line.split('\t')
        words = word_tokenize(row[1])
        all_tweets.extend([w.lower() for w in words])
        corpus_size += 1
    freq_distribution = FreqDist(all_tweets)
    cats_vocabulary_elements = []
    for word, frequency in freq_distribution.most_common(1000):
        if float(frequency)/float(corpus_size) < 0.7:
            cats_vocabulary_elements.append('["' + word + '", ' + str(frequency) + ']')
    cats_vocabulary = '['+','.join(cats_vocabulary_elements)+']'
    print(cats_vocabulary)
    result_data = {'token': t_token, 'result': cats_vocabulary}
    json_data = json.dumps(result_data)
    results_request = urllib2.Request('http://mediamining.univ-lyon2.fr:8080/cats/module/resultFile')
    results_request.add_header('Content-Type', 'application/json')
    results_request.data = json_data.encode('utf-8')
    urllib2.urlopen(results_request)
    print('Transmitted vocabulary for token '+t_token)
    os.remove('csv/' + t_token + '.csv')


def transmit_topic_model(t_token, t_model, t_k, t_min_tf, t_max_tf, t_lang):
    TOM_Corpus.MAX_FEATURES = 5000
    corpus = TOM_Corpus(source_file_path='csv/'+t_token + '.csv',
                        vectorization='tf',
                        max_relative_frequency=t_max_tf,
                        min_absolute_frequency=t_min_tf,
                        language=t_lang,
                        preprocessor=None)
    global topic_model
    if t_model == 'LDA':
        topic_model = LatentDirichletAllocation(corpus)
    elif t_model == 'NMF':
        topic_model = NonNegativeMatrixFactorization(corpus)
    if topic_model is not None:
        if t_k is None:
            t_k = 10
        t_k = int(t_k)
        topic_model.infer_topics(t_k)
        result_data = {'token': t_token, 'result': '<a href="http://mediamining.univ-lyon2.fr/people/guille/cats/tom/' +
                                                   t_token+'/topic_cloud.html" target="_blank">Open the topic model browser in a new window</a>'}
        json_data = json.dumps(result_data)
        results_request = urllib2.Request('http://mediamining.univ-lyon2.fr:8080/cats/module/result')
        results_request.add_header('Content-Type', 'application/json')
        results_request.data = json_data.encode('utf-8')
        urllib2.urlopen(results_request)
        print('Transmitted topic model for token '+t_token)
        prepare_topic_model_browser()
        freeze_topic_model_browser()
        prepare_topic_model_browser()
        os.remove('csv/' + t_token + '.csv')


@topic_model_browser_app.route('/')
def index():
    return render_template('index.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(topic_model.corpus.size),
                           method=type(topic_model).__name__,
                           corpus_size=topic_model.corpus.size,
                           vocabulary_size=len(topic_model.corpus.vocabulary),
                           max_tf=0.8,
                           min_tf=4,
                           vectorization='tf-idf',
                           preprocessor='None',
                           num_topics=k)


@topic_model_browser_app.route('/topic_cloud.html')
def topic_cloud():
    return render_template('topic_cloud.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(topic_model.corpus.size))


@topic_model_browser_app.route('/vocabulary.html')
def vocabulary():
    word_list = []
    for i in range(len(topic_model.corpus.vocabulary)):
        word_list.append((i, topic_model.corpus.word_for_id(i)))
    splitted_vocabulary = []
    words_per_column = int(len(topic_model.corpus.vocabulary)/5)
    for j in range(5):
        sub_vocabulary = []
        for l in range(j*words_per_column, (j+1)*words_per_column):
            sub_vocabulary.append(word_list[l])
        splitted_vocabulary.append(sub_vocabulary)
    return render_template('vocabulary.html',
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(topic_model.corpus.size),
                           splitted_vocabulary=splitted_vocabulary,
                           vocabulary_size=len(word_list))


@topic_model_browser_app.route('/topic/<tid>.html')
def topic_details(tid):
    ids = topic_associations[int(tid)]
    documents = []
    for document_id in ids:
        document_author_id = []
        for author_name in topic_model.corpus.authors(document_id):
            document_author_id.append((author_list.index(author_name), author_name))
        documents.append((topic_model.corpus.short_content(document_id),
                          document_author_id,
                          topic_model.corpus.date(document_id), document_id))
    return render_template('topic.html',
                           topic_id=tid,
                           frequency=round(topic_model.topic_frequency(int(tid))*100, 2),
                           documents=documents,
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(topic_model.corpus.size))


@topic_model_browser_app.route('/document/<did>.html')
def document_details(did):
    vector = topic_model.corpus.vector_for_document(int(did))
    word_list = []
    for a_word_id in range(len(vector)):
        word_list.append((topic_model.corpus.word_for_id(a_word_id), round(vector[a_word_id], 3), a_word_id))
    word_list.sort(key=lambda x: x[1])
    word_list.reverse()
    nb_words = 20
    documents = []
    return render_template('document.html',
                           doc_id=did,
                           words=word_list[:nb_words],
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(topic_model.corpus.size),
                           documents=documents,
                           authors=', '.join(topic_model.corpus.authors(int(did))),
                           year=topic_model.corpus.date(int(did)),
                           short_content=topic_model.corpus.short_content(int(did)),
                           article_id='')


@topic_model_browser_app.route('/word/<wid>.html')
def word_details(wid):
    documents = []
    for document_id in topic_model.corpus.docs_for_word(int(wid)):
        document_author_id = []
        for author_name in topic_model.corpus.authors(document_id):
            document_author_id.append((author_list.index(author_name), author_name))
        documents.append((topic_model.corpus.short_content(document_id).capitalize(),
                          document_author_id,
                          topic_model.corpus.date(document_id), document_id))
    return render_template('word.html',
                           word_id=wid,
                           word=topic_model.corpus.word_for_id(int(wid)),
                           topic_ids=range(topic_model.nb_topics),
                           doc_ids=range(topic_model.corpus.size),
                           documents=documents)


def prepare_topic_model_browser():
    global topic_associations, author_list, token
    os.makedirs('tom/'+token+'/static/data')
    topic_associations = topic_model.documents_per_topic()
    author_list = topic_model.corpus.all_authors()
    utils.save_topic_cloud(topic_model, 'tom/' + token + '/static/data/topic_cloud.json')
    print('Export details about topics')
    for topic_id in range(topic_model.nb_topics):
        utils.save_word_distribution(topic_model.top_words(topic_id, 20),
                                     'tom/' + token + '/static/data/word_distribution' + str(topic_id) + '.tsv')
    print('Export details about documents')
    for doc_id in range(topic_model.corpus.size):
        utils.save_topic_distribution(topic_model.topic_distribution_for_document(doc_id),
                                      'tom/' + token + '/static/data/topic_distribution_d' + str(doc_id) + '.tsv')
    print('Export details about words')
    for word_id in range(len(topic_model.corpus.vocabulary)):
        utils.save_topic_distribution(topic_model.topic_distribution_for_word(word_id),
                                      'tom/' + token + '/static/data/topic_distribution_w' + str(word_id) + '.tsv')


def freeze_topic_model_browser():
    global token
    topic_model_browser_app.config.update(
        FREEZER_BASE_URL='http://mediamining.univ-lyon2.fr/people/guille/cats/tom/' + token,
        FREEZER_DESTINATION='tom/' + token,
        FREEZER_IGNORE_404_NOT_FOUND=True,
        FREEZER_REMOVE_EXTRA_FILES=False,
    )
    topic_model_browser_app.debug = False
    topic_model_browser_app.testing = True
    topic_model_browser_app.config['ASSETS_DEBUG'] = False
    print('Freeze topic model browser')
    topic_model_freezer = Freezer(topic_model_browser_app)
    print('Finalizing the topic model browser...')

    @topic_model_freezer.register_generator
    def topic_details():
        for _topic_id in range(topic_model.nb_topics):
            yield {'tid': _topic_id}

    @topic_model_freezer.register_generator
    def document_details():
        for _doc_id in range(topic_model.corpus.size):
            yield {'did': _doc_id}

    @topic_model_freezer.register_generator
    def word_details():
        for _word_id in range(len(topic_model.corpus.vocabulary)):
            yield {'wid': _word_id}

    topic_model_freezer.freeze()
    print('Done.')


@event_browser_app.route('/')
def index():
    event_descriptions = []
    impact_data = []
    formatted_dates = []
    for i in range(0, mabed.corpus.time_slice_count):
        formatted_dates.append(int(time.mktime(mabed.corpus.to_date(i).timetuple())) * 1000)
    for event in mabed.events:
        mag = event[0]
        main_term = event[2]
        raw_anomaly = event[4]
        formatted_anomaly = []
        time_interval = event[1]
        related_terms = []
        for related_term in event[3]:
            related_terms.append(related_term[0] + ' (' + str("{0:.2f}".format(related_term[1])) + ')')
        event_descriptions.append((mag,
                                   str(mabed.corpus.to_date(time_interval[0])),
                                   str(mabed.corpus.to_date(time_interval[1])),
                                   main_term,
                                   ', '.join(related_terms)))
        for i in range(0, mabed.corpus.time_slice_count):
            value = 0
            if time_interval[0] <= i <= time_interval[1]:
                value = raw_anomaly[i]
            formatted_anomaly.append('[' + str(formatted_dates[i]) + ',' + str(value) + ']')
        impact_data.append('{"key":"' + main_term + '", "values":[' + ','.join(formatted_anomaly) + ']}')
    return render_template('template.html',
                           events=event_descriptions,
                           event_impact='['+','.join(impact_data) + ']',
                           k=k,
                           theta=theta,
                           sigma=sigma)


def freeze_event_browser():
    global topic_associations, author_list, token
    os.makedirs('mabed/'+token+'/static/data')
    print('Freeze event browser')
    event_browser_app.config.update(
        FREEZER_BASE_URL='http://mediamining.univ-lyon2.fr/people/guille/cats/mabed/'+token,
        FREEZER_DESTINATION='mabed/'+token,
        FREEZER_IGNORE_404_NOT_FOUND=True,
        FREEZER_REMOVE_EXTRA_FILES=False,
    )
    event_browser_freezer = Freezer(event_browser_app)
    event_browser_app.debug = False
    event_browser_app.testing = True
    event_browser_app.config['ASSETS_DEBUG'] = False
    event_browser_freezer.freeze()
    print('Done.')

if __name__ == '__main__':
    web_service_app.run(debug=True, host='localhost', port=5000)
