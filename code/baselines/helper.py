import numpy as np

def get_glove_embeddings_dict():
    word_embeddings = {}
    f = open('glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    return word_embeddings

def get_sentence_vector(sentence, embeddings_dict):
    sentence_vector = []
    if len(sentence) != 0:
        sentence_vector = sum([embeddings_dict.get(w, np.zeros((300,))) for w in sentence.split()]) / len(sentence.split())
    else:
        sentence_vector = np.random.rand(300, )
    return sentence_vector

def stats(dev_article_ids, dev_span_starts, dev_span_ends):
    count_map = {}
    for id, start, end in zip(dev_article_ids, dev_span_starts, dev_span_ends):
        if id not in count_map:
            count_map[id] = {}
        if start not in count_map[id]:
            count_map[id][start] = {}
        if end not in count_map[id][start]:
            count_map[id][start][end] = 1
        else:
            count_map[id][start][end] += 1

    for id in count_map.keys():
        for start in count_map[id].keys():
            for end in count_map[id][start].keys():
                if count_map[id][start][end] > 1:
                    print(id, start, end, count_map[id][start][end])

def get_predictions_from_model(data, model, dev_article_ids, dev_span_starts, dev_span_ends):
    prediction_map = {}
    predictions = []
    for record, id, start, end in zip(data, dev_article_ids, dev_span_starts, dev_span_ends):
        if (id+start+end) in prediction_map:
            prediction_map[id + start + end] += 1
        else:
            prediction_map[id + start + end] = 1
        index = -(prediction_map[id + start + end])
        prediction = model.classes_[np.argsort(model.predict_proba([record]))[0][index]]
        predictions.append(prediction)
    return  predictions