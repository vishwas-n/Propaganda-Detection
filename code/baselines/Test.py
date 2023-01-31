train_folder = "../datasets/train-articles" # check that the path to the datasets folder is correct,
dev_folder = "../datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "../datasets/train-task-flc-tc.labels"
# dev_template_labels_file = "../datasets/test-task-tc-template.out"
dev_template_labels_file = "../datasets/dev-task-flc-tc.labels"
task_TC_output_file = "baseline-output-TC_train_test.txt"

import glob
import os.path
import codecs
import helper
import json

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles


def read_predictions_from_file(filename):
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels


def get_texts(articles_map, article_ids, span_starts, span_ends):
    texts = []
    for article_id, sp_start, sp_end in zip(article_ids, span_starts, span_ends):
        sentence = articles_map.get(article_id)[int(sp_start):int(sp_end)]
        texts.append(sentence)
    return texts

def get_data_stats(texts, gold_labels):
    pt_map = {}
    for technique, sentence in zip(gold_labels, texts):
        if technique not in pt_map:
            pt_map[technique] = []
        pt_map[technique].append(sentence)
    return pt_map



### MAIN ###
# loading articles' content from *.txt files in the train folder
articles_map = read_articles_from_file_list(train_folder)

# loading gold labels, articles ids and sentence ids from files *.task-TC.labels in the train labels folder
train_articles_ids, train_span_starts, train_span_ends, train_labels = read_predictions_from_file(train_labels_file)
print("Loaded TRAIN %d annotations from %d articles" % (len(train_span_starts), len(set(train_articles_ids))))
train_texts = get_texts(articles_map, train_articles_ids, train_span_starts, train_span_ends)


# reading data from the development set
dev_articles_map = read_articles_from_file_list(dev_folder)
dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
print("Loaded DEV %d annotations from %d articles" % (len(train_span_starts), len(set(train_articles_ids))))
dev_texts = get_texts(dev_articles_map, dev_article_ids, dev_span_starts, dev_span_ends)

helper.stats(dev_article_ids, dev_span_starts, dev_span_ends)

def write_stats_to_file(prop_tech_map, filename):
    stats_map = {}
    stats_map['propaganda_techniques'] = list(prop_tech_map.keys())
    for key, value in prop_tech_map.items():
        stats_map[key] = {}
        stats_map[key]['count'] = len(value)
        stats_map[key]['sentences'] = value

    with open('results/'+filename, 'w', encoding='utf-8') as fout:
        json.dump(stats_map, fout, ensure_ascii=False)

train_pt_map = get_data_stats(train_texts, train_labels)
dev_pt_map = get_data_stats(dev_texts, dev_labels)

write_stats_to_file(train_pt_map, 'train_stats_map.json')
write_stats_to_file(dev_pt_map, 'dev_stats_map.json')

