import pandas as pd
import tqdm
import json
import copy
import numpy as np
from ast import literal_eval

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import plot_multiclass_precision_recall


final_file = 'dataset/balanced_embedding.jsonl'

def preprocess(final_file):
    data = list(map(json.loads, open(final_file)))
    df = pd.DataFrame(data)
    df['embedding'] = df.embedding.apply(np.array)
    x_train, x_test, y_train, y_test = train_test_split(list(df.embedding.values), df.best_strategy, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
def train(x_train, y_train, num):
    clf = RandomForestClassifier(n_estimators=num, random_state=42)
    clf.fit(x_train, y_train)
    return clf

def eval(x_test, y_test, clf):
    preds = clf.predict(x_test)
    probas = clf.predict_proba(x_test)
    
    # report = classification_report(y_test, preds)
    # print(report)
    plot_multiclass_precision_recall(probas, y_test, [0, 1, 2, 3, 4], clf)
    # plot_multiclass_precision_recall(probas, y_test, ['Zeroshot', 'Zeroshot_CoT', 'Fewshot', 'Fewshot_CoT', 'SelfDebug'], clf)




if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess(final_file)
    for i in list(range(10, 201, 10)):
        clf = train(x_train, y_train, i)
        eval(x_test, y_test, clf)
    # clf = train(x_train, y_train, 140)
    # eval(x_test, y_test, clf)
    