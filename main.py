#コンバージョンに至ったユーザーの行動履歴からsvmで分類気作成
from sklearn import svm
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # Here
args = sys.argv

cwd=os.getcwd()
print (cwd)

# 重要語データ
terms=[]
f = open(args[1])
line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
while line:
    terms.append(line.replace('\n',''))
    line = f.readline()
f.close
print(terms)

# 学習データ
data_training = []
label_training = []
f = open(args[2])
line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
while line:
    data_training.append([1 if terms[i] in line.split('\t')[1] else 0 for i in range(len(terms))])
    label_training.append(int(line.split('\t')[0]))
    line = f.readline()
f.close


print("data_training:", data_training)
print("label_training:", label_training)


# 試験データ
data_test = []
label_test = []
f = open(args[3])
line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
while line:
    data_test.append([1 if terms[i] in line.split('\t')[1] else 0 for i in range(len(terms))])
    label_test.append(int(line.split('\t')[0]))
    line = f.readline()
f.close
print("data_test:", data_test)
print("label_test:", label_test)

# 学習(グリッドサーチ-データを複数分割し組み合わせの中から最適解を抽出)
tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]
score = 'f1'
clf = GridSearchCV(
    SVC(), # 識別器
    tuned_parameters, # 最適化したいパラメータセット
    cv=5, # 交差検定の回数
    scoring='%s_weighted' % score ) # モデルの評価関数の指定

clf.fit(data_training, label_training)

#結果表示
print("# Tuning hyper-parameters for %s" % score)
print()
print("Best parameters set found on development set: %s" % clf.best_params_)
print()

# それぞれのパラメータでの試行結果の表示
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

# テストデータセットでの分類精度を表示
print("The scores are computed on the full evaluation set.")
print()

print(classification_report(label_test, clf.predict(data_test)))
