from gensim.models import doc2vec
import numpy as np
import networkx as nx
import node2vec
import os
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tg_split

dim = 100
window_size = 5
seed = 1
d2v_alpha = 0.025
d2v_min_alpha = 0.0001
neg = 10
n2v_iter = 1
d2v_iter = 20


def read_graph(edgelist):
    g = nx.read_edgelist(edgelist, nodetype=int, create_using=nx.DiGraph())
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1
    g = g.to_undirected()
    return g

def read_content(doc_file):
    c = {}
    with open(doc_file) as f:
        for line in f.readlines():
            line = line.split()
            doc_id = line[0]
            doc_content = [str(x) for x in np.where(np.array(line[1:-1]) == '1')[0]]
            c[doc_id] = doc_content
    return c

def create_embeddings(content,docs,walks,combined):
    #d2v-sg
    d2v_sg = doc2vec.Doc2Vec(docs, size=dim, window=window_size, iter=d2v_iter, sample=0, min_count=0, dm=0, negative=neg, alpha=d2v_alpha, min_alpha=d2v_min_alpha, seed=seed)
    with open('emb/d2v-sg.emb','w') as f:
        for t in content:
            f.write(str(t) + " " + ' '.join(map(str,d2v_sg.docvecs[t])) + '\n')

    #d2v-dm
    d2v_dm = doc2vec.Doc2Vec(docs, size=dim, window=window_size, iter=d2v_iter, sample=0, min_count=0, dm=1, negative=neg, alpha=d2v_alpha, min_alpha=d2v_min_alpha, seed=seed)
    with open('emb/d2v-dm.emb','w') as f:
        for t in content:
            f.write(str(t) + " " + ' '.join(map(str,d2v_dm.docvecs[t])) + '\n')

    #n2v
    n2v_model = Word2Vec(walks, size=dim, window=window_size,min_count=0,sg=1,sample=0,workers=8,iter=n2v_iter,alpha=0.025,min_alpha=d2v_min_alpha,negative=neg,seed=seed)
    n2v_model.save_word2vec_format('emb/n2v.emb')

    # concat
    with open('emb/concat.emb', 'w') as f:
        for t in content:
            f.write(str(t) + ' ' +  ' '.join(map(str,d2v_sg.docvecs[t])) + ' ' +  ' '.join(map(str,n2v_model[t])) + '\n')

    #tg-dm
    tgdm_model = doc2vec.Doc2Vec(combined, size=dim, window=window_size, iter=10, sample=0, min_count=0, dm=1, negative=neg, alpha=d2v_alpha, min_alpha=d2v_min_alpha, seed=seed)
    with open('emb/tg-dm.emb','w') as f:
        for t in content:
            f.write(str(t) + " " + ' '.join(map(str,tgdm_model.docvecs[t])) + '\n')

    #tf-idf
    with open('input/cora.tfidf') as tf:
        with open('emb/cora-tfidf.emb','w') as out:
            for line in tf.readlines():
                out.write(' '.join(map(str,line.split()[:-1])) + '\n')

    #tg-sg
    raw_docs = []
    for t in content:
        raw_docs.append([t] + content[t])
    tgsg_model = tg_split.TGSG(walks,raw_docs, dim=dim, alpha=0.025, neg_samples=neg, window=window_size, seed=seed)
    tgsg_model.train(iterations=5)
    with open('emb/tg-sg.emb', 'w') as f:
        for i in range(len(tgsg_model.index2word)):
            f.write(str(tgsg_model.index2word[i]) + " " + ' '.join(map(str,tgsg_model.i2h[i])) + '\n')

    #p2v
    p2v_model = doc2vec.Doc2Vec(alpha=0.025, window=5, min_count=5, min_alpha=0.025, size=100, workers=8, seed=seed)  # use fixed learning rate
    p2v_model.build_vocab(docs)
    for epoch in range(10):
        p2v_model.train(docs)
        p2v_model.alpha -= 0.002
        p2v_model.min_alpha = p2v_model.alpha
    with open('emb/p2v-dm.emb','w') as f:
        f.write('2708 100\n')
        for t in content:
            f.write(str(t) + " " + ' '.join(map(str,p2v_model.docvecs[t])) + '\n')
    # knn
    index2doc = dict()
    X = []
    i = 0
    for t in content:
        index2doc[i] = t
        X.append(p2v_model.docvecs[t])
        i+=1
    X = np.array(X)
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    d,idx = nbrs.kneighbors(X)

    with open('input/cora.edgelist') as f:
        with open('input/cora.artifical.edgelist','w') as c:
            for line in f:
                c.write(line)
            for i in range(len(idx)):
                for n in idx[i][1:]:
                    c.write(str(index2doc[i]) + '\t' + str(index2doc[n]) + '\n')
    a_nx_g = read_graph('input/cora.artifical.edgelist')
    a_G = node2vec.Graph(a_nx_g, False, 1, 1)
    a_G.preprocess_transition_probs()
    a_walks = a_G.simulate_walks(10, 40)
    a_walks = [[str(x) for x in walk] for walk in a_walks]
    with open('input/cora.artifical.walks','w') as w:
        for walk in a_walks:
            w.write(' '.join(walk) + '\n')
    p2v_enriched = Word2Vec(size=100, window=5, alpha=0.0025, min_count=0, workers=8, seed=seed)
    p2v_enriched.build_vocab(a_walks)
    p2v_enriched.intersect_word2vec_format('emb/p2v-dm.emb')
    p2v_enriched.train(a_walks)
    p2v_enriched.save_word2vec_format('emb/p2v.emb')


def get_emb(targets, emb_file):
    emb = {}
    with open(emb_file) as f:
        for line in f.readlines():
            line = line.split()
            if line[0] in targets:
                emb[line[0]] = [float(x) for x in line[1:]]

    X = []
    y = []
    for t in targets:
        X.append(emb[t])
        y.append(targets[t])
    return X,y


def evaluate():
    targets = {}
    with open('input/cora.content') as f:
        for line in f.readlines():
            line = line.split()
            targets[line[0]] = line[-1]

    path = 'emb/'
    for file in os.listdir(path):
        print(file)
        X,y = get_emb(targets,path+file)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        clf = SVC(kernel='rbf', C=1.5)
        scores = {}
        num_iter = 20
        percentages = []
        for i in range(10,91,10):
            p = i / 100.0
            percentages.append(p)
            scores[p] = []
        for it in range(num_iter):
            for percentage in percentages:
                X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,train_size=percentage, random_state=it)
                clf.fit(X_train,y_train)
                pred = clf.predict(X_test)
                score = accuracy_score(y_test, pred)
                scores[percentage].append(score)
        for p in percentages:
            with open('results/'+str(p) + '-percent.csv','a') as res:
                res.write(file[:-4] + ' ' + ' '.join([str(x) for x in scores[p]]) + '\n')
        means = [np.mean(scores[p]) for p in percentages]
        with open('results/means.csv', 'a') as res:
            res.write(file[:-4] + ' & ' + ' & '.join([str("%.1f" % (x * 100.0)) for x in means]) + '\n')
        with open('results/overview.csv', 'a') as res:
            all = []
            for p in percentages:
                all += scores[p]
            res.write(file[:-4] + ' ' + str("%.1f" % (np.mean(all) * 100.0)) + '\n')
        print(means)



def main():
    # documents
    content = read_content('input/cora.content')
    docs = []
    for key in content:
        docs.append(doc2vec.TaggedDocument(content[key],[key]))

    #walks
    nx_g = read_graph('input/cora.edgelist')
    G = node2vec.Graph(nx_g, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(10, 40)
    walks = [[str(x) for x in walk] for walk in walks]
    with open('input/cora.walks','w') as w:
        for walk in walks:
            w.write(' '.join(walk) + '\n')

    combined = []
    for key in content:
        combined.append(doc2vec.TaggedDocument(content[key],[key]))
    for walk in walks:
        combined.append(doc2vec.TaggedDocument(walk, [walk[0]]))

    create_embeddings(content,docs,walks,combined)
    evaluate()


if __name__ == "__main__":
    main()