#grammar parsing
from django.db import connection
import psycopg2


def parse_sentence(user_input):                               #returns root word, triples of StanfordDependencyParser
    import os
    from nltk.parse.stanford import StanfordDependencyParser

    path = '/Users/woojgh/Documents/politician-tweets/stanford-corenlp/'
    path_to_jar = path + 'stanford-corenlp-3.9.1.jar'
    path_to_models_jar = path + 'stanford-corenlp-3.9.1-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
    os.environ['JAVA_HOME'] = "/Library/Java/JavaVirtualMachines/jdk-11.0.1.jdk/Contents/Home"
    result = dependency_parser.raw_parse(user_input)
    dep = next(result)                                                          # get next item from the iterator result
    return dep.triples(),dep.root["word"]


# classification into statements questions and chat
def classify_model():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    FNAME = '/Users/woojgh/Documents/politician-tweets/words/featuresDump.csv'
    df = pd.read_csv(filepath_or_buffer=FNAME,  error_bad_lines=False)
    df.columns = df.columns[:].str.strip()

    # Strip any leading spaces from col names
    breakpoint()
    df['class'] = df['class'].map(lambda x: x.strip())
    width = df.shape[1]

    # split into test and training (is_train: True / False col)
    np.random.seed(seed=1)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]
    features = df.columns[1:width-1]  #remove the first ID col and last col=classifier

    # Fit an RF Model for "class" given features
    clf = RandomForestClassifier(n_jobs=2, n_estimators = 100)
    clf.fit(train[features], train['class'])

    # Predict against test set
    preds = clf.predict(test[features])
    predout = pd.DataFrame({ 'id' : test['id'], 'predicted' : preds, 'actual' : test['class'] })
    return clf


def classify_sentence(clf,user_input):
    from token_maker import features_dict
    import pandas as pd
    keys = ["id",
            "wordCount",
            "stemmedCount",
            "stemmedEndNN",
            "CD",
            "NN",
            "NNP",
            "NNPS",
            "NNS",
            "PRP",
            "VBG",
            "VBZ",
            "startTuple0",
            "endTuple0",
            "endTuple1",
            "endTuple2",
            "verbBeforeNoun",
            "qMark",
            "qVerbCombo",
            "qTripleScore",
            "sTripleScore",
            "class"]

    myFeatures = features_dict('1',user_input, 'X')
    values=[]
    for key in keys:
        values.append(myFeatures[key])
    s = pd.Series(values)
    width = len(s)
    myFeatures = s[1:width-1]  #All but the last item (this is the class for supervised learning mode)
    predict = clf.predict([myFeatures])
    return predict[0].strip()


# setup database
def setup_database():
    db = psycopg2.connect("dbname = 'poli_dict' user = 'woojgh' host = 'localhost' password = 'woojgh'")
    with db.cursor() as cur:
        cur.execute("CREATE TABLE IF NOT EXISTS chat_table(id bigserial PRIMARY KEY, root_word VARCHAR(40), subject VARCHAR(40), verb VARCHAR(40), sentence VARCHAR(200))")
        cur.execute("CREATE TABLE IF NOT EXISTS statement_table(id bigserial PRIMARY KEY, root_word VARCHAR(40), subject VARCHAR(40), verb VARCHAR(40), sentence VARCHAR(200))")
        cur.execute("CREATE TABLE IF NOT EXISTS question_table(id bigserial PRIMARY KEY, root_word VARCHAR(40), subject VARCHAR(40), verb VARCHAR(40), sentence VARCHAR(200))")
        # cur.execute("CREATE TABLE IF NOT EXISTS directions_table(id bigserial PRIMARY KEY, origin_location VARCHAR(100), destination_location VARCHAR(100))")
        cur.execute("CREATE TABLE IF NOT EXISTS weather_table(day VARCHAR(100) PRIMARY KEY, location VARCHAR(100))")
        cur.execute("CREATE TABLE IF NOT EXISTS dictionary_table(id bigserial PRIMARY KEY, lookup_word VARCHAR(100), lookup_date VARCHAR(100))")
    db.commit()
    db.close()


# add classified sentences to database
def add_to_database(classification,subject,root,verb,H):

    db = psycopg2.connect("dbname = 'poli_dict' user = 'woojgh' host = 'localhost' password = 'woojgh'")
    cur = db.cursor()
    if classification == 'C':
        cur.execute("INSERT INTO chat_table(root_word,verb,sentence) VALUES (%s,%s,%s)",(str(root),str(verb),H))
        db.commit()
    elif classification == 'Q':
        cur.execute("SELECT sentence FROM question_table")
        res = cur.fetchall()
        exist = 0
        for r in res:
            if r[-1] == H:
                exist = 1
                break
        if exist == 0:                                                          #do not add if question already exists
            cur.execute("INSERT INTO question_table(subject,root_word,verb,sentence) VALUES (%s,%s,%s,%s)",(str(subject),str(root),str(verb),H))
            db.commit()
    else:
        cur.execute("SELECT sentence FROM statement_table")
        res = cur.fetchall()
        exist = 0
        for r in res:
            if r[-1] == H:
                exist = 1
                break
        if exist == 0:                                                          #do not add if question already exists
            cur.execute("INSERT INTO statement_table(subject,root_word,verb,sentence) VALUES (%s,%s,%s,%s)",(str(subject),str(root),str(verb),H))
            db.commit()
    db.close()

#get a random chat response
def get_chat_response():

    db = psycopg2.connect("dbname = 'poli_dict' user = 'woojgh' host = 'localhost' password = 'woojgh'")
    cur = db.cursor()
    
    cur.execute("SELECT COUNT(*) FROM chat_table")
    res = cur.fetchone()
    total_chat_records = res[0]
    import random
    chat_id = random.randint(1,total_chat_records+1)
    cur.execute("SELECT sentence FROM chat_table WHERE id = %s" % (int(chat_id)))
    res = cur.fetchone()
    try:
        B = res[0]
    except:
        import pdb; pdb.set_trace()
    return B

def get_question_response(subject,root,verb):

    db = psycopg2.connect("dbname = 'poli_dict' user = 'woojgh' host = 'localhost' password = 'woojgh'")
    cur = db.cursor()
    if str(subject) == '[]':
        cur.execute('SELECT verb FROM statement_table')
        res = cur.fetchall()
        found = 0
        for r in res:
            if r[-1] == str(verb):
                found = 1
                break
        if found == 1:
            cur.execute('SELECT sentence FROM statement_table WHERE verb="%s"'% (str(verb)))
            res = cur.fetchone()
            B = res[0]
            return B,0
        else:
            B = "Sorry I don't know the response to this. Please train me."
            return B,1
    else:
        cur.execute('SELECT subject FROM statement_table')
        res = cur.fetchall()
        found = 0
        for r in res:
            if r[-1] == str(subject):
                found = 1
                break
        if found == 1:
            try:
                cur.execute('SELECT verb FROM statement_table WHERE subject="%s"' % (str(subject)))
            except:
                import pdb; pdb.set_trace()
            res = cur.fetchone()
            checkVerb = res[0]                                                  #checkVerb is a string while verb is a list. checkVerb ['verb']
            if checkVerb == '[]':
                cur.execute('SELECT sentence FROM statement_table WHERE subject="%s"' % (str(subject)))
                res = cur.fetchone()
                B = res[0]
                return B,0
            else:
                if checkVerb[2:-2] == verb[0]:
                    cur.execute('SELECT sentence FROM statement_table WHERE subject="%s"' % (str(subject)))
                    res = cur.fetchone()
                    B = res[0]
                    return B,0
                else:
                    B = "Sorry I don't know the response to this. Please train me."
                    return B,1
        else:
            B = "Sorry I don't know the response to this. Please train me."
            return B,1

def add_learnt_statement_to_database(subject,root,verb):

    db = psycopg2.connect("dbname = 'poli_dict' user = 'woojgh' host = 'localhost' password = 'woojgh'")
    cur = db.cursor()
    cur.execute("INSERT INTO statement_table(subject,root_word,verb) VALUES (%s,%s,%s)",(str(subject),str(root),str(verb)))
    db.commit()

def learn_question_response(H):

    db = psycopg2.connect("dbname = 'poli_dict' user = 'woojgh' host = 'localhost' password = 'woojgh'")
    cur = db.cursor()
    cur.execute("SELECT id FROM statement_table ORDER BY id DESC")
    res = cur.fetchone()
    last_id = res[0]
    cur.execute('UPDATE statement_table SET sentence=%s WHERE id=%s',(H,last_id))
    db.commit()
    B = "Thank you! I have learnt this."
    return B,0
