from flask import Flask, render_template, request
import pickle, string, util
import numpy as np
import json, random
from os.path import exists

app = Flask(__name__)
app.static_folder = 'static'

with open("static/script/script.json", 'r') as f:
  data = json.load(f)

with open('static/models/lstm/depression&suicide.pkl', 'rb') as f:
    tokenizer, model = pickle.load(f)

with open('static/models/logistic/finalized_model.pkl', 'rb') as f:
        vectorizer, clf = pickle.load(f)

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


########################## ML Model ######################

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

# predict the output
def predict(sen, vectorizer, clf):

    # dictionary (convert the numeric result back to text)
    dic = {4: 'sadness', 3:'anger', 1:'love', 0:'fear', 2:'joy'}

    pun_sen = remove_punctuation(sen)
    vec_sen = vectorizer.transform([pun_sen])
    
    y_pred = clf.predict(vec_sen)
    # print(f'Input sentence: {sen}')
    # print("Prediction: ", dic[y_pred[0]])

    return dic[y_pred[0]]




######################### CNN Model ########################
# def load_cnn_model():
#     model = load_model('static/models/emotion_cnn/model-recent.h5')
#     return model

# def preprocess_image(img):
#     he = util.face(img)
#     img2 = he.cropDetect()

#     img = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
#     resized = cv2.resize(img, (48,48), interpolation = cv2.INTER_LANCZOS4)
#     print(resized.shape)

#     img_batch = np.expand_dims(resized, 0)
#     print(img_batch.shape)

#     return img_batch

# def predict_cnn(model, img_batch):
#     y_val_pred = model.predict(img_batch)
#     print(y_val_pred)

## live on browser
# @app.route('/hy')
# def main():
#     return render_template('hybrid.html')

# @app.route("/res", methods=['POST', 'GET'])
# def res_cnn_response():
#     print("Yoshiiiiidcndncdncndcdcdcdcd\n\n\n\n\n\n\n")
#     img = request.args.get('img')
#     img_batch = preprocess_image(img)
#     print("dcndncdncndcdcdcdcd\n\n\n\n\n\n\n")
#     return img_batch


################### BOT #########################
@app.route("/get")
def get_bot_response():
        userText = request.args.get('msg')
        i = int(request.args.get('i'))
        data2 = data['intents'][i]
        i += 1
        l = len(data2['responses'])
        ques = data2['responses'][random.randint(0,l-1)]
        # (userText, ques)

        remove_punctuation(userText)
        pre = predict(userText, vectorizer, clf)
        load_lstm_model(tokenizer, model, userText, pre, ques)

        return ques


################### LSTM ######################
def load_lstm_model(tokenizer, model, text, pre, ques):

    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 150
    
    pred = util.trigger(text)
    if not pred:
        new_complaint = [text]
        seq = tokenizer.texts_to_sequences(new_complaint)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        pred = model.predict(padded)

    labels = ['Neutral', 'Sucide', 'Depressed']

    path1 = 'static/report.txt'
    
    if exists(path1):
        file = open(path1, "a")
        file.write(ques + ', ' + text + ', ' + labels[np.argmax(pred)] + ', ' + pre + '\n')
        file.close()
    else:
        file = open(path1, "w")
        file.write(ques + ', ' + text + ', ' + labels[np.argmax(pred)] + ', ' + pre + '\n')
        file.close()

    print(pred, labels[np.argmax(pred)])
    



################## MAIN ######################
@app.route('/')
def home():
    ## ML Model
    # vectorizer, clf, dic = load_ml_model()
    # sen = 'I am great..'
    # predict(sen, vectorizer, clf, dic)

    # img = 'static/images/j1.jpeg'
    # ## CNN model
    # model = load_cnn_model()
    # img_batch = preprocess_image(img)
    # predict_cnn(model, img_batch)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()