import numpy as np
import dynet as dy
import string
from collections import Counter
import random

from sklearn.preprocessing import OneHotEncoder


class Trumpspeechgenerator:

    # model decleration and layer sizes with parameters are initialized in init method
    def __init__(self):
        print("hello from Talha Yılmaz :)")

        self.word_number=15
        self.epoch_num = 4
        self.embedding_layer_size = 32
        self.hidden_layer_size = 32
        self.min_count = 2

        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model)

        self.pW_hidden = self.model.add_parameters((self.hidden_layer_size, self.embedding_layer_size))
        self.pB_hidden = self.model.add_parameters(self.hidden_layer_size)

    # method to read trump speeches
    def load_doc(self, filename):
        file = open(filename, encoding="utf8")
        text = file.read()
        file.close()
        return text

    # cleaning on input data is done
    # Blank lines, 'SPEECH' words, punctuations are eliminated
    # All words set to lowercase
    def clean_doc(self, speech):
        sentence = speech.replace('\n',' ')
        sentence = sentence.replace('SPEECH', ' ')
        table = str.maketrans('', '', string.punctuation)

        tokens = sentence.split()
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        return tokens

    # training is done here
    # train for 1 epoch
    def train(self, oht_input, output_idxList):
        total_loss = 0.0
        temp_input_word = oht_input[0, :].tolist()
        input_word = list(map(int, temp_input_word))

        score = self.score_generate(input_word)
        loss = dy.pickneglogsoftmax(score, output_idxList[0])
        total_loss += loss.value()
        loss.backward()
        self.trainer.update()

        return total_loss

    # Softmax applied to generate scores for a word
    def score_generate(self, text):
        # Embedding parameters
        W_emb = self.model.add_lookup_parameters((self.vocabulary_size, self.embedding_layer_size))

        # Output layer parameters, softmax weight and bias
        pW_output = self.model.add_parameters((self.class_size, self.hidden_layer_size))
        pB_output = self.model.add_parameters((self.class_size))

        dy.renew_cg()
        word = text.index(1)

        h1 = dy.lookup(W_emb, word)
        h2 = dy.tanh(self.pW_hidden * h1 + self.pB_hidden)
        return pW_output * h2 + pB_output

    def generate_sentence(self, text_idx, idx_text, oht_input, idxArray):
        # Generate Sentence
        generated_sentence = []

        # start with random word,then predict according to that word
        idx = random.randint(0, len(idxArray))

        # each iteration ofthis loop, a word is predicted
        # prediction done according to the scores
        for i in range(self.word_number):
            oht_vector = oht_input[idx, :].tolist()
            temp_prediction = list(map(int, oht_vector))

            scores = self.score_generate(temp_prediction).npvalue()
            predicted_word = idx_text[np.argmax(scores)]

            generated_sentence.append(predicted_word)

            # for selecting next word
            predicted_idx = text_idx[predicted_word]
            selectable_idx_list = np.where(idxArray == predicted_idx)[0]

            idx = random.choice(selectable_idx_list)

        print(*generated_sentence, sep=" ")


    def generate_speech(self, clean_text):
        # a dictionary mapping all word to their indexes
        text_idx = dict()
        # a set containing text values according to indexes
        idx_text = []

        # if word isn't seen more than min_count number, it is not added
        counter = Counter(clean_text)
        for word, count in counter.items():
            if count >= self.min_count:
                idx_text.append(word)
                text_idx[word] = len(text_idx)

        # unique word size
        self.vocabulary_size = len(text_idx)
        total_size = len(clean_text)


        # one hot encoder generated using sklearn package
        # encıder generated from indexes
        idxList = []
        output_idxList = []

        for i in range(total_size-1):
            if text_idx.keys() >= set((clean_text[i],clean_text[i+1])):
                idxList.append(text_idx[clean_text[i]])
                output_idxList.append(text_idx[clean_text[i+1]])

        idxArray = np.array(idxList)
        oht_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        oht_input = oht_encoder.fit_transform(idxArray.reshape(-1, 1))

        self.class_size = len(set(output_idxList))

        # training
        for epoch in range(self.epoch_num):
            total_loss = self.train(oht_input, output_idxList)
            print("Epoch %d: loss=%f" % (epoch, total_loss))
        # prediction of words
        self.generate_sentence(text_idx, idx_text, oht_input, idxArray)


def main():
    trumpspeechgenerator = Trumpspeechgenerator()
    text = trumpspeechgenerator.load_doc('trumpspeeches.txt')
    clean_text = trumpspeechgenerator.clean_doc(text)
    trumpspeechgenerator.generate_speech(clean_text)


if __name__ == '__main__':
    main()