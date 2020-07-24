import ufal.udpipe
import pandas as pd
import numpy as np
from io import StringIO
import stopwordsiso as stopwords
from polyglot.detect import Detector
from polyglot.detect.base import UnknownLanguage
from polyglot.downloader import downloader
from polyglot.text import Text
from polyglot.mapping import Embedding


global lang_dict
lang_dict = {}


class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""
        output_format = ufal.udpipe.OutputFormat.newOutputFormat(format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()
        return output


########
def langmodelload(language):
    ########################
    global stop_words
    global question_words
    global embeddings
    global model
    global lang_dict
    ########################
    LibLocLang = "./udpipe-ud/"
    ########################
    if language == "en":
        model = Model(LibLocLang + 'english-ewt-ud-2.5-191206.udpipe')
    elif language == "ar":
        model = Model(LibLocLang + 'arabic-padt-ud-2.5-191206.udpipe')
    elif language == "zh":
        model = Model(LibLocLang + 'chinese-gsdsimp-ud-2.5-191206.udpipe')
    elif language == "id":
        model = Model(LibLocLang + 'indonesian-gsd-ud-2.5-191206.udpipe')
    elif language == "ko":
        model = Model(LibLocLang + 'korean-gsd-ud-2.5-191206.udpipe')
    elif language == "pt":
        model = Model(LibLocLang + 'portuguese-gsd-ud-2.5-191206.udpipe')
    elif language == "vi":
        model = Model(LibLocLang + 'vietnamese-vtb-ud-2.5-191206.udpipe')
    elif language == "hi":
        model = Model(LibLocLang + 'hindi-hdtb-ud-2.5-191206.udpipe')
    elif language == "jp":
        model = Model(LibLocLang + 'japanese-gsd-ud-2.5-191206.udpipe')
    elif language == 'es':
        model = Model(LibLocLang + 'spanish-gsd-ud-2.5-191206.udpipe')
    ########################
    base_question_words = ['where', 'which', "who", "why", "what", "when", "please", "how", "is", "are", "will",
                           "could", "should", "was", "were", "do", "did", "can"]
    question_words = []
    for i in range(0, len(base_question_words)):
        question_words.append(Text(base_question_words[i]).transliterate(language))
    ########################
    if stopwords.has_lang(
            language) and language != "hi" and language != "ar" and language != "zh" and language != "vi" and language != "ko" and language != "jp" and language != "id" and language != "ms":
        ########################
        stop_words = list(stopwords.stopwords(language))
        stop_words_list = []
        ########################
        for i in range(0, len(stop_words)):
            try:
                text = Text(stop_words[i], hint_language_code=language)
                ########################
                if (text.pos_tags[0][1] != "NOUN") and (text.pos_tags[0][1] != "VERB") and (
                        text.pos_tags[0][1] != "PRON"):
                    stop_words_list.append(text.pos_tags[0][0])
            except Exception as e:
                print(e)
        stop_words = stop_words_list
    else:
        print(language + " has errors.")
        stop_words = []
    ########################
    ########################

    embeddings = Embedding.load("./polyglot_data/embeddings2/" + language + "/embeddings_pkl.tar.bz2")
    lang_dict[language] = {'model': model, 'embeddings': embeddings, 'stop_words': stop_words}


###################################################################################
def emounifiedintentionmodel(SentenceToBe, synonym_num):
    global lang_dict
    ########################
    # if "langmodel" not in globals():
    #     global langmodel
    #     langmodel = 0
    #     global langcodeused
    #     langcodeused = ""
    ########################
    try:
        detector = Detector(SentenceToBe)
        langcode = detector.language.code
    except UnknownLanguage:
        langcode = 'en'
    if langcode == 'zh_Hant':
        langcode = 'zh'
    if langcode not in ['en', 'zh', 'ar', 'es', 'id', 'ja', 'ms', 'pt', 'vi']:
        # force to english if not recognized
        langcode = 'en'

    # if (langmodel != 1) or (langcode != langcodeused):
    #     langmodelload(langcode)
    #     langcodeused = langcode
    #     langmodel = 1
    try:
        model = lang_dict[langcode]['model']
        embeddings = lang_dict[langcode]['embeddings']
        stop_words = lang_dict[langcode]['stop_words']
    except KeyError:
        langmodelload(langcode)
        model = lang_dict[langcode]['model']
        embeddings = lang_dict[langcode]['embeddings']
        stop_words = lang_dict[langcode]['stop_words']

    ########################
    intention_filters = ['PRON', "VERB"]
    object_filters = ['NOUN', "PROPN"]
    ########################
    sentences = Text(SentenceToBe)
    sentences2 = model.tokenize(SentenceToBe)
    ########
    for s in sentences2:
        model.tag(s)  # inplace tagging
        model.parse(s)  # inplace parsing
    datause = pd.read_csv(StringIO(model.write(sentences2, "conllu")), sep="\t", header=None, skiprows=4).dropna()
    #datause.to_csv("./temp/"+langcode+".csv")
    PosTagIntention = datause[datause.columns[[1, 3]]].values.tolist()
    ################################
    #     ADJ: adjective
    #     ADP: adposition
    #     ADV: adverb
    #     AUX: auxiliary
    #     CCONJ: coordinating conjunction
    #     DET: determiner
    #     INTJ: interjection
    #     NOUN: noun
    #     NUM: numeral
    #     PART: particle
    #     PRON: pronoun
    #     PROPN: proper noun
    #     PUNCT: punctuation
    #     SCONJ: subordinating conjunction
    #     SYM: symbol
    #     VERB: verb
    #     X: other
    ################################
    sentence_intention = []
    sentence_object = []
    sentence_emotion = []
    ################################
    emotion_vec = 0
    negation_flag = 0
    ################################
    if len(PosTagIntention) > 1:
        emotion_vec = 0
        ################################
        for i in range(0, len(PosTagIntention)):
            #####
            if ((PosTagIntention[i][1] == "PUNCT") and (PosTagIntention[i][0] != ",")):
                sentence_emotion.append(str(emotion_vec))
                emotion_vec = 0
            #####
            if i == 0:
                if any(str(word).lower() in str(PosTagIntention[i][0]).lower() for word in question_words):
                    sentence_intention.append("Question")

                SingleWord = Text(PosTagIntention[i][0])
                SingleWord.language = langcode
                emotion_vec = SingleWord.words[0].polarity

                if (str(PosTagIntention[i][1]) == 'PART'):
                    negation_flag = 1
            #####
            else:
                if all(str(word).lower() != str(PosTagIntention[i][0]).lower() for word in stop_words):
                    if any(str(word).lower() in str(PosTagIntention[i][1]).lower() for word in intention_filters):
                        sentence_intention.append(PosTagIntention[i][0])
                    if any(str(word).lower() in str(PosTagIntention[i][1]).lower() for word in object_filters):
                        sentence_object.append(PosTagIntention[i][0])

                SingleWord = Text(PosTagIntention[i][0])
                SingleWord.language = langcode

                if (SingleWord.words[0].polarity != 0):
                    if str(PosTagIntention[i][1]) == 'ADJ':
                        if negation_flag == 0:
                            emotion_vec = SingleWord.words[0].polarity
                        else:
                            emotion_vec = SingleWord.words[0].polarity * (-1)
                    else:
                        emotion_vec = emotion_vec + abs(SingleWord.words[0].polarity)
                #####
        #####
        sentence_intention = list(set(sentence_intention))
        sentence_object = list(set(sentence_object))
        #####
    else:
        sentence_intention = []
        sentence_object = []
        sentence_emotion = []
        ################################
    intlength = len(sentence_intention)
    for i in range(0, intlength):
        try:
            neighbors = embeddings.nearest_neighbors(sentence_intention[i], synonym_num)
            if len(neighbors) > 1:
                sentence_intention.extend((neighbors))
        except:
            print(sentence_intention[i] + " cannot be extended")
    #####
    intlength = len(sentence_object)
    for i in range(0, intlength):
        try:
            neighbors = embeddings.nearest_neighbors(sentence_object[i], synonym_num)
            if len(neighbors) > 1:
                sentence_object.extend((neighbors))
        except:
            print(sentence_object[i] + " cannot be extended")
    #####
    NERobj = Text(SentenceToBe, hint_language_code=langcode).entities
    #####
    sentence_intention = list(set(sentence_intention))
    sentence_object = list(set(sentence_object))
    #####
    overall = sentence_intention + sentence_object
    return sentence_intention, sentence_object, overall, NERobj, sentence_emotion
    #####


def count_occurance(search_list, cur_list):
    count = 0
    found_terms = set([item.lower() for item in search_list]).intersection(set([item.lower() for item in cur_list]))
    count = len(found_terms)
    return count


def find_terms(search_list, cur_list):
    found_terms = set([item.lower() for item in search_list]).intersection(set([item.lower() for item in cur_list]))
    # logger.info(found_terms)
    return list(found_terms)
