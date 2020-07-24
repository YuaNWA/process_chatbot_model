from io import StringIO

import pandas as pd
import stopwordsiso as stopwords
import ufal.udpipe
from polyglot.text import Text


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


################################################################################
CurLibLocLang = "./udpipe-ud/"


########
def langmodelload(language, LibLocLang=CurLibLocLang):
    ##
    global model
    global stop_words
    global question_words
    ###
    if language == "en":
        model = Model(LibLocLang + 'english-ewt-ud-2.5-191206.udpipe')
        question_words = ['where', 'which', "who", "why", "what", "when", "please", "how", "is", "are", "will", "could",
                          "should", "was", "were", "do", "did", "can"]
    elif language == "ar":
        model = Model(LibLocLang + 'arabic-padt-ud-2.5-191206.udpipe')
        question_words = ['أين', "أي", "من", "لماذا", "ماذا", "متى", "من فضلك", "كيف", "هي", "هي", "سوف", "يمكن", "يجب",
                          "كانت ", " كان ", " فعل ", " فعل ", " يمكنه "]
    elif language == "zh":
        model = Model(LibLocLang + 'chinese-gsdsimp-ud-2.5-191206.udpipe')
        question_words = ["哪里", "哪个", "谁", "为什么", "什么", "何时", "请", "如何", "是", "将", "可以", "应该", "被", "做"]
    elif language == "id":
        model = Model(LibLocLang + 'indonesian-gsd-ud-2.5-191206.udpipe')
        question_words = ['dimana', 'yang', "siapa", "mengapa", "apa", "ketika", "tolong", "bagaimana", "adalah",
                          "adalah", "akan", "bisa", "harus", "adalah", "adalah", "adalah", "lakukan ", " melakukan ",
                          " bisa "]
    elif language == "ko":
        model = Model(LibLocLang + 'korean-gsd-ud-2.5-191206.udpipe')
        question_words = ['어느', "누가 왜", "무엇", "언제", "제발", "어떻게", "는", "은", "의지", "할 수있다", "해야한다", "있었다", "있었다", "할",
                          "했다 ", "할 수있다"]
    elif language == "pt":
        model = Model(LibLocLang + 'portuguese-gsd-ud-2.5-191206.udpipe')
        question_words = ['onde', 'qual', "quem", "por que", "o que", "quando", "por favor", "como", "é", "vontade",
                          "poderia", "deveria", "era", "faz", "fez", "pode"]
    elif language == "vn":
        model = Model(LibLocLang + 'vietnamese-vtb-ud-2.5-191206.udpipe')
        question_words = ['đâu', 'cái nào', "Ai", "tại sao", "gì", "khi", "làm ơn", "làm thế nào", "là", "là", "sẽ",
                          "có thể", "nên", "đã", "đã", "làm", "đã", "có thể "]
    ########################
    if stopwords.has_lang(language):
        ########################
        stop_words = list(stopwords.stopwords(language))
        stop_words_list = []
        ########################
        for i in range(0, len(stop_words)):
            try:
                sentences = model.tokenize(stop_words[i])
                ########
                for s in sentences:
                    model.tag(s)  # inplace tagging
                    model.parse(s)  # inplace parsing
                ########
                datause = pd.read_csv(StringIO(model.write(sentences, "conllu")), sep="\t", header=None, skiprows=4)
                PosTagIntention = datause[datause.columns[2:4]].values.tolist()
                if (PosTagIntention[0][1] != "NOUN") and (PosTagIntention[0][1] != "VERB") and (
                        PosTagIntention[0][1] != "PRON"):
                    stop_words_list.append(PosTagIntention[0][0])
            except:
                print()
        stop_words = stop_words_list
    else:
        print(language + " has errors.")
        stop_words = []


###################################################################################
CurFileLocThe = "./uwn_tsv/uwn-dump_201012.tsv"


def langthesaurusload(language, FileLocThe=CurFileLocThe):
    ###############
    global thesaurus
    ###############
    thesaurusbase = pd.read_csv(FileLocThe, header=None, sep="\t")
    thesaurusbase.columns = ["subject", "predicate", "object", "weight"]
    ###############
    wordbase = thesaurusbase[thesaurusbase["predicate"] == "rel:means"].reset_index()
    basefile = wordbase["subject"].str.split("/", n=3, expand=True)
    wordbase["subject"] = basefile[2]
    wordbase["lang"] = basefile[1]
    basefile = wordbase["object"].str.split("/", n=2, expand=True)
    wordbase["object"] = basefile[1]
    ###############
    wordbase = wordbase[["subject", "predicate", "object", "weight", "lang"]]
    refbase = thesaurusbase[thesaurusbase["predicate"] == "rel:lexicalization"].reset_index()
    basefile = refbase["subject"].str.split("/", n=2, expand=True)
    refbase["subject"] = basefile[1]
    basefile = refbase["object"].str.split("/", n=3, expand=True)
    refbase["lang"] = basefile[1]
    refbase["object"] = basefile[2]
    refbase['pos'] = refbase['subject'].str[0]
    refbase = refbase[((refbase['pos'] == "n") | (refbase['pos'] == "v"))].reset_index()
    refbase = refbase[["subject", "object", "lang", 'pos']]
    refbase.columns = ["object", "word", "lang", 'pos']
    ###############
    thesaurus = pd.merge(wordbase, refbase, on=["object", "lang"], how="inner")
    thesaurus = thesaurus[thesaurus["subject"] != thesaurus["word"]]
    ###############
    if language == "en":
        thesaurus = thesaurus[(thesaurus['lang'] == 'eng')].reset_index()
    elif language == "ar":
        thesaurus = thesaurus[(thesaurus['lang'] == 'ara')].reset_index()
    elif language == "zh":
        thesaurus = thesaurus[(thesaurus['lang'] == 'zho')].reset_index()
    elif language == "id":
        thesaurus = thesaurus[(thesaurus['lang'] == 'ind')].reset_index()
    elif language == "ko":
        thesaurus = thesaurus[(thesaurus['lang'] == 'kor')].reset_index()
    elif language == "pt":
        thesaurus = thesaurus[(thesaurus['lang'] == 'por')].reset_index()
    elif language == "vn":
        thesaurus = thesaurus[(thesaurus['lang'] == 'vie')].reset_index()
    ########################


########
def emotechintentionmodel(SentenceToBe, synonym_num):
    ########
    # detector = Detector(SentenceToBe)
    # if detector.language.code == "vi":
    #     langthesaurusload("vn",FileLocThe)
    #     langmodelload("vn",LibLocLang)
    # else:
    #     langthesaurusload(detector.language.code,FileLocThe)
    #     langmodelload(detector.language.code,LibLocLang)
    # langthesaurusload("en", FileLocThe)
    # langmodelload("en",LibLocLang)
    ########
    intention_filters = ['PRON', "VERB"]
    object_filters = ['NOUN', 'PROPN']
    ########
    sentences = model.tokenize(SentenceToBe)
    ########
    for s in sentences:
        model.tag(s)  # inplace tagging
        model.parse(s)  # inplace parsing
    datause = pd.read_csv(StringIO(model.write(sentences, "conllu")), sep="\t", header=None, skiprows=4)
    PosTagIntention = datause[datause.columns[1:4]].values.tolist()
    print(PosTagIntention)
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
    ####
    if len(PosTagIntention) > 1:
        for i in range(0, len(PosTagIntention)):
            #####
            if i == 0:
                if any(str(word).lower() in str(PosTagIntention[i][0]).lower() for word in question_words):
                    sentence_intention.append("Question")
            #####
            else:
                if all(str(word).lower() != str(PosTagIntention[i][0]).lower() for word in stop_words):
                    if any(str(word).lower() in str(PosTagIntention[i][2]).lower() for word in intention_filters):
                        sentence_intention.append(PosTagIntention[i][0])
                    if any(str(word).lower() in str(PosTagIntention[i][2]).lower() for word in object_filters):
                        sentence_object.append(PosTagIntention[i][0])
        #####
        sentence_intention = list(set(sentence_intention))
        sentence_object = list(set(sentence_object))
    #####
    else:
        sentence_intention = []
        sentence_object = []
    #####
    intlength = len(sentence_intention)
    for i in range(0, intlength):
        temp = thesaurus[(thesaurus['subject'] == sentence_intention[i].lower()) & (thesaurus['weight'] >= 0.9)]
        # get top 2
        sorted_temp = temp.sort_values(by=['weight'], ascending=False)
        try:
            sorted_temp = sorted_temp[0:synonym_num]
        except IndexError:
            print("not enough items")
        if len(sorted_temp) > 1:
            sentence_intention.extend((sorted_temp['word']))
    #####
    intlength = len(sentence_object)
    for i in range(0, intlength):
        temp = thesaurus[(thesaurus['subject'] == sentence_object[i].lower()) & (thesaurus['weight'] >= 0.9)]
        # get top 2
        sorted_temp = temp.sort_values(by=['weight'], ascending=False)
        try:
            sorted_temp = sorted_temp[0:synonym_num]
        except IndexError:
            print("not enough items")
        if len(sorted_temp) > 1:
            sentence_object.extend((sorted_temp['word']))
    #####
    NERobj = Text(SentenceToBe, hint_language_code="en").entities
    #####
    sentence_intention = list(set(sentence_intention))
    sentence_object = list(set(sentence_object))
    #####
    overall = sentence_intention + sentence_object
    return (sentence_intention, sentence_object, overall, NERobj)


def count_occurance(search_list, cur_list):
    count = 0
    found_terms = set([item.lower() for item in search_list]).intersection(set([item.lower() for item in cur_list]))
    count = len(found_terms)
    return count


def find_terms(search_list, cur_list):
    found_terms = set([item.lower() for item in search_list]).intersection(set([item.lower() for item in cur_list]))
    # logger.info(found_terms)
    return list(found_terms)
