import numpy as np
np.set_printoptions(suppress=True)

class Paraphraser:

    model_name = "Pegasus-Parapharser"
    base_name = 'tuner007/pegasus_paraphrase'
    transformation = "Paraphraser"
    description = "Sentences are changed by adding new words and phrases without altering their meaning. (Token Length may vary)"

    def rephrase(self,text,max_length=None):
        return None

class Summmerizer:

    model_name = "BART-Summerizer"
    base_name = "facebook/bart-large-cnn"
    transformation = "Summmerizer"
    description = "Condenses the paragraph by trimming unnecessary sentences and occasionally adding new sentences."

    def rephrase(self,text,max_length=None):
        return None

class XSummerizer:

    model_name = "Pegasus-Xtrem_Summerizer"
    base_name = "google/pegasus-xsum"
    transformation = "XSummmerizer"
    description = "Condenses the paragraph to a single sentence. This an extremely destructive techniques which may result in loss of important information."

    def rephrase(self,text,max_length=None):
        return None
        