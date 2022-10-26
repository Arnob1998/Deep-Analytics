import pandas as pd

class Sanitizer:

    def remove_escapes(self,text):
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        clean_text = text.translate(translator)
        return clean_text

    def remove_leading_whitespace(self,text):
        return text.lstrip(' ')

    def add_endofline(self,text, addspaceatend=True):
        last_char = text[len(text)-1]
        if  last_char != "." or last_char != "!" or last_char != "?":
            if addspaceatend:
                text = text + ". "
            else:
                text = text + "."
        return text

    def lang_filter(self,text):
        pass

    def anomalyfilter(self,text):
        pass

    def garbage_remover(self,text):
        pass

class AmazonDataPrep:
    def process(self,dataframe):
        dataframe["Ratings"] = dataframe["Ratings"].str.split(" ", expand=True)[0].astype(float)
        return dataframe[["Ratings","Title","Content"]]

class DataLoader:
    def load_default(self):
        return pd.read_csv("data/test_fixedv3.txt")
        
    def load_file(self,loc):
        return pd.read_csv(loc)