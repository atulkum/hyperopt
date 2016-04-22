from bs4 import BeautifulSoup             
from nltk.corpus import stopwords 	
from nltk.stem.porter import PorterStemmer
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#import enchant
import query_spell_check as sc

#d = enchant.DictWithPWL("en_US","metadata/dictionary")
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def fixspell(raw_text):
	if raw_text in sc.spell_check_dict:
		return sc.spell_check_dict[raw_text]
	else:
		return raw_text
	"""
	letters_only = re.sub("[^a-zA-Z0-9]", " ", raw_text) 
	words =[]
	for w in letters_only.lower().split():           
		if not d.check(w) and d.suggest(w):
			words.append(d.suggest(w)[0])
		else:
			words.append(w)

	return( " ".join(words )) 
	"""
	
def do_stem(raw_text):
	words = raw_text.lower().split()                
	words = [stemmer.stem(w) for w in words]
	return( " ".join(words ))  

def do_lemma(raw_text):
	words = raw_text.lower().split()                
	words_pos = pos_tag(words)
	words = [lemmatiser.lemmatize(w, pos=get_wordnet_pos(p)) \
		if p=='' else lemmatiser.lemmatize(w) for w, p in words_pos]
	return( " ".join(words ))  

def do_stopWord(raw_text):
	words = raw_text.lower().split()                
	stops = set(stopwords.words("english"))                  
	words = [w for w in words if not w in stops]
	return( " ".join(words ))  

def do_clean(raw_desc):
	desc_text = BeautifulSoup(raw_desc, 'lxml').get_text() 
	words = re.sub("[^a-zA-Z0-9]", " ", desc_text) 
	return( " ".join(words ))  

