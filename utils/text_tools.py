# -*- coding: utf-8 -*-
import re
import sys


abbreviations = ['a.','b.','c.','d.','e.','f.','g.','h.','i.','j.','k.','l.','m.','n.','o.','p.','q.','r.','s.','t.','u.','v.','x.','z.','y.','w.',\
'av.','ed.','ex.','sr.','dr.','etc.','me.','fls.','fig.','gov.','loc.','jr.','min.','max.','máx.','med.','méd.','hab.','pe.','pr.','pg.','dep.','adj.','inc.',\
'vs.','vol.','vel.','un.','séc.','prod.','prof.','rod.','sto.','sta.','cel.','neg.','doc.','obs.','dic.','bel.','num.','adm.','cia.','art.','cód.','end.','pág.','pra.','sra.','eng.',\
'pac.','pct.','trav.','lot.','apto.','ltda.','a.c.','d.c.','ph.d.','s.a.','w.c.','rg.','t.b.']

def single_tokenizer(sent):
    return sent.split()


def split_level_one(text):
    #simplifica sequencias estranhas (!!!...) (......) (???...) ()
    text = re.sub(r"\?+", "?", text)
    text = re.sub(r"\!+", "!", text)
    text = re.sub(r"\.+", ".", text)
    #replace (…) -> ( ), (?) -> ( ? ), (!) -> ( ! ),
    text = text.replace("…", " ").replace("?", " ? ").replace("!", " ! ").replace("*"," * ").replace("\""," \" ").replace("("," ( ").replace(")"," ) ")\
    .replace("["," [ ").replace("]"," ] ").replace("“"," “ ").replace("”"," ” ").replace("‘"," ‘ ").replace("’"," ’ ").replace('R$',' R$ ').replace("'"," ' ")
    #afasta se não for números (word1,word2) (word1 ,word2) (word1, word2)
    text = re.sub(r"\,([^\d+\,\d+])", r" , \1", text)
    #afasta se não for números (word1:word2) (word1: word2) (word1 :word2)
    text = re.sub(r"\;([^\d+\;\d+])", r" ; \1", text)
    #normalizar para um espaço em branco    ,.9o l
    text = re.sub(r"\s+", " ", text)
    return text

def split_level_two(text):
    words_list = []
    parts = text.split()
    for word in parts:
        if ('/' in word) or (':' in word) or ('.' in word):
            match = re.search(r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', word)
            if not match:
                #afasta palavras de : quando nao é numeros
                word = re.sub(r"\:([^\d+\:\d+])", r" : \1", word+" ").strip()
                if word.lower() not in abbreviations:
                    if word.endswith('.'):
                        #afasta se não for números (word1.word2) (word1. word2) (word1 .word2)
                        word = word.replace('.',' . ')
                if '/' in word:
                    if not re.search(r'[0-9]{1,20}/[0-9]{1,20}',word):
                        word = word.replace('/',' / ')                
                parts_aux = word.split()
                for element in parts_aux:
                    words_list.append(element)
            else:
                if word.strip().endswith(':'):
                    word = word.replace(':',' : ')
                    parts_aux = word.split()
                    for element in parts_aux:
                        words_list.append(element)
                else:
                    words_list.append(word)
        else:
            words_list.append(word)
            
    return words_list

def tokenize_pt(text):
    """
    Tokenizes Portuguese text from a string into a list of strings
    """
    #primeiros padrões, separação de palavra de [. , ? ! ( ) [ ] : ; ' ' " " ]
    return split_level_two(split_level_one(text))

def tokenize_tag(text):
    """
    Tokenizes POSTagger text from a string into a list of strings
    """
    return [tok for tok in single_tokenizer(text)]