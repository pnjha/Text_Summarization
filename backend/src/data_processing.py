from Lang import *
from packages import *

def process_text(text,contraction_mapping,flag=False):
    stop_words = stopwords.words('english')
    text = text.lower()
    text = text.replace('\n','')
    text = re.sub(r'\(.*\)','',text)
    text = re.sub(r'[^a-zA-Z0-9. ]','',text)
    text = re.sub(r'\.',' . ',text)
    text = text.replace('.','')
    text = text.split()
    for i in range(len(text)):
        word = text[i]
        if word in contraction_mapping:
            text[i] = contraction_mapping[word]
    newtext = []
    for word in text:
        if word not in stop_words and len(word)>0:
            newtext.append(word)
    text = newtext
    if flag:
        text = text[::-1]
    text = " ".join(text)
    text = text.replace("'s",'') 
    return text

def prepareInput(sourcefilePath,destFilePath,fileName,contraction_mapping,DATA_SIZE):
    source = sourcefilePath
    target = destFilePath
    save_trans = fileName
    
    corpus_source = open(source, 'r').readlines()
    corpus_target = open(target, 'r').readlines()
  
    writer = open(save_trans, 'w')
    num_lines = 0
    for k, v in zip(corpus_source, corpus_target):
        k = process_text(k,contraction_mapping,True)
        v = process_text(v,contraction_mapping,False)
        writer.write(k + '\t' + v + '\n')
        num_lines += 1
        if num_lines > DATA_SIZE:
            break
    writer.flush()
    writer.close()

def parseLanguageInput(lang1,lang2,fullFilePath,SOS_token,EOS_token,UNK_token):

    lines = open(fullFilePath, encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]

    input_lang = Lang(lang1,SOS_token,EOS_token,UNK_token)
    output_lang = Lang(lang2,SOS_token,EOS_token,UNK_token)

    return input_lang, output_lang, pairs

def prepareData(lang1,lang2,fullFilePath,SOS_token,EOS_token,UNK_token):
    input_lang, output_lang, pairs = None,None,None
    input_lang, output_lang, pairs = parseLanguageInput(lang1,lang2,fullFilePath,SOS_token,EOS_token,UNK_token)

    for pair in pairs:
      input_lang.addSentence(pair[0])
      output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence, UNK_token):

    return [lang.word2index[word] if word in lang.word2index else UNK_token for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, device, EOS_token, UNK_token):
    indexes = indexesFromSentence(lang, sentence, UNK_token)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair,input_lang,output_lang, device, EOS_token, UNK_token):
    input_tensor = tensorFromSentence(input_lang, pair[0], device, EOS_token, UNK_token)
    target_tensor = tensorFromSentence(output_lang, pair[1], device, EOS_token, UNK_token)
    return (input_tensor, target_tensor)