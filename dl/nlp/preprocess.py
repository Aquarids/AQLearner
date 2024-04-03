import numpy

sos = '<sos>'
eos = '<eos>'
pad = '<pad>'
unk = '<unk>'

pad_token = 0
sos_token = 1
eos_token = 2
unk_token = 3

language_en = 'en'
language_zh = 'zh'

def tokenize(text, language='en'):
    if language == 'en':
        return text.lower().split(' ')
    elif language == 'zh':
        return text.lower().split(' ') # 仅演示用
    else:
        raise ValueError('Unsupported language')

def build_vocab(tokenized_texts):
    vocab = {pad: pad_token, sos: sos_token, eos: eos_token, unk: unk_token}
    idx = 4  # 开始的索引
    for tokens in tokenized_texts:
        for token in tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab

def numericalize(tokenized_text, vocab):
    return [vocab.get(token, vocab[unk]) for token in tokenized_text]

def add_sos_eos(numericialzed_text):
    return [sos_token] + numericialzed_text + [eos_token]

def pad_sequence(numericialzed_text, max_length):
    return numericialzed_text + [pad_token] * (max_length - len(numericialzed_text))

def preprocess_texts(data):
    src_data_tokenized, tgt_data_tokenized = [tokenize(d[0], language_en) for d in data], [tokenize(d[1], language_zh) for d in data]
    src_vocab, tgt_vocab = build_vocab(src_data_tokenized), build_vocab(tgt_data_tokenized)

    index_to_src_word = {index: word for word, index in src_vocab.items()}
    index_to_tgt_word = {index: word for word, index in tgt_vocab.items()}

    print("Source Vocabulary:", src_vocab)
    print("Target Vocabulary:", tgt_vocab)

    max_length = 10
    src_numericalized = []
    tgt_numericalized = []
    for d in data:
        src_data = d[0]
        tgt_data = d[1]
        # print("Source:", src_data)
        # print("Target:", tgt_data)
        src_data_tokenized = tokenize(src_data, language_en)
        tgt_data_tokenized = tokenize(tgt_data, language_zh)
        # print("Source Tokenized:", src_data_tokenized)
        # print("Target Tokenized:", tgt_data_tokenized)
        src_data_numericalized = add_sos_eos(numericalize(src_data_tokenized, src_vocab))
        tgt_data_numericalized = add_sos_eos(numericalize(tgt_data_tokenized, tgt_vocab))
        # print("Source Numericalized:", src_data_numericalized)
        # print("Target Numericalized:", tgt_data_numericalized)
        src_data_numericalized = pad_sequence(src_data_numericalized, max_length)
        tgt_data_numericalized = pad_sequence(tgt_data_numericalized, max_length) 
        # print("Source Numericalized (Padded):", src_data_numericalized)
        # print("Target Numericalized (Padded):", tgt_data_numericalized)
        src_numericalized.append(src_data_numericalized)
        tgt_numericalized.append(tgt_data_numericalized)

    return src_numericalized, tgt_numericalized, src_vocab, tgt_vocab, index_to_src_word, index_to_tgt_word

def decode_sentence(indices, index_to_word, special_tokens={sos, eos, pad, unk}):
    words = [index_to_word.get(idx, unk) for idx in indices]
    if len(words) == 0 or words.count(unk) / len(words) > 0.5:
        print("Unknown words are more than 50%")
        return "unknown"
    filtered_words = [word for word in words if word not in special_tokens]
    if not filtered_words:
        print("No words left after filtering")
        return "unknown"
    return ' '.join(filtered_words)