import nltk


def _remove_duplicates(caption):
    words = nltk.word_tokenize(caption)
    uwords = []
    prev = words[0]
    for w in words[1:]:
        if w != prev:
            uwords.append(w)
            prev = w
    return ' '.join(uwords)


def post_process(caption):
    return _remove_duplicates(caption)
