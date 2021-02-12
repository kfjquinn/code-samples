
def strip_jeopardy_prefix(request):
    """ If the user responds with the correct Jeopardy prefix, remove it. """
    phrases = {'what is', 'what are', 'who is', 'who are'}
    if any(request.startswith(phrase) for phrase in phrases):
        return re.sub(Constants.JEOPARDY_PREFIX_PATTERN, '', request).strip()
    return request


def strip_a_an_the(request):
    """ If the request starts with common determiner, remove it. """
    phrases = {'a ', 'an ', 'the '}
    if any(request.startswith(phrase) for phrase in phrases):
        return re.sub('(a|an|the)\s', '', request).strip()
    return request


class AnswerActiveLearner(object):

    @staticmethod
    def selection(data, idx=0):
        return list(map(itemgetter(idx), data))

    @staticmethod
    def preprocess(data):
        """ Preprocess the text prior to passing through the model. """
        def instance(request, rewrite):
            return strip_a_an_the(strip_jeopardy_prefix(request)), strip_a_an_the(strip_jeopardy_prefix(rewrite))
        return list(starmap(instance, data))

    @staticmethod
    def featurize(data):
        def instance(request, rewrite):
            req_words = request.split()
            rew_words = rewrite.split()

            request_without_prefix = strip_jeopardy_prefix(request)
            one_word = len(req_words) == 1 and len(rew_words) == 1
            word_length_diff = abs(len(request) - len(rewrite))
            ld1 = Levenshtein.distance(request, rewrite)
            ld2 = Levenshtein.distance(request_without_prefix, rewrite)
            normalized_ld1 = ld1 / max(len(request), len(rewrite))
            normalized_ld2 = ld2 / max(len(request), len(rewrite))

            return [
                1 if (len(request) > 3 and ld1 == 1) else 0,
                1 if (len(request) > 6 and ld1 == 2) else 0,
                1 if (len(request) > 8 and ld1 == 3) else 0,
                1 if (len(request_without_prefix) > 3 and ld2 == 1) else 0,
                1 if (len(request_without_prefix) > 6 and ld2 == 2) else 0,
                1 if (len(request_without_prefix) > 8 and ld2 == 3) else 0,
                1 if (one_word and len(request) <= 4 and word_length_diff <= 3 and request[0:2] == rewrite[0:2]) else 0,
                1 if (one_word and len(request) <= 4 and word_length_diff <= 3 and request[-2:] == rewrite[-2:]) else 0,
                1 if (one_word and len(request) <= 6 and word_length_diff <= 3 and request[0:3] == rewrite[0:3]) else 0,
                1 if (one_word and len(request) <= 6 and word_length_diff <= 3 and request[-3:] == rewrite[-3:]) else 0,
                1 if (one_word and len(request) <= 8 and word_length_diff <= 3 and request[0:4] == rewrite[0:4]) else 0,
                1 if (one_word and len(request) <= 8 and word_length_diff <= 3 and request[-4:] == rewrite[-4:]) else 0,
                1 if (sorted(request) == sorted(rewrite)) else 0,
                1 if (sorted(request_without_prefix) == sorted(rewrite)) else 0,
                1 if ((request + 's') == rewrite) else 0,
                1 if (request == (rewrite + 's')) else 0,
                1 if ((request_without_prefix + 's') == rewrite) else 0,
                1 if (request_without_prefix == (rewrite + 's')) else 0,
                1 if (len(re.findall('({0} (what|who) (is|are) {1})'.format(rewrite, rewrite), request)) > 0) else 0,
                1 if (req_words[-1] == rewrite) else 0,
                ld1,
                ld2,
                normalized_ld1,
                normalized_ld2
            ]

        return list(starmap(instance, data))

    @staticmethod
    def pipeline():
        return Pipeline([
            ('union', FeatureUnion([
                ('1', Pipeline([
                    ('select', FunctionTransformer(AnswerActiveLearner.selection, validate=False, kw_args={'idx': 0})),
                    ('tfidf_vectorizer', TfidfVectorizer(
                        analyzer='char',
                        preprocessor=AnswerActiveLearner.preprocess,
                        lowercase=False,
                        ngram_range=(3,),
                        max_features=1000
                    ))
                ])),
                ('2', Pipeline([
                    ('select', FunctionTransformer(AnswerActiveLearner.selection, validate=False, kw_args={'idx': 1})),
                    ('tfidf_vectorizer', TfidfVectorizer(
                        analyzer='char',
                        preprocessor=AnswerActiveLearner.preprocess,
                        lowercase=False,
                        ngram_range=(3,),
                        max_features=1000
                    ))
                ])),
                ('3', Pipeline([
                    ('levenshtein', FunctionTransformer(AnswerActiveLearner.featurize, validate=False))
                ]))
            ])),
            ('model', SVC(kernel='linear', C=1.0, probability=True))
        ])

