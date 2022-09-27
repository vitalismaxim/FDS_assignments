from nltk.corpus import stopwords


sw = stopwords.words("english")
sw += ["â", "see", "going", "u", "thank", ',',
                          "you", "itâ", "s", "well", "us", "weâ", 'country', 'countries', 'state','world','people',
                          "will", "continue", 'hello', 'good afternoon', 'afternoon','support',
                          "now", "re", 'thank you', 'thanks', 'thank', 'thanks', 'good morning', 'morning',
              'mr.', 'mr', 'president', 'secretary', 'thank', 'thanks', 'you', 'mrs', 'mrs.', 'united', 'nations', 'un', 'must', 'also']