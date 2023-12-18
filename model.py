import math, random, re

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * c

def ngrams(c, text):
    padded_text = start_pad(c) + text
    
    list_of_ngrams = []

    for i in range(c, len(padded_text)):
        context = padded_text[i-c:i] # gets previous 2 letters
        character = padded_text[i]
        tuple_ = context, character 
        list_of_ngrams.append(tuple_)
        ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    
    return list_of_ngrams

    
def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.c = c
        self.k = k
        self.character_set = set()
        self.list_of_ngrams = [] # contains dupes
        self.character_count_dict = {"~": self.c}
        self.context_count_dict = {}
        self.ngrams_count_dict = {}

        # initializes any necessary internal variables

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        # return self.character_count_dict.keys()
        return self.character_set
    
    def get_ngram_count(self):
        return self.ngrams_count_dict


    def get_ngrams(self):
        return self.list_of_ngrams
        

    def update(self, text):
        
        # maintaining character counts
        # i grab character counts before adding the padding, to have accurate counts for spaces
        for letter in text:
            if letter in self.character_count_dict:
                self.character_count_dict[letter] += 1
            else:
                 self.character_count_dict[letter] = 1



        [self.character_set.add(letter) for letter in text] # adds letter to the character_set



        # returns a list of tuples
        new_list_of_ngrams = ngrams(self.c, text) 

        # adds them to the back of the list
        self.list_of_ngrams.extend(new_list_of_ngrams) 

        # Updating the context counts in dictionary
        for ngram in new_list_of_ngrams:

            # keeping ngram counts ("context", 'char')
            if ngram in self.ngrams_count_dict:
                self.ngrams_count_dict[ngram] += 1
            else:
              self.ngrams_count_dict[ngram] = 1

            # keeping context counts
            context = ngram[0]
            if context in self.context_count_dict:
                self.context_count_dict[context] += 1
            else:
                self.context_count_dict[context] = 1



    
    def prob(self, context, char):
        ''' Returns the probability of char appearing after context without smoothing
        '''
        # if context not in training data, we return 1/V
        if context not in self.context_count_dict:
            return 1 / len(self.character_set)

        # find number of occurrences of the context and character combination
        full_context_tuple = context, char
        num_occ_context = self.ngrams_count_dict.get(full_context_tuple, 0)

        # get the count of the context
        context_count = self.context_count_dict.get(context, 0)

        # calculate probability without smoothing
        if context_count == 0:
            return 0
        else:
            return num_occ_context / context_count

        
        
    def random_char(self, context):
        random_num = random.random() # returns a number between 0 and 1
        sum_of_probabilities = 0

        sorted_vocab = sorted(self.character_set)
        for char in sorted_vocab:

            prob = self.prob(context, char)
            sum_of_probabilities += prob
            
            if sum_of_probabilities > random_num:
                return char
        

        return random.choice(sorted_vocab) if sorted_vocab else None

        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''


    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.c) # returns '~' * c
        string_builder = ""
        for i in range(0,length):
            char = self.random_char(context)

            string_builder += char
            context = context[1:] + char
        return string_builder

        

    def perplexity(self, text):
        ngrams_list = ngrams(self.c, text)
        perplexity=1
        for i in range(1, len(ngrams_list)):
            ngram = ngrams_list[i]
            probability = self.prob(ngram[0], ngram[1])
            if probability == 0:
                return float('inf')
            
            

            one_over_probability = 1 / probability
            perplexity *= one_over_probability

        return (perplexity** ( 1 / len( ngrams_list)))




class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c, k)
        self.models = [NgramModel(i, k) for i in range(0, c + 1)]
        
        lambdas = [1 / (c + 1) for _ in range(c + 1)]
        self.lambdas = lambdas

    def update(self, text):
        for model in self.models:
            model.update(text)

    def prob(self, context, char):
        ''' Returns the interpolated probability of char appearing after context
        '''
        interpolated_prob = 0
        for i, model in enumerate(self.models):
            sub_context = context[-model.c:] if model.c > 0 else ''
            prob = model.prob(sub_context, char)
            interpolated_prob += self.lambdas[i] * prob

        return interpolated_prob

    def random_char(self, context):
        random_num = random.random()
        sum_of_probabilities = 0

        all_chars = set(char for model in self.models for char in model.get_vocab())
        sorted_vocab = sorted(all_chars)

        for char in sorted_vocab:
            prob = self.prob(context, char)
            sum_of_probabilities += prob

            if sum_of_probabilities > random_num:
                return char

        return random.choice(sorted_vocab) if sorted_vocab else None

    def perplexity(self, text):
        ngrams_list = ngrams(self.c, text)
        log_perplexity = 0
        for ngram in ngrams_list:
            probability = self.prob(ngram[0], ngram[1])
            if probability > 0:
                log_perplexity += math.log(probability)
            else:
                return float('inf')

        return math.exp(-log_perplexity / len(ngrams_list))


