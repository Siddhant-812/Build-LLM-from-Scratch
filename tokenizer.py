import collections

def get_stats(ids):
    """
    Computes the frequency of adjacent pairs of token IDs.
    """
    counts = collections.defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    """
    Merges all occurrences of a specific pair of token IDs into a new token ID.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """
    A Byte-Pair Encoding (BPE) Tokenizer.
    """
    def __init__(self):
        """
        Initializes the tokenizer.
        """
        self.merges = {}
        self.vocab = {}


    def train(self, text, vocab_size, verbose=False):
        """
        Trains the BPE tokenizer on a given text.

        Args:
            text (str): The text corpus to train on.
            vocab_size (int): The desired final vocabulary size.
            verbose (bool): Whether to print progress updates.
        """
        # 1. Start with the character-level vocabulary
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        num_merges = vocab_size - 256 # The initial vocab is the 256 bytes

        # 2. Iteratively merge the most frequent pairs
        ids = list(tokens)
        self.merges = {}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")

        # 3. Create the final vocabulary
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]


    def encode(self, text):
        """
        Converts a string of text into a list of token IDs.
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # Nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        """
        Converts a list of token IDs back into a string.
        """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")
    
class CharacterTokenizer:
    """
    A simple character-level tokenizer.
    """
    def __init__(self, corpus):
        """
        Initializes the tokenizer and builds the vocabulary.

        Args:
            corpus (str): The text corpus to build the vocabulary from.
        """
        # Get all unique characters in the corpus
        self.chars = sorted(list(set(corpus)))
        self.vocab_size = len(self.chars)

        # Create the string-to-integer and integer-to-string mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        print("="*80)
        print(f"Vocabulary successfully created.")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Vocabulary: {''.join(self.chars)}")
        print("="*80)


    def encode(self, text):
        """
        Converts a string of text into a list of integers.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: The encoded text as a list of integers.
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids):
        """
        Converts a list of integers back into a string of text.

        Args:
            token_ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        return ''.join([self.itos[i] for i in token_ids])