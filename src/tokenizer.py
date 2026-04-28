"""
SMILES-Aware Tokenizer for Deep Learning on Molecular Structures

Design Philosophy:
- Pattern-based tokenization (NOT BPE/WordPiece) to preserve chemical syntax
- Cl, Br treated as single tokens (standard chemistry notation)
- Ring numbers, brackets, operators as individual tokens
- No [UNK] tokens for valid SMILES if vocab is properly initialized
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILESTokenizer:
    """
    Pattern-based SMILES tokenizer that preserves chemical structure.
    
    Token categories:
    1. Multi-char atoms: Cl, Br, [nH], [NH3+], etc.
    2. Ring numbers: 1-9, %10-%99
    3. Operators: =, #, (, ), [, ], -, /, \\, @, @@
    4. Single atoms: C, N, O, S, P, F, I, B, c, n, o, s, p
    5. Digits: 0-9 (for ring closures and atom labels)
    """
    
    # Define token patterns in order of priority (longest match first)
    TOKEN_PATTERNS = [
        # Special tokens
        r'\[CLS\]',
        r'\[PAD\]',
        r'\[EOS\]',
        r'\[UNK\]',
        
        # Charged atoms in brackets [nH], [NH3+], [OH-], etc.
        r'\[[^\]]*\]',  # Catch-all for bracketed atoms
        
        # Multi-char atoms (must come before single chars)
        r'Cl',
        r'Br',
        
        # Ring numbers with % prefix (%10-%99)
        r'%\d{2}',
        
        # Single-char atoms (aromatic and aliphatic)
        r'[CNOSPFIBBrcnosp]',
        
        # Ring numbers (1-9)
        r'[1-9]',
        
        # Operators and bonds
        r'[=#@/\\()-]',
        
        # Stereochemistry markers
        r'[@]{1,2}',
        
        # Catch any remaining single char
        r'.',
    ]
    
    def __init__(self, vocab_size: int = 100, max_length: int = 256):
        """
        Initialize tokenizer with empty vocabulary.
        Call build_vocab() after initialization to populate vocab from SMILES.
        
        Args:
            vocab_size: Expected vocabulary size (for pre-allocation)
            max_length: Maximum sequence length for padding/truncation
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize with special tokens
        self.token2id = {
            '[CLS]': 0,
            '[PAD]': 1,
            '[EOS]': 2,
            '[UNK]': 3,
        }
        
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.next_token_id = 4
        
        # Compile regex pattern for efficient tokenization
        self.pattern = re.compile('|'.join(self.TOKEN_PATTERNS))
    
    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string using pattern matching.
        
        Args:
            smiles: SMILES string (e.g., "CCO" for ethanol)
            
        Returns:
            List of token strings
        """
        tokens = self.pattern.findall(smiles)
        return tokens
    
    def encode(self, smiles: str, add_special_tokens: bool = True, 
               padding: bool = True, truncation: bool = True) -> Dict[str, List[int]]:
        """
        Convert SMILES to token IDs with padding/truncation.
        
        Args:
            smiles: SMILES string
            add_special_tokens: Add [CLS] at start and [EOS] at end
            padding: Pad to max_length with [PAD]
            truncation: Truncate to max_length if needed
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        tokens = self.tokenize(smiles)
        
        # Add special tokens
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[EOS]']
        
        # Truncate if needed
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to IDs
        input_ids = []
        for token in tokens:
            if token in self.token2id:
                input_ids.append(self.token2id[token])
            else:
                # Assign new ID if token not in vocab
                input_ids.append(self.token2id['[UNK]'])
                logger.warning(f"Token '{token}' not in vocabulary, using [UNK]")
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        if padding and len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids.extend([self.token2id['[PAD]']] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
    def decode(self, input_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to SMILES string.
        
        Args:
            input_ids: List of token IDs
            skip_special_tokens: Whether to skip [CLS], [PAD], [EOS], [UNK]
            
        Returns:
            Reconstructed SMILES string (approximately, may have spaces)
        """
        tokens = []
        for token_id in input_ids:
            if token_id in self.id2token:
                token = self.id2token[token_id]
                if skip_special_tokens and token in ['[CLS]', '[PAD]', '[EOS]', '[UNK]']:
                    continue
                tokens.append(token)
        
        return ''.join(tokens)
    
    def add_token(self, token: str) -> int:
        """
        Add a new token to vocabulary and return its ID.
        
        Args:
            token: Token string to add
            
        Returns:
            Token ID
        """
        if token not in self.token2id:
            token_id = self.next_token_id
            self.token2id[token] = token_id
            self.id2token[token_id] = token
            self.next_token_id += 1
            return token_id
        return self.token2id[token]
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 1):
        """
        Build vocabulary from a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum frequency for a token to be included
        """
        token_freq = {}
        
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Add frequent tokens to vocabulary
        for token, freq in sorted(token_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq and token not in self.token2id:
                self.add_token(token)
        
        logger.info(f"Built vocabulary with {len(self.token2id)} tokens")
        return token_freq
    
    def save(self, path: str):
        """Save tokenizer vocabulary to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        vocab_data = {
            'token2id': self.token2id,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'max_length': self.max_length,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer vocabulary from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token2id = vocab_data['token2id']
        self.id2token = {int(k): v for k, v in vocab_data['id2token'].items()}
        self.max_length = vocab_data.get('max_length', 256)
        self.next_token_id = max(self.id2token.keys()) + 1 if self.id2token else 4
        logger.info(f"Tokenizer loaded from {path} with {len(self.token2id)} tokens")
    
    def __call__(self, smiles: str, **kwargs) -> Dict[str, List[int]]:
        """Allow tokenizer to be called as a function."""
        return self.encode(smiles, **kwargs)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token2id)


# ============================================================================
# Unit Tests for SMILES Tokenization
# ============================================================================

def test_tokenizer():
    """Test SMILES tokenizer on known molecules."""
    tokenizer = SMILESTokenizer()
    
    # Test cases: (SMILES, expected_tokens)
    test_cases = [
        ("CCO", ["C", "C", "O"]),  # Ethanol
        ("c1ccccc1", ["c", "1", "c", "c", "c", "c", "c", "1"]),  # Benzene
        ("CCl", ["C", "Cl"]),  # Chloromethane - Cl as single token
        ("CBr", ["C", "Br"]),  # Bromomethane - Br as single token
        ("C1CCCCC1", ["C", "1", "C", "C", "C", "C", "C", "1"]),  # Cyclohexane
        ("C=C", ["C", "=", "C"]),  # Ethene
        ("C#C", ["C", "#", "C"]),  # Ethyne
        ("C(=O)O", ["C", "(", "=", "O", ")", "O"]),  # Carboxylic acid group
        ("[nH]", ["[nH]"]),  # Pyrrole-like nitrogen
        ("[NH3+]", ["[NH3+]"]),  # Ammonium
    ]
    
    print("\n" + "="*60)
    print("SMILES TOKENIZER TEST")
    print("="*60)
    
    for smiles, expected in test_cases:
        tokens = tokenizer.tokenize(smiles)
        match = tokens == expected
        status = "PASS" if match else "FAIL"
        print(f"{status}: {smiles}")
        if not match:
            print(f"  Expected: {expected}")
            print(f"  Got:      {tokens}")
    
    print("="*60 + "\n")


def test_vocab_building():
    """Test vocabulary building from SMILES list."""
    smiles_list = [
        "CCCCCCCCCCOC(CCCN(C(=O)CCCCC(=O)OC(CCCCCC)CCCCCC)C(C(=O)NCCCCN1CCCCC1CC)C(CCC)CCC)OCCCCCCCCCC",
        "CCl",
        "CBr",
        "c1ccccc1",
    ]
    
    tokenizer = SMILESTokenizer()
    token_freq = tokenizer.build_vocab(smiles_list)
    
    print("\n" + "="*60)
    print("VOCABULARY BUILDING TEST")
    print("="*60)
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Top 20 tokens by frequency:")
    for token, freq in sorted(token_freq.items(), key=lambda x: -x[1])[:20]:
        print(f"  {token}: {freq}")
    print("="*60 + "\n")


def test_encode_decode():
    """Test encoding and decoding round-trip."""
    tokenizer = SMILESTokenizer()
    smiles = "CCO"
    
    # Encode
    encoded = tokenizer.encode(smiles)
    print(f"\nOriginal SMILES: {smiles}")
    print(f"Encoded input_ids: {encoded['input_ids'][:20]}")  # Show first 20
    print(f"Attention mask: {encoded['attention_mask'][:20]}")
    
    # Decode
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded SMILES: {decoded}")
    print()


if __name__ == "__main__":
    test_tokenizer()
    test_vocab_building()
    test_encode_decode()
