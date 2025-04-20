import re

def is_signed_numeric(s):
    return bool(re.fullmatch(r'-?\d+', s))

print(is_signed_numeric("123"))     
print(is_signed_numeric("-123"))    
print(is_signed_numeric("--123"))   
print(is_signed_numeric("12-3"))    
print(is_signed_numeric("abc"))     

import pickle

print(pickle.load(open("arithmetic_results.pkl", "rb")))