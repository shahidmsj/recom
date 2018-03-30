import json
from difflib import get_close_matches

data=json.load(open("data.json"))

def translate(w):
    w.lower()
    if w in data:
        return data[w]
    elif len(get_close_matches(w, data.keys())) > 0 :
        yn = input("Did you mean {}? type y for yes and n for no : " .format(get_close_matches(w, data.keys())[0]))
        if yn=="y":
            return data[get_close_matches(w, data.keys())[0]]
        elif yn=="n":
            return "This word is not present in my dictionary"
        else:
            return "Enter a proper word"
    else:
        return "The word doesn't exist."
    
    
    
    
word=input("Enter the word : ")
output=translate(word)

if type(output)==list:
    for i in output:
        print(i)
else:
    print(output)