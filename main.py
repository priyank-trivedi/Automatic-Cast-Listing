import pysrt
import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer
import datetime

SEARCH_THRESHOLD = 5

def remove_utf_symbols(text):
    chars = {
    b'\xc2\x82' : b',',        # High code comma
    b'\xc2\x84' : b',,',       # High code double comma
    b'\xc2\x85' : b'...',      # Triple dot
    b'\xc2\x88' : b'^',        # High carat
    b'\xc2\x91' : b'\x27',     # Forward single quote
    b'\xc2\x92' : b'\x27',     # Reverse single quote
    b'\xc2\x93' : b'\x22',     # Forward double quote
    b'\xc2\x94' : b'\x22',     # Reverse double quote
    b'\xc2\x95' : b' ',
    b'\xc2\x96' : b'-',        # High hyphen
    b'\xc2\x97' : b'--',       # Double hyphen
    b'\xc2\x99' : b' ',
    b'\xc2\xa0' : b' ',
    b'\xc2\xa6' : b'|',        # Split vertical bar
    b'\xc2\xab' : b'<<',       # Double less than
    b'\xc2\xbb' : b'>>',       # Double greater than
    b'\xc2\xbc' : b'1/4',      # one quarter
    b'\xc2\xbd' : b'1/2',      # one half
    b'\xc2\xbe' : b'3/4',      # three quarters
    b'\xca\xbf' : b'\x27',     # c-single quote
    b'\xcc\xa8' : b'',         # modifier - under curve
    b'\xcc\xb1' : b''          # modifier - under line
}

    def replace_chars(match):
        char = match.group(0)
        return chars[char]

    return re.sub(b'|'.join(chars.keys()), replace_chars, text)

def process_subtitles():
    subs = pysrt.open("Friends-6x05 The One with Joey's Porsche.srt")
    subtitles = []
    for i in range(len(subs)):
        subtitles.append([subs[i].start.to_time(), subs[i].end.to_time(), subs[i].text.replace('\n', ' ')])
    return subtitles

def process_script():
    url = 'http://www.livesinabox.com/friends/season6/605towjp.htm'
    response = requests.get(url)
    html_doc = response.content
    soup = BeautifulSoup(html_doc, 'html.parser')
    script = []

    for i in soup.select("p > b:nth-of-type(1)"):
        text = ' '
        for j in i.next_siblings:
            text += str(j)
            text += ' '
        text = re.sub(r'\<[^>]*\>', '', text)
        text = text.encode('utf-8')
        text = remove_utf_symbols(text).decode('utf-8')
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = text.replace('\r','').replace('\n',' ')
        script.append([i.text[:-1],text.strip()])

    return script

def tokenize(lists,type):
    tokenizer = RegexpTokenizer(r'\w+')
    words = []
    if type==1:
        for i in lists:
            words.append(tokenizer.tokenize(i[2]))
    else:
        for i in lists:
            words.append(tokenizer.tokenize(i[1]))
    return [j for i in words for j in i]

def find_next_matching(subs_words,script_words,count_i,count_j):

    while True:
        if count_i+SEARCH_THRESHOLD > len(subs_words):
            i_end = len(subs_words)
        else:
            i_end = count_i + SEARCH_THRESHOLD

        if count_j+SEARCH_THRESHOLD > len(script_words):
            j_end = len(script_words)
        else:
            j_end = count_j + SEARCH_THRESHOLD

        for i in range(count_i,i_end):
            for j in range(count_j,j_end):
                if(subs_words[i]==script_words[j]):
                    return i,j

        if i_end==len(subs_words) or j_end == len(script_words):
            return len(subs_words)+1,len(script_words) + 1
        count_i+=SEARCH_THRESHOLD
        count_j+=SEARCH_THRESHOLD


def merge(subtitles, script, subs_words, script_words):
    final = []
    count_i = 0
    count_j = 0
    count_ii = 0
    count_jj = 0
    while True:
        if count_i >= len(subs_words):
            break
        temp = [subs_words[count_i]]
        if subtitles[count_ii][2].find(subs_words[count_i]) != -1:
            temp.append(subtitles[count_ii][0])
            temp.append(subtitles[count_ii][1])
        else:
            count_ii += 1
            temp.append(subtitles[count_ii][0])
            temp.append(subtitles[count_ii][1])
        count_i += 1
        final.append(temp)

    count_i = 0
    while True:
        if count_i > len(subs_words) or count_j > len(script_words):
            break
        if subs_words[count_i].lower() == script_words[count_j].lower():
            if script[count_jj][1].find(script_words[count_j]) != -1:
                final[count_i].append(script[count_jj][0])
            else:
                count_jj += 1
                final[count_i].append(script[count_jj][0])

            count_i += 1
            count_j += 1
        else:
            count_i, count_j = find_next_matching(subs_words, script_words, count_i, count_j)

    return final
def main():
    subtitles = process_subtitles()
    script = process_script()
    subs_words = tokenize(subtitles,1)
    script_words = tokenize(script,2)
    print(subs_words)
    print(script_words)
    final = merge(subtitles,script,subs_words,script_words)
    for i in range(len(final)):
        if len(final[i])>3:
            print(final[i][0],final[i][3])
        else:
            print(final[i][0])



if __name__ == '__main__':
    main()