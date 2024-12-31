def computeLPSArray(pattern):
    length = 0
    lps = [0] * len(pattern)
    idx = 1
    while idx < len(pattern):
        if pattern[idx] == pattern[length]:
            length += 1
            lps[idx] = length
            idx += 1
        else:
            if length:
                length = lps[length - 1]
            else:
                lps[idx] = 0
                idx += 1
    return lps
def knuthMorrisPrattAlgo(pattern,text):
    res = []
    lps = computeLPSArray(pattern)
    idxText, idxPattern = 0, 0
    while idxText < len(text):
        if pattern[idxPattern] == text[idxText]:
            idxPattern += 1
            idxText += 1
        if idxPattern == len(pattern):
            res.append(idxText - idxPattern)
            idxPattern = lps[idxPattern - 1]
        elif idxText < len(text) and pattern[idxPattern] != text[idxText]:
            if idxPattern:
                idxPattern = lps[idxPattern - 1]
            else:
                idxText += 1
    return res
