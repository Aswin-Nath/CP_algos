def rabin_karp(text, pattern):
            n = len(text)
            m = len(pattern)
            if m > n:
                return []
            d = 256 
            q = 10**9 + 7  
            pattern_hash = 0
            text_hash = 0
            h = (d**(m-1))%q
            for i in range(m):
                pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
                text_hash = (d * text_hash + ord(text[i])) % q
            result = []
            for i in range(n - m + 1):
                if pattern_hash == text_hash:
                    result.append(i)
                if i < n - m:
                    text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % q
                    if text_hash < 0:
                        text_hash += q
            return result
