# preprocessing/ocr/cer.py

def cer(pred: str, truth: str) -> float:
    """
    CER = (삽입 + 삭제 + 교체) / 정답 글자 수
    edit distance 기반
    """
    import numpy as np

    p = list(pred)
    t = list(truth)

    dp = np.zeros((len(t)+1, len(p)+1), dtype=int)

    for i in range(len(t)+1):
        dp[i][0] = i
    for j in range(len(p)+1):
        dp[0][j] = j

    for i in range(1, len(t)+1):
        for j in range(1, len(p)+1):
            cost = 0 if t[i-1] == p[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[len(t)][len(p)] / max(1, len(t))
