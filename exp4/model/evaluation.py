from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider


# 计算BLEU-4指标
def calculate_bleu_4(generated, reference):
    # 4-gram权重
    weights = [0.25, 0.25, 0.25, 0.25]

    # 使用加一平滑
    smoothing_function = SmoothingFunction().method1

    score = sentence_bleu([reference], generated, weights=weights, smoothing_function=smoothing_function)
    return score


# 计算ROUGE-2指标
def calculate_rouge_2(generated, reference):
    scorer = rouge_scorer.RougeScorer(['rouge2'])
    scores = scorer.score(' '.join(generated), ' '.join(reference))
    score = scores['rouge2'][0]
    return score


# 计算CIDEr指标
def calculate_cider(generated, reference):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts={0: [' '.join(reference)]}, res={0: [' '.join(generated)]})
    return score
