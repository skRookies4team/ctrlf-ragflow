from .cer import cer

def compute_cer_multi(pred_list, gt_list):
    if not pred_list or not gt_list:
        return None

    page_cers = []

    for pred, gt in zip(pred_list, gt_list):
        pred = pred if isinstance(pred, str) else ""
        gt = gt if isinstance(gt, str) else ""

        # 둘 다 비어있으면 CER 계산 의미 없음
        if not pred.strip() or not gt.strip():
            continue
        
        # 페이지 단위로 계산
        try:
            score = cer(pred, gt)
            page_cers.append(score)
        except Exception:
            continue

    if not page_cers:
        return None

    # 전체 평균 CER
    return sum(page_cers) / len(page_cers)
