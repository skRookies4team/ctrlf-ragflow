#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding 모델 평가 보고서 생성

모든 모델의 결과를 비교하고 최종 보고서를 생성합니다.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
REPORT_DIR = EVAL_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def load_all_results() -> List[Dict]:
    """모든 결과 파일 로드"""
    results = []
    for json_file in RESULTS_DIR.glob("*_result.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            results.append(json.load(f))
    return results


def generate_markdown_report(results: List[Dict]) -> str:
    """마크다운 형식 보고서 생성"""

    report = []
    report.append("# Embedding 모델 성능 평가 보고서\n")
    report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 요약
    report.append("## 1. 평가 개요\n")
    report.append("### 테스트 모델")
    report.append("| # | 모델명 |")
    report.append("|---|--------|")
    for i, r in enumerate(results, 1):
        model_name = r['model_name'].split('/')[-1] if '/' in r['model_name'] else r['model_name']
        report.append(f"| {i} | {model_name} |")
    report.append("")

    report.append("### 평가 지표")
    report.append("- **Precision@K**: 상위 K개 검색 결과 중 관련 문서 비율")
    report.append("- **Recall@K**: 관련 문서 중 상위 K개에 포함된 비율")
    report.append("- **MRR**: 첫 번째 관련 문서 순위의 역수 평균")
    report.append("- **NDCG@5**: 순위 가중치를 적용한 검색 품질")
    report.append("- **Hit Rate@K**: 상위 K개에 관련 문서가 1개 이상 포함된 비율")
    report.append("- **Latency**: 임베딩 생성 평균 시간 (ms)\n")

    # 2. 결과 비교 테이블
    report.append("## 2. 성능 비교\n")
    report.append("### 정확도 지표 (Precision)")
    report.append("| 모델 | P@1 | P@3 | P@5 |")
    report.append("|------|-----|-----|-----|")
    for r in results:
        name = r['model_name'].split('/')[-1][:20]
        report.append(f"| {name} | {r['precision_at_1']:.4f} | {r['precision_at_3']:.4f} | {r['precision_at_5']:.4f} |")
    report.append("")

    report.append("### 재현율 지표 (Recall)")
    report.append("| 모델 | R@1 | R@3 | R@5 |")
    report.append("|------|-----|-----|-----|")
    for r in results:
        name = r['model_name'].split('/')[-1][:20]
        report.append(f"| {name} | {r['recall_at_1']:.4f} | {r['recall_at_3']:.4f} | {r['recall_at_5']:.4f} |")
    report.append("")

    report.append("### 순위 및 Hit Rate 지표")
    report.append("| 모델 | MRR | NDCG@5 | Hit@1 | Hit@3 | Hit@5 |")
    report.append("|------|-----|--------|-------|-------|-------|")
    for r in results:
        name = r['model_name'].split('/')[-1][:20]
        report.append(f"| {name} | {r['mrr']:.4f} | {r['ndcg_at_5']:.4f} | {r['hit_rate_at_1']:.4f} | {r['hit_rate_at_3']:.4f} | {r['hit_rate_at_5']:.4f} |")
    report.append("")

    report.append("### 성능 지표 (Latency)")
    report.append("| 모델 | 평균 Latency (ms) |")
    report.append("|------|-------------------|")
    for r in results:
        name = r['model_name'].split('/')[-1][:20]
        report.append(f"| {name} | {r['avg_latency_ms']:.2f} |")
    report.append("")

    # 3. 종합 순위
    report.append("## 3. 종합 분석\n")

    # MRR 기준 순위
    sorted_by_mrr = sorted(results, key=lambda x: x['mrr'], reverse=True)
    report.append("### MRR 기준 순위")
    report.append("| 순위 | 모델 | MRR |")
    report.append("|------|------|-----|")
    for i, r in enumerate(sorted_by_mrr, 1):
        name = r['model_name'].split('/')[-1][:25]
        report.append(f"| {i} | {name} | {r['mrr']:.4f} |")
    report.append("")

    # Hit@5 기준 순위
    sorted_by_hit5 = sorted(results, key=lambda x: x['hit_rate_at_5'], reverse=True)
    report.append("### Hit Rate@5 기준 순위")
    report.append("| 순위 | 모델 | Hit@5 |")
    report.append("|------|------|-------|")
    for i, r in enumerate(sorted_by_hit5, 1):
        name = r['model_name'].split('/')[-1][:25]
        report.append(f"| {i} | {name} | {r['hit_rate_at_5']:.4f} |")
    report.append("")

    # Latency 기준 순위
    sorted_by_latency = sorted(results, key=lambda x: x['avg_latency_ms'])
    report.append("### Latency 기준 순위 (빠른 순)")
    report.append("| 순위 | 모델 | Latency (ms) |")
    report.append("|------|------|--------------|")
    for i, r in enumerate(sorted_by_latency, 1):
        name = r['model_name'].split('/')[-1][:25]
        report.append(f"| {i} | {name} | {r['avg_latency_ms']:.2f} |")
    report.append("")

    # 4. 결론
    report.append("## 4. 결론 및 권장사항\n")

    if results:
        best_mrr = sorted_by_mrr[0]
        best_hit = sorted_by_hit5[0]
        fastest = sorted_by_latency[0]

        report.append("### 최고 성능 모델")
        report.append(f"- **정확도 (MRR) 최고**: {best_mrr['model_name'].split('/')[-1]} (MRR: {best_mrr['mrr']:.4f})")
        report.append(f"- **검색 성공률 (Hit@5) 최고**: {best_hit['model_name'].split('/')[-1]} (Hit@5: {best_hit['hit_rate_at_5']:.4f})")
        report.append(f"- **속도 최고**: {fastest['model_name'].split('/')[-1]} (Latency: {fastest['avg_latency_ms']:.2f}ms)")
        report.append("")

        # 종합 점수 계산 (정규화)
        report.append("### 종합 점수 (가중 평균)")
        report.append("```")
        report.append("종합 점수 = MRR × 0.3 + Hit@5 × 0.3 + NDCG@5 × 0.2 + (1 - 정규화Latency) × 0.2")
        report.append("```")
        report.append("")

        # 종합 점수 계산
        max_latency = max(r['avg_latency_ms'] for r in results)
        for r in results:
            norm_latency = r['avg_latency_ms'] / max_latency if max_latency > 0 else 0
            r['composite_score'] = (
                r['mrr'] * 0.3 +
                r['hit_rate_at_5'] * 0.3 +
                r['ndcg_at_5'] * 0.2 +
                (1 - norm_latency) * 0.2
            )

        sorted_composite = sorted(results, key=lambda x: x['composite_score'], reverse=True)
        report.append("| 순위 | 모델 | 종합 점수 |")
        report.append("|------|------|-----------|")
        for i, r in enumerate(sorted_composite, 1):
            name = r['model_name'].split('/')[-1][:25]
            report.append(f"| {i} | {name} | {r['composite_score']:.4f} |")
        report.append("")

        report.append(f"### 최종 권장 모델: **{sorted_composite[0]['model_name'].split('/')[-1]}**")

    return "\n".join(report)


def main():
    print("=" * 60)
    print("  Embedding 모델 평가 보고서 생성")
    print("=" * 60)

    # 결과 로드
    print("\n결과 파일 로드 중...")
    results = load_all_results()

    if not results:
        print("결과 파일이 없습니다. 먼저 embedding_eval.py를 실행하세요.")
        return

    print(f"  {len(results)}개 모델 결과 로드")

    # 보고서 생성
    print("\n보고서 생성 중...")
    report = generate_markdown_report(results)

    # 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORT_DIR / f"embedding_eval_report_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n보고서 저장 완료: {report_path}")

    # 콘솔에도 출력
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
