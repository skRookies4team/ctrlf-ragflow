import json
from pipeline import PreprocessPipeline

def test_all():
    pdf = "sample.pdf"

    print("\n================= Preprocess Pipeline (V1) =================")
    pipeline = PreprocessPipeline()
    result = pipeline.run(pdf)

    # 기본 요약 출력
    print("\n===== RESULT SUMMARY =====")
    print("페이지 수:", result.get("page_count"))
    print("평균 품질:", result.get("avg_quality"))

    # 교정된 텍스트 미리보기
    print("\n===== FIRST PAGE PREVIEW =====")
    first_page = result["pages"][0]["text"]
    print(first_page[:500], "...\n")

    # JSON 저장 (원할 경우)
    with open("pipeline_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("JSON 저장 완료 → pipeline_output.json")

if __name__ == "__main__":
    test_all()
