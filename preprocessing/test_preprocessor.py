import json
from pipeline import PreprocessPipeline

def test_pipeline():
    sample_pdf = "sample.pdf"
    pipeline = PreprocessPipeline()

    result = pipeline.run(sample_pdf, chunk_size=1200)

    print("\n===== PIPELINE OUTPUT =====")
    print(json.dumps(result["result_json"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_pipeline()
