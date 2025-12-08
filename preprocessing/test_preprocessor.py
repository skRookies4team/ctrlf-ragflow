from pipeline import (
    PreprocessPipeline,
    PreprocessPipelineV2,
    PreprocessPipelineUnified
)
import json

def test_all():
    pdf = "sample.pdf"

    print("\n================= V1 =================")
    v1 = PreprocessPipeline()
    r1 = v1.run(pdf)
    print(json.dumps(r1, ensure_ascii=False, indent=2))

    print("\n================= V2 =================")
    v2 = PreprocessPipelineV2()
    r2 = v2.run(pdf)
    print(json.dumps(r2, ensure_ascii=False, indent=2))

    print("\n================= Unified (BEST) =================")
    u = PreprocessPipelineUnified()
    r3 = u.run(pdf)
    print(json.dumps(r3, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_all()
