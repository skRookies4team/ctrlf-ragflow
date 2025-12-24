import json
import logging
logger = logging.getLogger(__name__)

class JSONSerializer:
    @staticmethod
    def to_chunk_json(chunks, meta=None):
        return json.dumps({
            "meta": meta or {},
            "chunks": [{"id": i, "content": c} for i, c in enumerate(chunks)]
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def save_chunks(chunks):
        json_str = JSONSerializer.to_chunk_json(chunks)
        data = json.loads(json_str)
        return data["chunks"]
