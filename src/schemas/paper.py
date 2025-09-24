from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Literal, TypedDict

from src.modules.utils import save_result


class PaperDict(TypedDict):
    title: str = ""
    abstract: str = ""
    md_text: str = ""


@dataclass
class Paper:
    title: str = ""
    abstract: str = ""
    md_text: str = ""
    paper_type: Literal["method", "theory", "benchmark", "survey"] = ""
    attri: dict | None = None
    bib_name: str = ""
    mount_outline: list | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        save_result(json.dumps(self.to_dict(), indent=4), path)

    @staticmethod
    def from_json(path: str) -> Paper:
        dic = json.load(open(path, "r", encoding="utf-8"))
        assert {
            "title",
            "abstract",
            "md_text",
            "paper_type",
            "attri",
            "mount_outline",
        }.issubset(dic)
        return Paper(
            **{k: v for k, v in dic.items() if k in Paper.__dataclass_fields__}
        )

    @staticmethod
    def from_dir(dir_path: str) -> list[Paper]:
        return [
            Paper.from_json(os.path.join(dir_path, file))
            for file in os.listdir(dir_path)
        ]
