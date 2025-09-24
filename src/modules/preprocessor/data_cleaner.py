import json
import os
import re
from pathlib import Path
from typing import Union

from tqdm import tqdm

from src.configs.config import BASE_DIR, CHAT_AGENT_WORKERS, MD_TEXT_LENGTH
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.models.LLM.utils import cut_text_by_token, load_prompt
from src.models.monitor.time_monitor import TimeMonitor
from src.modules.utils import (
    clean_chat_agent_format,
    load_file_as_string,
    sanitize_filename,
    save_result,
)

logger = get_logger("src.modules.preprocessor.DataCleaner")


class                                                                                                                                                             DataCleaner:
    def __init__(self, papers: list[dict] = []):
        self.papers: list[dict] = papers
        self.chat_agent_workers = CHAT_AGENT_WORKERS

    def load_json_dir(self, json_path_dir: Path):
        """load papers from json directory."""
        papers = []
        cnt_total = 0
        for file in os.listdir(json_path_dir):
            if file.endswith(".json"):
                p = os.path.join(json_path_dir, file)
                dic = json.loads(load_file_as_string(p))
                if (
                    "md_text" in dic
                ):  # Only consider those with `md_text` as available papers.
                    papers.append(dic)
                cnt_total += 1
        logger.info(f"Find {len(papers)} out of {cnt_total} papers available.")
        self.papers = papers

    def complete_title(self):
        for paper in tqdm(self.papers, desc="completing title..."):
            if "title" not in paper:
                paper["title"] = paper["md_text"].splitlines()[0].strip(" #")
                paper["title"] = paper["title"][:32]  # avoid too long title

    def complete_abstract(self):
        pattern = r"\s*a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*"  # find "abstract" substring, with whitespace bettween letters.
        for paper in tqdm(self.papers, desc="completing abstract..."):
            if "abstract" in paper and len(paper["abstract"]) > 500:
                continue
            match = re.search(pattern, paper["md_text"], re.IGNORECASE)
            if match:
                index = match.start()
                paper["abstract"] = paper["md_text"][index : index + 2000]
            else:
                paper["abstract"] = paper["md_text"][:2000]

    def complete_bib(self, bib_file_save_path: str):
        """Not only complete the bib_name, also need to save all bibnames into a references.bib file"""
        var_name_i = 0
        bib_all = []
        remove_non_ascii_chars = (
            lambda input_string: input_string.replace(",", "")
            .encode("ascii", "ignore")
            .decode("ascii")
        )

        for paper in tqdm(self.papers, desc="completing bibname..."):
            if "reference" in paper:
                bib_name = paper["reference"].splitlines()[0].split("{")[1].strip(",")
                new_bib_name = remove_non_ascii_chars(bib_name)

                paper["bib_name"] = new_bib_name
                paper["reference"] = paper["reference"].replace(bib_name, new_bib_name)
            else:
                title = remove_non_ascii_chars(paper["title"])
                bib_name = "".join([c for c in title if not c.isspace()][:10]) + str(
                    var_name_i
                )
                var_name_i += 1
                bib_tex = f"@article{{{bib_name},\ntitle={{{title}}}\n}}"

                paper["reference"] = bib_tex
                paper["bib_name"] = bib_name

            bib_all.append(paper["reference"])

        save_result("\n".join(bib_all), bib_file_save_path)

    def check_md_text_length(self):
        for paper in self.papers:
            if "md_text" not in paper:
                continue
            md_text = paper["md_text"]
            paper["md_text"] = cut_text_by_token(md_text, MD_TEXT_LENGTH)

    def __process_paper_type_response(self, res: str, paper_index: int):
        kinds = ["method", "benchmark", "theory", "survey"]
        for k in kinds:
            if k in res.lower():
                self.papers[paper_index]["paper_type"] = k
                return True
        logger.error(
            f"failed to extract papertype of {self.papers[paper_index]['title']}"
        )
        logger.error(f"The response from gpt is {res}")
        return False

    def get_paper_type(self, chat_agent: ChatAgent):
        """complete the paper type field with chatgpt."""
        # load prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            abstract = paper["abstract"]
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/preprocessor/paper_type_classification.md",
                abstract=abstract,
            )
            prompts_and_index.append([prompt, i])
        # batch_chat
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(prompts, desc="getting paper type...")
            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_paper_type_response(res, paper_index)
            ]
            cnt += 1

    def __process_attri_response(self, res: str, paper_index: int):
        res = clean_chat_agent_format(content=res)
        try:
            res_dic = json.loads(res)
            self.papers[paper_index]["attri"] = {**res_dic}
            return True
        except Exception as e:
            logger.debug(
                f"Failed to process {self.papers[paper_index]['title']}; The res: {res[:100]}; {e}"
            )
            return False

    def get_attri(self, chat_agent: ChatAgent):
        """extract attribute tree from paper"""
        # 获取所有含 "md_text" 的文件并生成 prompts
        prompts_and_index = []
        for i, paper in enumerate(self.papers):
            # 根据 paper_type 加载对应的 prompt
            paper_type = paper["paper_type"].lower()
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/preprocessor/attri_tree_for_{paper_type}.md",
                paper=paper["md_text"],
            )
            prompts_and_index.append([prompt, i])

        # 批量处理 prompts
        cnt = 0
        while prompts_and_index and cnt < 3:
            prompts = [x[0] for x in prompts_and_index]
            res_l = chat_agent.batch_remote_chat(
                prompts, desc="getting attribute tree from paper......"
            )

            prompts_and_index = [
                (prompt, paper_index)
                for res, (prompt, paper_index) in zip(res_l, prompts_and_index)
                if not self.__process_attri_response(res, paper_index)
            ]
            cnt += 1

    def save_papers(
        self, save_dir: Union[str, Path], file_name_attr: str = "title"
    ) -> None:
        """save every cleaned paper."""
        filter_field = [
            "from",
            "scholar_id",
            "detail_id",
            "title",
            "abstract",
            "bib_name",
            "md_text",
            "paper_type",
            "attri",
            "mount_outline",
            "similarity_score",
            "image",
        ]
        for paper in self.papers:
            try:
                file_name = paper[file_name_attr] + ".json"
                file_name = sanitize_filename(file_name)
                file_path = os.path.join(save_dir, file_name)
                save_dic = {key: paper.get(key, None) for key in filter_field}
                save_result(json.dumps(save_dic, indent=4), file_path)
            except Exception as e:
                logger.error(
                    f"There is an error when saving {file_path}. The error is: {e}"
                )
        return self.papers

    def quick_check(self) -> list[dict]:
        """Used in PaperRecaller for quick check"""
        papers_with_md = [paper for paper in self.papers if "md_text" in paper]
        self.papers = papers_with_md
        self.complete_title()
        self.complete_abstract()
        return self.papers

    def offline_proc(self, task_id: str, ref_path: str) -> None:
        ref_data_path = Path(ref_path)
        md_texts = [p.read_text() for p in ref_data_path.glob("*.md") if p.is_file()]
        self.papers = [{"md_text": md_text} for md_text in md_texts]

        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        self.complete_bib(bib_file_path)

        self.check_md_text_length()
        chat_agent = ChatAgent()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

    def run(self, task_id: str, chat_agent: ChatAgent = None):
        time_monitor = TimeMonitor(task_id)
        time_monitor.start("clean paper")

        self.load_json_dir(Path(OUTPUT_DIR) / task_id / "jsons")
        self.complete_title()
        self.complete_abstract()
        bib_file_path = Path(OUTPUT_DIR) / task_id / "latex" / "references.bib"
        self.complete_bib(bib_file_path)

        self.check_md_text_length()
        if chat_agent is None:
            chat_agent = ChatAgent()
        self.get_paper_type(chat_agent=chat_agent)
        self.get_attri(chat_agent=chat_agent)

        save_path = Path(f"{OUTPUT_DIR}/{task_id}/papers")
        self.save_papers(save_dir=save_path)
        logger.info(f"========== {len(self.papers)} remain after cleaning. ==========")

        time_monitor.end("clean paper")


# python -m src.modules.preprocessor.data_cleaner
if __name__ == "__main__":
    dc = DataCleaner()
    dc.offline_proc("ref1")
    print(len(dc.papers))
