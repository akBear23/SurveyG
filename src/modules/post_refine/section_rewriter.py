"""
@Reference:
1. How to create llama index templates: https://blog.csdn.net/lovechris00/article/details/137782020
"""

import re
from tqdm import tqdm
import traceback
from pathlib import Path

from src.configs.constants import OUTPUT_DIR, RESOURCE_DIR
from src.configs.config import ADVANCED_CHATAGENT_MODEL
from src.configs.logger import get_logger
from src.configs.utils import load_latest_task_id
from src.models.monitor.token_monitor import TokenMonitor
from src.modules.utils import save_result, load_prompt, clean_chat_agent_format
from src.modules.post_refine import BaseRefiner

logger = get_logger("src.modules.post_refine.SectionRewriter")


class SectionRewriter(BaseRefiner):
    def __init__(self, task_id: str = None, **kwargs) -> None:
        super().__init__(task_id, **kwargs)
        self.refined_mainbody_path = Path(
            f"{OUTPUT_DIR}/{self.task_id}/tmp/mainbody_sec_rewritten.tex"
        )
        self.rewrite_prompt_dir = Path(f"{RESOURCE_DIR}/LLM/prompts/section_rewriter")

        self.maintain_commands = ["section", "subsection", "label", "autoref"]
        self.chat_agent.token_monitor = TokenMonitor(task_id, "section rewrite")

    def extract_environment_content(self, content, command):
        """Extracts the content of LaTeX environment commands from a text."""
        pattern = re.compile(r"\\" + command + r"\{([^}]+)\}")
        res = pattern.findall(content)
        return res if res is not None else []

    def extract_fig_name(self, content):
        pattern = re.compile(r"\\includegraphics\[.*?\]\{figs/([^.]+)\.png\}")
        return pattern.findall(content)

    def replace_environment_contents(
        self, rewritten_text: str, original_text: str, commands: list[str]
    ):
        for command in commands:
            """Replaces the content of LaTeX environment commands in a text."""
            original_contents = self.extract_environment_content(
                content=original_text, command=command
            )
            current_contents = self.extract_environment_content(
                content=rewritten_text, command=command
            )
            if len(original_contents) != len(current_contents):
                logger.error(
                    f"Rewritting failed. original_contents: {original_contents}; current_contents: {current_contents}"
                )
                rewritten_text = original_text
                continue
            for i, content in enumerate(current_contents):
                if content != original_contents[i]:
                    rewritten_text = rewritten_text.replace(
                        f"\\{command}{{{content}}}",
                        f"\\{command}{{{current_contents[i]}}}",
                    )
        # 图片也要对应一下
        original_content = self.extract_fig_name(content=original_text)
        current_content = self.extract_fig_name(content=rewritten_text)
        for i, content in enumerate(current_content):
            if content != original_content[i]:
                rewritten_text = rewritten_text.replace(content, original_content[i])
        return rewritten_text

    def extract_section_line(self, latex_text):
        # 使用正则表达式匹配\section部分
        match = re.search(r"\\section\{([^}]*)\}\s*\\label\{([^}]*)\}", latex_text)
        if match:
            section_title = match.group(0)  # 提取完整的\section那一行
            return section_title
        return None

    def rule_address_gpt_rewrite_issues(
        self, origin_sec_contents: list, new_sec_contents: list
    ):
        # gpt 有一个很奇怪的bug，会丢掉\section那一行，尝试用规则解决这个问题，或者把\section改成\subsection
        if len(origin_sec_contents) != len(new_sec_contents):
            logger.error(
                f"origin_sec_contents: {origin_sec_contents};\n new_sec_contents: {new_sec_contents}"
            )
        assert len(origin_sec_contents) == len(new_sec_contents)
        for idx in range(len(origin_sec_contents)):
            if ("\\section" in origin_sec_contents[idx]) and (
                "\\section" not in new_sec_contents[idx]
            ):
                origin_sec_head = origin_sec_contents[idx].strip()[:200]
                sec_line = self.extract_section_line(latex_text=origin_sec_head)
                if sec_line not in origin_sec_head:
                    raise ValueError(
                        f"sec_content is None, origin_sec_head: {origin_sec_head}"
                    )
                new_sec_contents[idx] = f"{sec_line}\n\n{new_sec_contents[idx]}"
                logger.debug(f'"{sec_line}" is inserted into original section')

                potential_subsec_line = sec_line.replace("\\section", "\\subsection")
                if potential_subsec_line in new_sec_contents[idx]:
                    logger.debug(
                        f"{potential_subsec_line} was added in rewritten section by mistake, which has been corrected"
                    )
                    new_sec_contents[idx] = new_sec_contents[idx].replace(
                        potential_subsec_line, ""
                    )

        # 保证不改变特定内容
        assert len(origin_sec_contents) == len(new_sec_contents)
        for idx in range(len(origin_sec_contents)):
            new_sec_contents[idx] = self.replace_environment_contents(
                rewritten_text=new_sec_contents[idx],
                original_text=origin_sec_contents[idx],
                commands=self.maintain_commands,
            )

    def compress_sections(self, origin_sec_contents: list[str]) -> list[str]:
        # make the paper content more compact
        sec_prompts = []
        origin_lengths = []
        for sec_content in origin_sec_contents:
            sec_rewrite_prompt = load_prompt(
                filename=str(
                    self.rewrite_prompt_dir.joinpath("compress_sections.md").absolute()
                ),
                content=sec_content,
            )
            origin_lengths.append(len(sec_content.strip().split()))
            sec_prompts.append(sec_rewrite_prompt)
        section_content_list = self.chat_agent.batch_remote_chat(
            prompt_l=sec_prompts, desc="compress sections..."
        )
        new_sec_contents = [
            clean_chat_agent_format(content=one) for one in section_content_list
        ]

        self.rule_address_gpt_rewrite_issues(
            origin_sec_contents=origin_sec_contents, new_sec_contents=new_sec_contents
        )

        # print length
        current_lengths = [len(one.strip().split()) for one in new_sec_contents]
        for id_, (ori, cur) in enumerate(zip(origin_lengths, current_lengths)):
            logger.info(
                f"Section {id_ + 1}: the words of original section is {ori}, and the words of compressed section is {cur}, the ratio is {round(cur / ori, 2)}"
            )

        return new_sec_contents

    def rewrite_main_sections(self, origin_sec_contents: list[str]) -> list[str]:
        new_sec_contents = []

        # compress the introduction section
        introduction_content = origin_sec_contents[0]
        intro_compression_prompt = load_prompt(
            filename=str(
                self.rewrite_prompt_dir.joinpath(
                    "introduction_compression.md"
                ).absolute()
            ),
            introduction_sec=introduction_content,
        )
        compressed_intro = self.chat_agent.remote_chat(
            intro_compression_prompt, model=ADVANCED_CHATAGENT_MODEL
        )
        new_sec_contents.append(introduction_content)

        # iteratively rewrite sections
        compressed_context = compressed_intro
        sec_id = 2
        for sec_content in tqdm(
            origin_sec_contents[1:], desc=f"iteratively rewriting section no.{sec_id}"
        ):
            sec_rewrite_prompt = load_prompt(
                filename=str(
                    self.rewrite_prompt_dir.joinpath(
                        "rewrite_section_with_compressed_context.md"
                    ).absolute()
                ),
                context=compressed_context,
                content=sec_content,
            )
            new_sec_content = self.chat_agent.remote_chat(
                sec_rewrite_prompt, model=ADVANCED_CHATAGENT_MODEL
            )
            new_sec_content = clean_chat_agent_format(content=new_sec_content)
            new_sec_contents.append(new_sec_content)

            sec_id += 1
            if len(new_sec_contents) >= len(origin_sec_contents):
                break

            # ----
            compression_prompt = load_prompt(
                filename=str(
                    self.rewrite_prompt_dir.joinpath(
                        "iterative_compression.md"
                    ).absolute()
                ),
                previous_compression=compressed_context,
                content_for_compressions=sec_content,
            )
            compressed_context = self.chat_agent.remote_chat(
                compression_prompt, model=ADVANCED_CHATAGENT_MODEL
            )
            compressed_context = clean_chat_agent_format(content=compressed_context)
        self.rule_address_gpt_rewrite_issues(
            origin_sec_contents=origin_sec_contents, new_sec_contents=new_sec_contents
        )

        # print length
        origin_lengths = [len(one.strip().split()) for one in origin_sec_contents]
        current_lengths = [len(one.strip().split()) for one in new_sec_contents]
        for id_, (ori, cur) in enumerate(zip(origin_lengths, current_lengths)):
            logger.info(
                f"Section {id_ + 1}: the words of original section is {ori}, and the words of rewritten section is {cur}, the ratio is {round(cur / ori, 2)}"
            )

        return new_sec_contents

    def rewrite_conclusion(self, origin_conclusion_text: str, introduction_text: str):
        prompt = load_prompt(
            filename=str(
                self.rewrite_prompt_dir.joinpath("rewrite_conclusion.md").absolute()
            ),
            introduction=introduction_text,
            origin_conclusion=origin_conclusion_text,
        )
        conclusion = self.chat_agent.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
        conclusion = clean_chat_agent_format(content=conclusion)

        # 查看重写后是否有问题
        origin_conclusion_tmp_list = [origin_conclusion_text]
        new_conclusion_tmp_list = [conclusion]
        self.rule_address_gpt_rewrite_issues(
            origin_sec_contents=origin_conclusion_tmp_list,
            new_sec_contents=new_conclusion_tmp_list,
        )
        logger.info(f"Rewrote the conclusion section.")
        conclusion = new_conclusion_tmp_list[0]
        return conclusion

    def run(self, mainbody_path=None):
        if mainbody_path is None:
            mainbody_path = self.mainbody_path
        survey_sections = self.load_survey_sections(mainbody_path)
        # compression
        sec_contents = [one.content for one in survey_sections]
        try:
            compressed_survey_sections = self.compress_sections(
                origin_sec_contents=sec_contents
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"Compression failed.  An error occurred: {e}; The traceback: {tb_str} "
            )
            compressed_survey_sections = sec_contents

        # ------ 使用迭代重写 --------
        # enhance coherence
        try:
            rewritten_survey_sections = self.rewrite_main_sections(
                origin_sec_contents=compressed_survey_sections[:-1]
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(
                f"Rewritten failed.  An error occurred: {e}; The traceback: {tb_str} "
            )
            rewritten_survey_sections = compressed_survey_sections[:-1]

        new_conclusion = self.rewrite_conclusion(
            origin_conclusion_text=compressed_survey_sections[-1],
            introduction_text=compressed_survey_sections[0],
        )
        rewritten_survey_sections.append(new_conclusion)

        # # ------- 不使用迭代重写 --------
        # rewritten_survey_sections = compressed_survey_sections

        rewritten_survey_text = "\n".join(rewritten_survey_sections)
        save_result(rewritten_survey_text, self.refined_mainbody_path)
        logger.debug(f"Save content to {self.refined_mainbody_path}.")
        return rewritten_survey_text


if __name__ == "__main__":
    task_id = load_latest_task_id()
    print(f"task_id: {task_id}")
    # store vector index into local directory for the convenience of debugging
    sec_rewriter = SectionRewriter(task_id=task_id)
    sec_rewriter.run()
