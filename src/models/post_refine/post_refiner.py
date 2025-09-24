"""
@Reference:
1. How to create llama index templates: https://blog.csdn.net/lovechris00/article/details/137782020
"""

from pathlib import Path
import traceback

from src.configs.constants import OUTPUT_DIR, RESOURCE_DIR
from src.configs.logger import get_logger
from src.configs.utils import load_latest_task_id
from src.models.monitor.token_monitor import TokenMonitor
from src.models.LLM import ChatAgent
from src.models.monitor.time_monitor import TimeMonitor
from src.modules.latex_handler.latex_comparison_table_builder import (
    LatexComparisonTableBuilder,
)
from src.modules.latex_handler.latex_figure_builder import LatexFigureBuilder
from src.modules.latex_handler.latex_list_table_builder import LatexListTableBuilder
from src.modules.latex_handler.latex_summary_table_builder import (
    LatexSummaryTableBuilder,
)
from src.modules.latex_handler.latex_figure_builder import LatexFigureBuilder
from src.modules.latex_handler.latex_summary_table_builder import (
    LatexSummaryTableBuilder,
)
from src.modules.post_refine import (
    FigRetrieveRefiner,
    RagRefiner,
    RuleBasedRefiner,
    SectionRewriter,
)
from src.modules.post_refine.base_refiner import BaseRefiner
from src.modules.utils import save_result

logger = get_logger("src.models.post_refine.PostRefiner")


class PostRefiner(BaseRefiner):
    def __init__(self, task_id: str = None, **kwargs) -> None:
        llamaindex_topk = (
            30 if "llamaindex_topk" not in kwargs else kwargs["llamaindex_topk"]
        )
        task_id = load_latest_task_id() if task_id is None else task_id
        assert task_id is not None
        super().__init__(task_id, **kwargs)
        self.paper_dir = Path(f"{OUTPUT_DIR}/{self.task_id}/papers")
        self.mainbody_path = Path(f"{OUTPUT_DIR}/{self.task_id}/tmp/mainbody.tex")
        self.refined_mainbody_path = Path(
            f"{OUTPUT_DIR}/{self.task_id}/tmp/mainbody_post_refined.tex"
        )

        self.max_retry_times = 2
        self.max_words = 10000

        # refine settings
        if "papers" not in kwargs:
            self.papers = self.load_papers(self.paper_dir)
        else:
            self.papers = kwargs["papers"]
        logger.info(f"PostRefiner load {len(self.papers)} papers")
        self.llamaindex_topk = llamaindex_topk

        # refining modules
        self.llamaindex_store_local = False
        if "llamaindex_wrapper" in kwargs:
            self.llamaindex_wrapper = kwargs["llamaindex_wrapper"]
            self.rag_refiner = RagRefiner(
                task_id,
                llamaindex_store_local=self.llamaindex_store_local,
                papers=self.papers,
                llamaindex_wrapper=self.llamaindex_wrapper,
            )
        else:
            self.rag_refiner = RagRefiner(
                task_id,
                llamaindex_store_local=self.llamaindex_store_local,
                papers=self.papers,
            )
            self.llamaindex_wrapper = self.rag_refiner.llamaindex_wrapper
        self.sec_rewriter = SectionRewriter(task_id, papers=self.papers)
        self.fig_builder = LatexFigureBuilder(task_id=task_id)

        self.rule_based_refiner = RuleBasedRefiner(task_id=task_id, papers=self.papers)
        self.fig_retrieve_refiner = FigRetrieveRefiner(
            task_id=task_id,
            llamaindex_topk=llamaindex_topk,
            papers=self.papers,
            llamaindex_wrapper=self.llamaindex_wrapper,
        )

        # -- tables
        chat_agent = ChatAgent(TokenMonitor(task_id, "generate table"))
        self.paper_dir = Path(f"{OUTPUT_DIR}/{str(self.task_id)}/papers")
        self.tmp_path = Path(f"{OUTPUT_DIR}/{str(self.task_id)}/tmp/table_gen")
        self.latex_path = Path(f"{OUTPUT_DIR}/{str(self.task_id)}/latex")
        self.prompt_dir = Path(f"{RESOURCE_DIR}/LLM/prompts/latex_table_builder")
        self.mainbody_tex_path = Path(
            f"{OUTPUT_DIR}/{str(self.task_id)}/tmp/mainbody_post_refined.tex"
        )
        self.outlines_path = Path(f"{OUTPUT_DIR}/{str(self.task_id)}/outlines.json")

        self.summary_table_builder = LatexSummaryTableBuilder(
            main_body_path=self.mainbody_tex_path,
            tmp_path=self.tmp_path,
            outline_path=self.outlines_path,
            latex_path=self.latex_path,
            paper_dir=self.paper_dir,
            prompt_dir=self.prompt_dir,
            chat_agent=chat_agent,
        )
        self.list_table_builder = LatexListTableBuilder(
            main_body_path=self.mainbody_tex_path,
            tmp_path=self.tmp_path,
            outline_path=self.outlines_path,
            latex_path=self.latex_path,
            paper_dir=self.paper_dir,
            prompt_dir=self.prompt_dir,
            chat_agent=chat_agent,
        )
        self.comparison_table_builder = LatexComparisonTableBuilder(
            main_body_path=self.mainbody_tex_path,
            tmp_path=self.tmp_path,
            outline_path=self.outlines_path,
            latex_path=self.latex_path,
            paper_dir=self.paper_dir,
            prompt_dir=self.prompt_dir,
            chat_agent=chat_agent,
        )

    def generate_tables(self):
        time_monitor = TimeMonitor(self.task_id)
        time_monitor.start("generate tabel")

        # -- tables 直接在mainbody_post_refined.tex文件上修改
        try:
            self.summary_table_builder.run()
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"An error occurred: {e}; The traceback: {tb_str}")

        try:
            self.list_table_builder.run()
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"An error occurred: {e}; The traceback: {tb_str}")

        try:
            self.comparison_table_builder.run()
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"An error occurred: {e}; The traceback: {tb_str}")

        time_monitor.end("generate tabel")

    def run(self, mainbody_path=None):
        time_monitor = TimeMonitor(self.task_id)
        time_monitor.start("post refine")

        if mainbody_path is None:
            mainbody_path = self.mainbody_path

        try_times = 0
        while try_times < self.max_retry_times:
            # rag_refiner
            self.rag_refiner.run(mainbody_path=mainbody_path)
            # fig_retrieve_refiner
            # self.fig_retrieve_refiner.run(mainbody_path=self.rag_refiner.refined_mainbody_path)
            # sec_rewriter
            self.sec_rewriter.run(mainbody_path=self.rag_refiner.refined_mainbody_path)
            # fig_builder
            self.fig_builder.run(mainbody_path=self.sec_rewriter.refined_mainbody_path)
            # rule_based_refiner
            final_refined_content = self.rule_based_refiner.run(
                mainbody_path=self.fig_builder.fig_mainbody_path
            )
            # save the final refined content
            save_result(final_refined_content, self.refined_mainbody_path)
            # generate tables
            self.generate_tables()
            words_count = len(final_refined_content.strip().split())
            if words_count < self.max_words:
                logger.debug(f"核验通过，postrefine后的main body总字数为{words_count}")
                break
            else:
                try_times += 1
                logger.debug(
                    f"核验不通过，postrefine后的main body总字数为{words_count}；postrefine后的main重新生成，trying times {try_times}, max trying: {self.max_retry_times}"
                )

        logger.info(f"Post refine and save content to {self.refined_mainbody_path}")
        time_monitor.end("post refine")
        return final_refined_content


# python -m src.models.post_refine.post_refiner
if __name__ == "__main__":
    task_id = load_latest_task_id()
    post_refiner = PostRefiner(task_id)
    # mainbody_path = Path(f"{OUTPUT_DIR}/{task_id}/tmp/mainbody_sec_rewritten.tex")
    mainbody_path = None
    post_refiner.run(mainbody_path=mainbody_path)
