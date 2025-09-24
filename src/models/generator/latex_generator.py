import io
import os
import subprocess
import sys
import traceback
from pathlib import Path

import fitz

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # run code in any path

from src.configs.constants import OUTPUT_DIR, RESOURCE_DIR
from src.configs.logger import get_logger
from src.configs.utils import load_latest_task_id
from src.models.LLM import ChatAgent
from src.models.monitor.time_monitor import TimeMonitor
from src.modules.latex_handler.latex_text_builder import LatexTextBuilder

logger = get_logger("src.modules.generator.LatexGenerator")


class LatexGenerator:
    def __init__(self, task_id: str, **kwargs):
        task_id = load_latest_task_id() if task_id is None else task_id
        assert task_id is not None
        self.task_id = task_id
        self.outlines_path = Path(f"{OUTPUT_DIR}/{str(self.task_id)}/outlines.json")

        # for text builder
        self.init_text_tex_path = Path(f"{RESOURCE_DIR}/latex/survey.ini.tex")
        self.mainbody_tex_path = Path(
            f"{OUTPUT_DIR}/{str(self.task_id)}/tmp/mainbody_post_refined.tex"
        )
        self.post_refined_mainbody_tex_path = Path(
            f"{OUTPUT_DIR}/{str(self.task_id)}/tmp/mainbody_post_refined.tex"
        )
        self.abstract_path = Path(f"{OUTPUT_DIR}/{str(self.task_id)}/tmp/abstract.tex")
        self.survey_tex_path = Path(
            f"{OUTPUT_DIR}/{str(self.task_id)}/latex/survey.tex"
        )

        # init builders
        # -- text
        self.text_builder = LatexTextBuilder(init_tex_path=self.init_text_tex_path)

    def add_watermark(self, input_pdf: Path, output_pdf: Path, watermark_pdf: Path):
        # 打开输入的 PDF 和水印 PDF
        doc = fitz.open(input_pdf)
        watermark = fitz.open(watermark_pdf)
        # 获取水印页面（假设水印文件只有一页）
        watermark_page = watermark[0]
        # 获取水印页面的 pixmap（即图像）
        watermark_pixmap = watermark_page.get_pixmap()
        # 将水印图像转换为字节流
        img_stream = io.BytesIO(watermark_pixmap.tobytes())
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 获取当前页面的尺寸
            page_rect = page.rect
            # 将水印图像插入页面
            page.insert_image(page_rect, stream=img_stream, overlay=False, alpha=0.3)
        # 保存修改后的 PDF
        doc.save(output_pdf)

    def compile_single_survey(self):
        time_monitor = TimeMonitor(self.task_id)
        time_monitor.start("compile latex")

        task_dir = Path(BASE_DIR) / "outputs" / self.task_id
        latex_dir = task_dir / "latex"
        sty_file_path = Path(BASE_DIR) / "resources" / "latex" / "neurips_2024.sty"
        water_mark_pdf_path = Path(BASE_DIR) / "resources" / "latex" / "watermark.png"

        os.chdir(task_dir)
        if task_dir.joinpath("survey.pdf").exists():
            # 删除文件 output/survey.pdf
            subprocess.run(f"rm survey.pdf", shell=True)
            subprocess.run(f"rm survey_wtmk.pdf", shell=True)

        # 切换到 latex 目录
        os.chdir(latex_dir)

        # prepare sty file
        subprocess.run(["cp", sty_file_path, "./neurips_2024.sty"])

        # 执行 latexmk 命令，将输出重定向到 compile.log
        with open("compile.log", "w") as output_file:
            logger.debug(
                f'Running "latexmk -pdf -interaction=nonstopmode -f survey.tex". The compile.log is at {latex_dir / "compile.log"}'
            )
            subprocess.run(
                "latexmk -pdf -interaction=nonstopmode -f survey.tex",
                shell=True,
                stdout=output_file,
                stderr=output_file,
            )

        # 执行 latexmk -c 删除中间文件
        with open("compile.log", "a") as output_file:
            logger.debug(f'Running "latexmk -c"')
            subprocess.run("latexmk -c", shell=True, stdout=output_file)

        # 删除所有 .bbl 文件
        subprocess.run("rm *.bbl", shell=True)

        subprocess.run("rm neurips_2024.sty", shell=True)

        # 将生成的 survey.pdf 移动到上一级目录
        subprocess.run("mv survey.pdf ../", shell=True)
        self.add_watermark(
            task_dir / "survey.pdf", task_dir / "survey_wtmk.pdf", water_mark_pdf_path
        )

        time_monitor.end("compile latex")

    def generate_full_survey(self):
        # make survey.tex
        tex_content = self.text_builder.run(
            outlines_path=self.outlines_path,
            abstract_path=self.abstract_path,
            main_body_path=self.post_refined_mainbody_tex_path,
            latex_save_path=self.survey_tex_path,
        )
        return tex_content

# python -m src.models.generator.latex_generator
if __name__ == "__main__":
    # task_id = load_latest_task_id()
    task_id = "ref1"
    print(f"task_id: {task_id}")
    latex_generator = LatexGenerator(task_id=task_id)

    latex_generator.generate_full_survey()
    latex_generator.compile_single_survey()
