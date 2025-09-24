import re
from pathlib import Path
from typing import List

from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from src.configs.config import ADVANCED_CHATAGENT_MODEL, BASE_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.modules.latex_handler.latex_base_table_builder import LatexBaseTableBuilder
from src.modules.utils import (
    load_file_as_string,
    load_meta_data,
    load_prompt,
    load_single_file,
    save_result,
)

logger = get_logger("src.modules.latex_handler.LatexListTableBuilder")


class LatexListTableBuilder(LatexBaseTableBuilder):
    def __init__(
        self,
        main_body_path: Path,
        tmp_path: Path,
        latex_path: Path,
        outline_path: Path,
        paper_dir: Path,
        prompt_dir: Path,
        chat_agent: ChatAgent,
    ):
        self.chat_agent = chat_agent
        super().__init__(chat_agent=self.chat_agent)
        self.main_body_path = main_body_path
        self.latex_path = latex_path
        self.outline_path = outline_path
        self.benchmark_table_tex_path = latex_path / "benchmark_table.tex"
        self.paper_dir = paper_dir
        # self.benchmark_prompt_path = prompt_dir / 'Benchmark.txt'
        self.attr_extract_prompt_path = prompt_dir / "Attribute_Extract.txt"
        self.attr_fill_prompt_path = prompt_dir / "Attribute_fill.txt"
        self.description_prompt_path = prompt_dir / "Table_description.txt"
        self.rewrite_prompt_path = prompt_dir / "Content_rewrite.txt"
        self.tmp_path = tmp_path
        # self.exist_primary_attribute_path = os.path.join(tmp_path, "exist/Primary Attribute.json")

    def generate_benchmark_table_latex(self, data):
        # Define the column names
        columns = ["Benchmark", "Size", "Domain", "Task Format", "Metric"]

        # Start the LaTeX table code
        latex_code = r"""
    \begin{table}[h!]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{p{0.25\textwidth} p{0.15\textwidth} p{0.35\textwidth} p{0.30\textwidth} p{0.25\textwidth}}
    \toprule
    """
        # Add column headers with bold formatting
        latex_code += (
            " & ".join([f"\\textbf{{{col}}}" for col in columns])
            + r" \\"
            + "\n"
            + r"\midrule"
            + "\n"
        )

        # Add each row of data
        for item in data:
            row = (
                f"{item['cite_name']} & {item['size']} & {item['domain']} & "
                f"{item['task format']} & {item['metric']} \\\\"
            )
            latex_code += row + "\n"

        # Close the table
        latex_code += r"""
    \bottomrule
    \end{tabular}%
    }
    \caption{Representative Benchmarks}
    \label{tab:benchmark_table}
    \end{table}
    """
        return latex_code
        # Save the LaTeX code to a .tex file
        # with open(output_file, "w") as file:
        #     file.write(latex_code)
        # logger.info(f"Benchmark LaTeX table saved to {output_file}")

    def generate_arbitrary_table_latex(self, columns, data, index):
        """
        生成LaTeX表格代码并保存到文件

        参数:
        columns: List[str] - 表格的列名
        data: List[Dict[str, str]] - 表格的数据，每个字典对应一行
        output_file: str - 保存生成的LaTeX代码的文件名
        """
        # 设置列宽，第一列为 0.2，其他列为 0.4
        column_definitions = r"p{0.2\textwidth} " + " ".join(
            [r"p{0.4\textwidth}" for _ in columns[1:]]
        )

        # 开始生成LaTeX代码
        latex_code = (
            r"""
    \begin{table}[h!]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{"""
            + column_definitions
            + r"""}
    \toprule
    """
        )

        # 添加列名
        latex_code += (
            " & ".join([f"\\textbf{{{col}}}" for col in columns])
            + r" \\"
            + "\n"
            + r"\midrule"
            + "\n"
        )

        # 添加每行数据
        for item in data:
            # 获取值并转义特殊字符
            values = [
                str(value).replace("_", "\\_").replace("&", "\\&")
                for value in item.values()
            ]
            for i in range(1, 4):
                values[i] = self.format_string(values[i])
            row = " & ".join(values) + r" \\"
            latex_code += row + "\n"

        # 关闭表格
        latex_code += r"""
    \bottomrule
    \end{tabular}%
    }
    \caption{Your Table Caption Here}
    """
        latex_code += f"\n\\label{{tab:Arbitrary_table_{index}}}"
        latex_code += "\end{table}"
        return latex_code
        # # 保存到文件
        # with open(output_file, "w") as file:
        #     file.write(latex_code)

    def find_method_section(self) -> str:
        outlines = load_single_file(self.outline_path)
        section_titles, subsection_titles = self.parse_outline(outlines)
        ans = None  # 用于保存符合条件的元素
        for section_title in section_titles:
            if (
                "method" in section_title.lower()
                or "technique" in section_title.lower()
            ):
                ans = section_title
                break  # 满足条件，退出循环
        if ans == None:
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/latex_table_builder/find_method_section.md",
                outline=str(outlines),
            )
            res = self.chat_agent.remote_chat(
                text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
            )
            if res is None:
                return None
            try:
                ans = re.search(r"<Answer>(.*?)</Answer>", res, re.DOTALL)
                if ans:
                    ans = ans.group(1)
                else:
                    print("Error: No <Answer> tag found in the input text.")
                    return
            except Exception as e:
                return
        title = ans
        logger.debug(f"Select method section '{title}' to generate arbitrary_table.")
        return title

    def find_benchmark_section(self) -> str:
        outlines = load_single_file(self.outline_path)
        section_titles, subsection_titles = self.parse_outline(outlines)
        ans = None  # 用于保存符合条件的元素
        for subsection_title in subsection_titles:
            if "benchmark" in subsection_title.lower():
                ans = subsection_title
                break  # 满足条件，退出循环
        if ans == None:
            prompt = load_prompt(
                f"{BASE_DIR}/resources/LLM/prompts/latex_table_builder/find_benchmark_section.md",
                outline=str(outlines),
            )
            res = self.chat_agent.remote_chat(
                text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
            )
            if res is None:
                return None
            try:
                ans = re.search(r"<Answer>(.*?)</Answer>", res, re.DOTALL)
                if ans:
                    ans = ans.group(1)
                else:
                    print("Error: No <Answer> tag found in the input text.")
                    return
            except Exception as e:
                return
        title = ans
        logger.debug(f"Select benchmark section '{title}' to generate bechmark_table.")
        return title

    def generate_benchmark_tabel(self, section_name: str):
        if section_name is None:
            return
        target_size = 8
        meta_data = load_meta_data(self.paper_dir)
        subsection_content = self.extract_subsection_content(
            self.main_body_path, section_name
        )
        if subsection_content is None:
            return
        # section_name为要生成benchmark表格对应的章节名称
        cite_names = self.extract_cite_name(subsection_content)
        info_list = []
        for name in cite_names:
            info = self.cite_name_match_benchmark(meta_data, name)
            if info is not None:
                info_list.append(info)
        attr_list = []
        if len(info_list) < target_size:
            info_list = self.supplement_data(
                current_data=info_list, dir_path=self.paper_dir, target_size=target_size
            )
        for info in tqdm(
            info_list, desc="generating benchmark table...", total=len(info_list)
        ):
            attr = {}
            attr["size"] = info["size"]
            attr["domain"] = self.format_string(info["domain"])
            attr["task format"] = self.format_string(info["task format"])
            attr["bib_name"] = info["bib_name"]
            attr["metric"] = (
                re.sub(r"\s*\([^)]*\)", "", info["metric"])
                if isinstance(info["metric"], str)
                else info["metric"]
            )
            attr["cite_name"] = (
                info["name"].replace("&", "\\&") + f"\\cite{{{info['bib_name']}}}"
            )
            attr["Benchmark"] = info["name"]
            attr_list.append(attr)
        latex_code = self.generate_benchmark_table_latex(attr_list)
        caption, introductory_sentence = self.generate_description(
            latex_code, subsection_content
        )
        caption = re.sub(r"\{[^}]*\}", "", caption)
        caption = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", "", caption)
        if caption is not None and introductory_sentence is not None:
            try:
                latex_code = re.sub(
                    r"\\caption\{.*?\}", rf"\\caption{{{caption}}}", latex_code
                )
            except re.error as e:
                print(f"Error during regex substitution for latex_code: {e}")
                return
            try:
                introductory_sentence = re.sub(
                    r"\\?input\{.*?\}",
                    rf"\\ref{{tab:benchmark_table}}",
                    introductory_sentence,
                )
            except re.error as e:
                print(f"Error during regex substitution for introductory_sentence: {e}")
                return
            tex = load_file_as_string(self.main_body_path)
            prompt = load_prompt(
                self.rewrite_prompt_path,
                Content=subsection_content,
                Sentence=introductory_sentence,
            )
            res = self.chat_agent.remote_chat(
                text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
            )
            # Add introductory sentence
            try:
                revised_content = re.search(r"<Answer>(.*?)</Answer>", res, re.DOTALL)
                if revised_content:
                    revised_content = revised_content.group(1)
                else:
                    print("Error: No <Answer> tag found in the input text.")
                    return
            except Exception as e:
                return
            tex = tex.replace(subsection_content, revised_content)
            save_result(tex, self.main_body_path)
            # Add input\{benchmark_table}
            tex = load_file_as_string(self.main_body_path)
            subsection_title = self.extract_subsection_title(subsection_content)
            tex = tex.replace(
                subsection_title, f"{subsection_title}\n\n\\input{{benchmark_table}}"
            )
            save_result(tex, self.main_body_path)
            self.save_table_file(latex_code, self.benchmark_table_tex_path)
            logger.info(
                f"Benchmark LaTeX table saved to {self.benchmark_table_tex_path}"
            )

    def generate_arbitrary_table(self, section_name):
        attr_list = []
        if section_name is None:
            return
        section_content = self.extract_section_content(
            self.main_body_path, section_name
        )
        if section_content is None:
            return

        subsections, titles = self.extract_subsections(section_content)

        i = 0
        for subsection in tqdm(subsections, total=len(subsections)):
            if "\input{benchmark_table}" in subsection:
                print("A table has been generated here")
                continue
            attr_list = []
            valid = 1
            cite_names = self.extract_cite_name(subsection)
            meta_data = load_meta_data(self.paper_dir)
            count = self.cite_name_match_count(meta_data, cite_names)
            if count < 3:
                continue
            prompt = load_prompt(
                self.attr_extract_prompt_path,
                Content=subsection,
            )
            result = self.chat_agent.remote_chat(
                text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
            )
            attr_info = self.extract_and_convert(result)
            if attr_info is None:
                return
            # print(attr_info)
            attr_list.append("Method Name")
            for attr_name in ["Attribute1", "Attribute2", "Attribute3"]:
                value = attr_info.get(attr_name, None)  # 获取属性值，默认为 None
                if value is None:  # 如果值为 None，打印报错信息
                    valid = 0
                    break
                attr_list.append(value)  # 无论是否为 None，都添加到列表
            if valid == 0:
                continue
            # attr_list.append(attr_info['Attribute1'])
            # attr_list.append(attr_info['Attribute2'])
            # attr_list.append(attr_info['Attribute3'])

            data_list = []
            for name in cite_names:
                complete_info, content, title, method_name, abbr, bib_name = (
                    self.cite_name_match(meta_data, name)
                )
                if complete_info is not None:
                    prompt = load_prompt(
                        self.attr_fill_prompt_path,
                        Info=str(attr_info),
                        Content=complete_info,
                    )
                    # print(prompt)
                    result = self.chat_agent.remote_chat(
                        text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
                    )
                    temp_data = self.extract_and_convert(result)
                    # print(temp_data)
                    data = {}
                    data["Method name"] = (
                        abbr.replace("&", "\\&") + f"\\cite{{{bib_name}}}"
                    )
                    if temp_data is None:
                        continue
                    data.update(temp_data)
                    data_list.append(data)
            for element in data_list:
                for key, value in element.items():
                    if isinstance(value, str) and "Not " in value:
                        element[key] = "-"
            value_list = self.get_value_list(data_list)
            validity = 0
            for list in value_list:
                state = self.validity_judge(list)
                if state == 0:
                    validity += 1
                    break
            if validity >= 1:
                print("This table has too many invalid entries")
                continue
            value_list = self.get_value_list(data_list)
            repeat = 0
            for list in value_list:
                similarity_score = self.calculate_similarity(list)
                if similarity_score >= 0.5:
                    repeat += 1
                    break
            if repeat >= 1:
                print("This table has too many duplicate contents")
                continue
            i += 1
            table_file_name = f"Arbitrary_table_{i}.tex"
            output_file_path = self.latex_path / table_file_name
            latex_code = self.generate_arbitrary_table_latex(attr_list, data_list, i)
            caption, introductory_sentence = self.generate_description(
                latex_code, subsection
            )
            caption = re.sub(r"\{[^}]*\}", "", caption)
            caption = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", "", caption)
            if caption is not None and introductory_sentence is not None:
                try:
                    latex_code = re.sub(
                        r"\\caption\{.*?\}", rf"\\caption{{{caption}}}", latex_code
                    )
                except re.error as e:
                    print("Error")
                    print(f"Error during regex substitution for latex_code: {e}")
                    continue
                try:
                    introductory_sentence = re.sub(
                        r"\\?input\{.*?\}",
                        rf"\\ref{{tab:Arbitrary_table_{i}}}",
                        introductory_sentence,
                    )
                except re.error as e:
                    print("Error")
                    print(
                        f"Error during regex substitution for introductory_sentence: {e}"
                    )
                    continue
                tex = load_file_as_string(self.main_body_path)
                prompt = load_prompt(
                    self.rewrite_prompt_path,
                    Content=subsection,
                    Sentence=introductory_sentence,
                )
                res = self.chat_agent.remote_chat(
                    text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
                )
                # Add introductory sentence
                try:
                    revised_content = re.search(
                        r"<Answer>(.*?)</Answer>", res, re.DOTALL
                    )
                    if revised_content:
                        revised_content = revised_content.group(1)
                    else:
                        print("Error: No <Answer> tag found in the input text.")
                        return
                except Exception as e:
                    return
                tex = tex.replace(subsection, revised_content)
                save_result(tex, self.main_body_path)
                # Add input\{Arbitrary_table_{i}}
                tex = load_file_as_string(self.main_body_path)
                subsection_title = self.extract_subsection_title(subsection)
                tex = tex.replace(
                    subsection_title,
                    f"{subsection_title}\n\n\\input{{Arbitrary_table_{i}}}",
                )
                save_result(tex, self.main_body_path)
                self.save_table_file(latex_code, output_file_path)
                logger.info(f"Arbitrary LaTeX table saved to {output_file_path}")

    @staticmethod
    def get_sections(survey_path: str) -> List[str]:
        """
        Get the section names of the survey.
        """
        tex = open(survey_path, "r").read()
        pattern = r"\\section{"
        match_l = list(re.finditer(pattern, tex))
        res = []
        for i in range(len(match_l) - 1):
            section_tex = tex[match_l[i].start() : match_l[i + 1].start()]
            res.append(section_tex)
        return res

    @staticmethod
    def get_title(section: str) -> str:
        """
        Get the title of the section.
        """
        title = re.findall(r"\\section\{([^}]+)\}", section)[0]
        return title

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def run(self):
        benchmark_section = self.find_benchmark_section()
        self.generate_benchmark_tabel(benchmark_section)
        method_section = self.find_method_section()
        self.generate_arbitrary_table(method_section)
