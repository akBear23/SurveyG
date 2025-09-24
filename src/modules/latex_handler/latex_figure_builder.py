"""
input file:
    - init file:
        structure_fig.ini.tex
    - mainbody.tex
    - survey.tex
    - outline.json
output file:
    - structure_fig.tex
modified file:
    - survey.tex
"""

import json
import math
import random
import re
import traceback
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from src.configs.config import ADVANCED_CHATAGENT_MODEL, BASE_DIR
from src.configs.utils import load_latest_task_id
from src.configs.constants import OUTPUT_DIR
from src.configs.logger import get_logger
from src.models.LLM import ChatAgent
from src.models.LLM.utils import load_prompt
from src.models.monitor.token_monitor import TokenMonitor
from src.modules.latex_handler.utils import fuzzy_match
from src.modules.utils import clean_chat_agent_format, load_file_as_string, save_result
from src.schemas.outlines import Outlines
from src.schemas.paper import Paper
from src.schemas.paragraph import Paragraph

logger = get_logger("src.modules.latex_handler.LatexFigureBuilder")


class BaseFigureBuilder(ABC):
    palette = [
        "a44c34",
        "bc966f",
        "b9ac99",
        "96a25f",
        "f3e6f1",
        "fdcee4",
        "c77fa1",
        "c1c8cd",
        "727b83",
        "9C7C7C",
        "D3E0EA",
        "b8a9a9",
    ]

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.fig_mainbody_path: Path = (
            Path(OUTPUT_DIR) / task_id / "tmp" / "mainbody_fig_refined.tex"
        )
        self.refined_mainbody_path = self.fig_mainbody_path

    @staticmethod
    def reset_palette():
        random.shuffle(BaseFigureBuilder.palette)


# Integrate every figure builder here:
class LatexFigureBuilder(BaseFigureBuilder):
    def __init__(self, task_id: str) -> None:
        super().__init__(task_id)
        self.structure_fig_builder = StructureFigureBuilder(task_id)
        self.tree_fig_builder = TreeFigureBuilder(task_id)
        self.tiny_tree_fig_builder = TinyTreeFigureBuilder(task_id)

    def run(self, mainbody_path: Path):
        # -- chapter structure figure
        try:
            self.structure_fig_builder.create_structure_figure(
                input_mainbody_path=mainbody_path,
            )  # !! create figure mainbody file
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"An error occurred: {e}; The traceback: {tb_str} ")
        # -- tree figure
        try:
            self.tree_fig_builder.run(
                self.fig_mainbody_path
            )  # !! modify figure mainbody file
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"An error occurred: {e}; The traceback: {tb_str} ")
        # -- tiny tree figure
        try:
            self.tiny_tree_fig_builder.run()  # !! modify figure mainbody file
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"An error occurred: {e}; The traceback: {tb_str} ")

        logger.debug(f"Figure generated, save content to {self.fig_mainbody_path}.")


class StructureFigureBuilder(BaseFigureBuilder):
    def __init__(self, task_id: str):
        super().__init__(task_id)
        self.init_structure_fig_path: Path = (
            Path(BASE_DIR)
            / "resources"
            / "latex"
            / "figure_template"
            / "structure_fig.ini.tex"
        )

        self.tex = load_file_as_string(self.init_structure_fig_path)
        self.outlines_path: Path = Path(OUTPUT_DIR) / task_id / "outlines.json"
        # output file:
        self.structure_fig_path: Path = (
            Path(OUTPUT_DIR) / task_id / "latex" / "figs" / "structure_fig.tex"
        )

    # strcutre figure是创建了mainbody_fig_refined.tex，所以一定要第一个运行
    # 后面的builder都是在mainbody_fig_refined.tex上进行修改
    def create_structure_figure(self, input_mainbody_path: Path):
        self.tex = load_file_as_string(self.init_structure_fig_path)
        self.tex += self.color_define()

        # get the top3 citation numbers sections
        l = self.sort_section(input_mainbody_path)
        treecode = self.generate_tree(l)
        self.tex += self.insert(treecode)

        save_result(self.tex, self.structure_fig_path)
        self.insert_to_mainbody(input_mainbody_path, self.fig_mainbody_path)

    def sort_section(self, input_mainbody_path: Path) -> list[tuple[Paragraph, bool]]:
        mainbody = load_file_as_string(input_mainbody_path)
        l = Paragraph.from_mainbody(mainbody)
        ll = sorted(
            enumerate(l), key=lambda x: x[1].content.count(r"\cite{"), reverse=True
        )
        top3_index = [index for index, value in ll[:3]]
        l = [(x, True if i in top3_index else False) for i, x in enumerate(l)]
        return l

    def color_define(self):
        colors = [
            f"\\definecolor{{c{i}}}{{HTML}}{{{x}}}\n"
            for i, x in enumerate(random.sample(self.palette, len(self.palette)))
        ]
        return "\n".join(colors)

    def generate_tree(self, paragraph_list: list[Paragraph]):
        # root
        outlines = Outlines.from_saved(self.outlines_path)
        title = outlines.title
        treecode = f"[\\textbf{{{title}}}, root, ver <insert>]"
        # 1-level node
        res = ""
        textwidth_o1 = max(len(x[0].title) for x in paragraph_list) // 2
        for i, (x, t) in enumerate(paragraph_list):
            color = f"c{i}"
            res += f"[ \\S \\ref{{sec:{x.title}}}.\\ {x.title}, ot1, draw={color}, fill={color}, fill opacity=0.3, text width={textwidth_o1}em,\n"
            if t == False:
                res += "[ \\ \\ {}, edge={transparent},draw=none,fill=none,], \n]"
                continue
            textwidth = max(len(xx.title) for xx in x.sub) // 2 + 2
            for j, y in enumerate(x.sub):
                res += f"[ \\ \\ref{{subsec:{y.title}}} \\ {y.title}, ot2, draw={color}, fill={color}, fill opacity=0.3, text width={textwidth}em], \n"
            res += "]"
        return treecode.replace("<insert>", res)

    def insert(self, tex):
        origin = r"""
\begin{figure*}[!th]
    \centering
    \resizebox{1\textwidth}{!}
    {
        \begin{forest}
            % forked edges,
            for tree={
                grow=east,
                reversed=true,
                anchor=base west,
                parent anchor=east,
                child anchor=west,
                base=left,
                font=\normalsize,
                rectangle,
                draw=hidden-draw,
                rounded corners,
                align=left,
                minimum width=1em,
                edge+={darkgray, line width=1pt},
                s sep=10pt,
                inner xsep=0pt,
                inner ysep=3pt,
                line width=0.8pt,
                ver/.style={rotate=90, child anchor=north, parent anchor=south, anchor=center},
            }, 
            <insert>
        \end{forest}
    }
    \caption{chapter structure}
    \label{fig:chapter_structure}
\vspace{-0.3cm}
\end{figure*}
    """
        return origin.replace("<insert>", tex)

    def insert_to_mainbody(self, input_mainbody_path: Path, output_mainbody_path: Path):
        """insert the structure figure to the mainbody"""
        paragraph_l = Paragraph.from_mainbody_path(input_mainbody_path)

        intro = "The following sections are organized as shown in \\autoref{fig:chapter_structure}.\n"
        graph_code = "\n\\input{figs/structure_fig}\n"
        insert_pos_for_graph_code = paragraph_l[0].content.find("\n") + 1
        paragraph_l[0].content = (
            paragraph_l[0].content[:insert_pos_for_graph_code]
            + graph_code
            + paragraph_l[0].content[insert_pos_for_graph_code:].strip()
            + intro
        )
        Paragraph.save_to_file(paragraph_l, output_mainbody_path)


class TimeShaftFigureBuilder(BaseFigureBuilder):
    def __init__(self, task_id: str):
        super().__init__(task_id)


class TreeFigureBuilder(BaseFigureBuilder):
    @dataclass
    class Node:
        """Store the tree-structured information"""

        title: str
        child: list = field(default_factory=list)
        list_: list[str] | None = None

    LEAF_X_POS = 0  #
    LEAF_X_POS_DELTA = 3
    PARENT_CHILD_DISTANCE = 2.3

    def __init__(self, task_id: str):
        super().__init__(task_id)
        self.tex: str = ""
        self.list_tex: str = ""
        self.color_tex: str = ""
        self.init_tree_tex_path: Path = (
            Path(BASE_DIR)
            / "resources"
            / "latex"
            / "figure_template"
            / "tree_figure.ini.tex"
        )
        self.figure_dir: Path = Path(OUTPUT_DIR) / task_id / "latex" / "figs"
        self.chat_agent: ChatAgent = ChatAgent(
            TokenMonitor(task_id, "Template figure builder")
        )
        self.prompt_path: Path = (
            Path(BASE_DIR)
            / "resources"
            / "LLM"
            / "prompts"
            / "latex_figure_builder"
            / "extract_tree_architect.md"
        )
        self.leaf_node_counter: int = 0
        self.mindmap_tree_figure_builder = MindMapTreeFigureBuilder(task_id)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
    )
    def extract_architecture(self, paragraph: str) -> tuple:
        prompt = load_prompt(self.prompt_path, context=paragraph)
        response = self.chat_agent.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
        response = response.replace("```json", "").replace("```", "")
        try:
            ans = (
                re.search(r"(?<=<answer>)(.*?)(?=<\/answer>)", response, re.DOTALL)
                .group(0)
                .strip()
            )
            score = (
                re.search(r"(?<=<score>)(.*?)(?=<\/score>)", response, re.DOTALL)
                .group(0)
                .strip()
            )
            caption = (
                re.search(r"(?<=<caption>)(.*?)(?=<\/caption>)", response, re.DOTALL)
                .group(0)
                .strip()
            )
            archi = self._wrap_node(json.loads(ans))
        except (json.JSONDecodeError, AttributeError, AssertionError) as e:
            logger.error(str(e) + "\n" + response)
            raise ValueError

        return archi, score, caption

    @staticmethod
    def print_node(node: Node, indent: int = 0):
        """Recursively print the node"""
        print("\t" * indent, node.title)
        if node.list_:
            print("\t" * (indent + 1), "---", node.list_)
        for child in node.child:
            TreeFigureBuilder.print_node(child, indent + 1)

    def _wrap_node(self, node_dic: dict) -> Node:
        assert ("child" in node_dic or "list_" in node_dic) and "title" in node_dic
        node = TreeFigureBuilder.Node(title=node_dic["title"])
        for child in node_dic.get("child", []):
            node.child.append(self._wrap_node(child))
        node.list_ = node_dic.get("list_", None)
        return node

    def _gen_leaf_node_latex(self, node: Node) -> tuple:
        pos = (self.LEAF_X_POS, 0)
        self.LEAF_X_POS += self.LEAF_X_POS_DELTA
        leaf_node_name = f"L{str(pos[0]).replace('.', '_')}"
        self.tex += f"\\node[nodeB, fill={{teal!15}}, anchor=north] ({leaf_node_name}) at ({pos[0]}, {pos[1]}) {{{node.title}}};\n"
        # for list nodes
        list_tex = f"""\\begin{{scope}}[start chain={self.leaf_node_counter} going below]
\\chainin ({leaf_node_name}) [on chain, join];
<list_node>
\\end{{scope}}
<list_line>
\n"""
        # for list nodes
        list_node_tex = ""
        list_line_tex = ""
        for i, list_node in enumerate(node.list_):
            list_node_name = f"l_{pos[0]}_{i}"
            if i == 0:
                list_node_tex += f"\\node[nodeB, on chain, fill=blue!15, xshift=6mm, yshift=-7mm] ({list_node_name}) {{{list_node}}};\n"
            else:
                list_node_tex += f"\\node[nodeB, on chain, fill=blue!15] ({list_node_name}) {{{list_node}}};\n"
            list_line_tex += f"\\draw[->] ($({leaf_node_name}.south west) + (2mm, 0)$) |- ({list_node_name}.west);\n"
        list_tex = list_tex.replace("<list_node>", list_node_tex).replace(
            "<list_line>", list_line_tex
        )
        self.tex += list_tex

        self.leaf_node_counter += 1
        return pos, leaf_node_name

    def gen_node_latex(self, node: Node, level=1) -> tuple:
        if not node.child:
            return self._gen_leaf_node_latex(node)
        childs_pos = []
        childs_name = []
        for child in node.child:
            pos, name = self.gen_node_latex(child, level + 1)
            childs_pos.append(pos)
            childs_name.append(name)

        childs_pos_array = np.array(childs_pos)
        pos_x = np.mean(childs_pos_array[:, 0])
        pos_y = np.mean(childs_pos_array[:, 1]) + self.PARENT_CHILD_DISTANCE
        node_name = f"{level}_{pos_x:.0f}"
        # for node
        if level == 1:  # for root node
            self.tex += f"\\node[nodeB, fill={{teal!30}}, anchor=north, text width=15em] ({node_name}) at ({pos_x}, {pos_y}) {{{node.title}}};\n"
        else:
            self.tex += f"\\node[nodeB, fill={{teal!30}}, anchor=north] ({node_name}) at ({pos_x}, {pos_y}) {{{node.title}}};\n"
        # for Auxiliary point coordinate
        for name, (x, y) in zip(childs_name, childs_pos):
            self.tex += f"\\coordinate ({name}_up) at ({x}, {y + 0.4});\n"
            self.tex += f"\draw ({name}_up) -- ({name}.north);\n"
        for i in range(len(childs_name) - 1):
            self.tex += f"\draw ({childs_name[i]}_up) -- ({childs_name[i + 1]}_up);\n"
        self.tex += f"\draw ({node_name}.south) -- ({pos_x}, {y + 0.4});\n"

        return (pos_x, pos_y), node_name

    def gen_latex_code(self, node: Node, caption: str, file_name: str, label: str):
        self.tex = ""
        self.gen_node_latex(node, level=1)

        tree_tex = load_file_as_string(self.init_tree_tex_path)
        tree_tex = tree_tex.replace("<tree_code>", self.tex)
        tree_tex = tree_tex.replace("<caption>", caption)
        tree_tex = tree_tex.replace("<label>", label)
        save_result(tree_tex, self.figure_dir / file_name)

    def gen_tree_code(self, node: Node, caption: str, file_name: str, label: str):
        if len(node.child) >= 5:
            self.mindmap_tree_figure_builder.gen_latex_code(
                node, caption, file_name, label
            )
        else:
            self.gen_latex_code(node, caption, file_name, label)

    def extract_section_words(self, paragraph: Paragraph) -> str:
        pattern = (
            r"\\section{"
            + re.escape(paragraph.title)
            + r"\}(?:\s*\\label\{[^}]*\})?\s*(.*?)\\subsection\{"
        )
        section_words = (
            re.search(pattern, paragraph.content, re.DOTALL).group(1).strip()
        )
        return section_words

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def add_intro(
        self, paragraph: Paragraph, caption: str, image_label: str, chat: ChatAgent
    ):
        section_words = self.extract_section_words(paragraph)
        prompt = load_prompt(
            Path(BASE_DIR)
            / "resources"
            / "LLM"
            / "prompts"
            / "latex_figure_builder"
            / "add_intro.md",
            mainbody_text=section_words,
            image_description=caption,
            image_label=image_label,
        )
        res = chat.remote_chat(prompt)
        try:
            ans = re.findall(r"<answer>(.*?)</answer>", res, re.DOTALL)[0]
            ans = clean_chat_agent_format(ans)
            ans = re.sub(
                r"\\autoref\{[^}]*\}", f"\\\\autoref{{fig:{image_label}}}", ans
            )  # 修正autoref避免报错
        except:
            logger.error(
                f"Failed to get answer from the chat agent. The response is: {res}"
            )
            logger.error(f"Prompt: {prompt}")
            raise Exception("Failed to get answer from the chat agent")
        ans += f"\n\\input{{figs/{image_label}}}\n"  # add input code
        if len(paragraph.content.strip().split()) > 5000:
            logger.error(f"paragraph is too long: {paragraph.content}")
        paragraph.content = paragraph.content.replace(section_words, ans, 1)

    def run(self, mainbody_path: Path, num_workers=8):
        paragraph_l = Paragraph.from_mainbody_path(mainbody_path)
        content_l = [paragraph.content for paragraph in paragraph_l[2:-1]]

        # extract tree archi in parallel
        archi_and_score = [None] * len(content_l)
        pbar = tqdm(total=len(content_l), desc="Extracting tree figure key info...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(self.extract_architecture, paragraph): idx
                for idx, paragraph in enumerate(content_l)
            }
            for future in as_completed(future_to_index):
                result = future.result()
                idx = future_to_index[future]
                archi_and_score[idx] = result
                pbar.update(1)
        pbar.close()
        # choose the highest score.
        archi_with_max_score_index, archi_with_max_score = max(
            enumerate(archi_and_score, start=2), key=lambda x: x[1][1]
        )
        title_of_section = paragraph_l[archi_with_max_score_index].title
        archi, caption = archi_with_max_score[0], archi_with_max_score[2]
        tree_figure_file_name = f"tree_figure_{title_of_section[:5].encode('ascii', 'ignore').decode('ascii')}"
        logger.info(
            "scores in each chapter: "
            + ", ".join([score for archi, score, caption in archi_and_score])
        )
        # generate the figure latex code.
        self.gen_tree_code(
            archi, caption, tree_figure_file_name + ".tex", tree_figure_file_name
        )
        # add intro to the figure
        self.add_intro(
            paragraph_l[archi_with_max_score_index],
            caption,
            tree_figure_file_name,
            self.chat_agent,
        )

        Paragraph.save_to_file(paragraph_l, mainbody_path)


class TinyTreeFigureBuilder(TreeFigureBuilder):
    def __init__(self, task_id):
        super().__init__(task_id)
        self.init_tree_tex_path: Path = (
            Path(BASE_DIR)
            / "resources"
            / "latex"
            / "figure_template"
            / "tiny_tree_figure.ini.tex"
        )
        self.prompt_path: Path = (
            Path(BASE_DIR)
            / "resources"
            / "LLM"
            / "prompts"
            / "latex_figure_builder"
            / "extract_tiny_tree_architect.md"
        )
        self.paper_dir: Path = Path(OUTPUT_DIR) / task_id / "papers"

    def define_color(self):
        self.reset_palette()
        self.color_tex = ""
        self.color_tex += f"\definecolor{{color_root}}{{HTML}}{{{self.palette[0]}}}\n"
        self.color_tex += f"\definecolor{{color_node}}{{HTML}}{{{self.palette[1]}}}\n"
        self.color_tex += f"\definecolor{{color_list}}{{HTML}}{{{self.palette[2]}}}\n"

    def _gen_leaf_node_latex(self, node: TreeFigureBuilder.Node) -> str:
        leaf_node_name = f"leaf_{self.leaf_node_counter}"
        tree_tex = f"child {{node[nodeB, fill=color_node!40] ({leaf_node_name}) {{{node.title}}}}}"
        list_tex = f"""\\begin{{scope}}[start chain={self.leaf_node_counter} going below]
\\chainin ({leaf_node_name}) [on chain, join];
<list_node>
\\end{{scope}}
<list_line>
\n"""
        # for list nodes
        list_node_tex = ""
        list_line_tex = ""
        for i, list_node in enumerate(node.list_):
            list_node_name = f"l{self.leaf_node_counter}{i}"
            if i == 0:
                list_node_tex += f"\\node[nodeL, on chain, fill=color_list!15, xshift=6mm, yshift=-5mm] ({list_node_name}) {{{list_node}}};\n"
            else:
                list_node_tex += f"\\node[nodeL, on chain, fill=color_list!15] ({list_node_name}) {{{list_node}}};\n"
            list_line_tex += f"\\draw[->] ($({leaf_node_name}.south west) + (2mm, 0)$) |- ({list_node_name}.west);\n"
        list_tex = list_tex.replace("<list_node>", list_node_tex).replace(
            "<list_line>", list_line_tex
        )

        self.leaf_node_counter += 1
        self.list_tex += list_tex + "\n"
        return tree_tex

    def _gen_node_latex(self, node: TreeFigureBuilder.Node) -> str:
        if not node.child:
            return self._gen_leaf_node_latex(node)

        tree_tex = (
            f"child {{node[nodeB, fill=color_node!40] {{{node.title}}} <children> }} \n"
        )
        children_tex = ""
        for child in node.child:
            children_tex += self._gen_leaf_node_latex(child)
        tree_tex = tree_tex.replace("<children>", children_tex)
        return tree_tex

    def gen_node_latex(self, node: TreeFigureBuilder.Node) -> str:
        tree_code = f"\\node[root, fill=color_root!60] {{{node.title}}} <children> ;\n"
        children_tree_code = ""
        self.list_tex = ""
        for child in node.child:
            self.tree_tex = ""
            children_tree_code += self._gen_node_latex(child)
        tree_code = tree_code.replace("<children>", children_tree_code)
        tree_code += self.list_tex
        return tree_code

    def _load_all_bib_name(self) -> dict:
        papers = Paper.from_dir(self.paper_dir)
        res = {paper.bib_name: paper for paper in papers}
        return res

    def _find_bib_name(self, paragraph: str, all_bib_names: list) -> list[str]:
        match_list = re.findall(r"\\cite\{([^}]+)\}", paragraph)
        match_list = [fuzzy_match(x, all_bib_names)[0] for x in match_list]
        return [cite.strip() for group in match_list for cite in group.split(",")]

    def extract_attri_tree_from_paragraph(self, paragraph: str) -> str:
        all_bib_name_dict = self._load_all_bib_name()
        bib_name_in_paragraph = self._find_bib_name(paragraph, all_bib_name_dict.keys())
        attri_trees = "\n\n".join(
            [
                bib_name
                + ":\n"
                + json.dumps(all_bib_name_dict[bib_name].attri, indent=4)
                for bib_name in bib_name_in_paragraph
            ]
        )
        return attri_trees

    def extract_architecture(self, paragraph: str):
        trees = self.extract_attri_tree_from_paragraph(paragraph)
        prompt = load_prompt(self.prompt_path, context=paragraph, trees=trees)
        response = self.chat_agent.remote_chat(prompt, model=ADVANCED_CHATAGENT_MODEL)
        response = response.replace("```json", "").replace("```", "")
        try:
            ans = (
                re.search(r"(?<=<answer>)(.*?)(?=</answer>)", response, re.DOTALL)
                .group(0)
                .strip()
                .replace("\\", "\\\\")
            )
            score = (
                re.search(r"(?<=<score>)(.*?)(?=</score>)", response, re.DOTALL)
                .group(0)
                .strip()
            )
            caption = (
                re.search(r"(?<=<caption>)(.*?)(?=</caption>)", response, re.DOTALL)
                .group(0)
                .strip()
            )
            archi = self._wrap_node(json.loads(ans))
        except (json.JSONDecodeError, AttributeError, AssertionError) as e:
            logger.error(str(e) + "\n" + response)
            raise ValueError()

        return archi, score, caption

    def gen_latex_code(
        self, node: TreeFigureBuilder.Node, caption: str, file_name: str, label: str
    ):
        self.define_color()
        tree_node_tex = self.gen_node_latex(node)

        tree_tex = load_file_as_string(self.init_tree_tex_path)
        tree_tex = tree_tex.replace("<define_color>", self.color_tex)
        tree_tex = tree_tex.replace("<tree_code>", tree_node_tex)
        tree_tex = tree_tex.replace("<caption>", caption)
        tree_tex = tree_tex.replace("<label>", label)
        save_result(tree_tex, self.figure_dir / file_name)

    def insert_to_survey(self, title: str, file_name: str):
        survey_tex = load_file_as_string(self.fig_mainbody_path)

        # new_survey_lines = []
        # for line in survey_tex.splitlines():
        #     if r"\subsection{" + title + "}" in line:
        #         new_survey_lines.append(line+
        #         f"\n\\input{{figs/{file_name}}}")
        #     else:
        #         new_survey_lines.append(line)
        # survey_tex = "\n".join(new_survey_lines)

        survey_tex = "\n".join(
            line
            + (
                f"\n\\input{{figs/{file_name}}}\n"
                if r"\subsection{" + title + "}" in line
                else ""
            )
            for line in survey_tex.splitlines()
        )
        save_result(survey_tex, self.fig_mainbody_path)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def add_intro(self, section_title: str, caption: str, image_label: str):
        paragraph_l = Paragraph.from_mainbody_path(self.fig_mainbody_path)
        done = False
        for i, section in enumerate(paragraph_l):
            for j, subsection in enumerate(section.sub):
                if subsection.title == section_title:
                    section_content = subsection.content

                    prompt = load_prompt(
                        Path(BASE_DIR)
                        / "resources"
                        / "LLM"
                        / "prompts"
                        / "latex_figure_builder"
                        / "add_intro.md",
                        mainbody_text=section_content,
                        image_description=caption,
                        image_label=image_label,
                    )
                    res = self.chat_agent.remote_chat(prompt)
                    try:
                        ans = re.findall(r"<answer>(.*?)</answer>", res, re.DOTALL)[0]
                        ans = clean_chat_agent_format(ans)
                        ans = re.sub(
                            r"\\autoref\{[^}]*\}",
                            f"\\\\autoref{{fig:{image_label}}}",
                            ans,
                        )  # 修正autoref避免报错
                    except:
                        logger.error(
                            f"Failed to get answer from the chat agent. The response is: {res}"
                        )
                        logger.error(f"Prompt: {prompt[:100]}")
                        logger.error(f"Answer: {ans[:100]}")
                        raise Exception("Failed to get answer from the chat agent")
                    ans += f"\n\\input{{figs/{image_label}}}\n"  # add input code
                    paragraph_l[i].content = paragraph_l[i].content.replace(
                        section_content, ans, 1
                    )
                    done = True
                    if len(paragraph_l[i].content.strip().split()) > 5000:
                        logger.error(f"paragraph is too long: {paragraph_l[i].content}")
        if not done:
            logger.debug(f"Unable to find section whose title is {section_title}")
        Paragraph.save_to_file(paragraph_l, self.fig_mainbody_path)

    def run(self, num_workers=8):
        paragraph_l = Paragraph.from_mainbody_path(self.fig_mainbody_path)
        sub_paragraph_l = [sub for parag in paragraph_l[2:-1] for sub in parag.sub]
        content_l = [sub_paragraph.content for sub_paragraph in sub_paragraph_l]

        # extract tree archi in parallel
        archi_and_score = [None] * len(content_l)
        pbar = tqdm(total=len(content_l), desc="Extracting tree figure key info...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(self.extract_architecture, paragraph): idx
                for idx, paragraph in enumerate(content_l)
            }
            for future in as_completed(future_to_index):
                result = future.result()
                idx = future_to_index[future]
                archi_and_score[idx] = result
                pbar.update(1)
        pbar.close()
        logger.info(
            "scores in each chapter: "
            + ", ".join([score for archi, score, caption in archi_and_score])
        )

        # start to generate tiny tree figure on each section.
        start_of_section = 0
        for i, parag in tqdm(
            enumerate(paragraph_l[2:-1]),
            total=len(paragraph_l[2:-1]),
            desc="generating each tiny tree figure...",
        ):
            # 防止跟retrieve figs冲突
            if "fig:retrieve_fig" in parag.content:
                continue

            section_length = len(parag.sub)
            archi_with_max_score_index, archi_with_max_score = max(
                enumerate(
                    archi_and_score[
                        start_of_section : start_of_section + section_length
                    ]
                ),
                key=lambda x: x[1][1],
            )
            # if children num of root is less than 3, skip this generation.
            if len(archi_with_max_score[0].child) < 3:
                continue
            title_of_subsection = sub_paragraph_l[
                start_of_section + archi_with_max_score_index
            ].title
            tree_figure_file_name = f"tiny_tree_figure_{i}"
            # add intro in mainbody
            self.add_intro(
                title_of_subsection, archi_with_max_score[2], tree_figure_file_name
            )
            # generate figure.tex file
            self.gen_latex_code(
                archi_with_max_score[0],
                archi_with_max_score[2],
                tree_figure_file_name + ".tex",
                tree_figure_file_name,
            )
            start_of_section += section_length


class MindMapTreeFigureBuilder(BaseFigureBuilder):
    NODE_DISTANCE = 40
    LEVEL_TRANPARENCY = ["0", "50", "80", "90"]
    TEXT_WIDTH_RATIO_FOR_LIST = 0.4

    def __init__(self, task_id: str):
        super().__init__(task_id)
        self.tex: str = ""
        self.list_tex: str = ""
        self.color_tex: str = ""
        self.palette: list = [
            "a44c34",
            "bc966f",
            "b9ac99",
            "96a25f",
            "f3e6f1",
            "fdcee4",
            "c77fa1",
            "c1c8cd",
            "727b83",
        ]
        self.figure_dir: Path = Path(OUTPUT_DIR) / task_id / "latex" / "figs"
        self.init_tree_tex_path: Path = (
            Path(BASE_DIR)
            / "resources"
            / "latex"
            / "figure_template"
            / "mindmap_tree_figure.ini.tex"
        )
        self.leaf_node_counter: int = 0

    def define_color(self):
        self.reset_palette()
        self.color_tex = ""
        for i, color in enumerate(self.palette):
            self.color_tex += f"\\definecolor{{c{i}}}{{HTML}}{{{color}}}\n"

    def _get_quardrant(self, angle: float) -> int:
        angle = int(angle) % 360
        if angle < 90:
            return 0
        if angle < 180:
            return 1
        if angle < 270:
            return 2
        if angle < 360:
            return 3
        logger.error(f"angle {angle} is not in 0-360")

    def calculate_new_angle(self, x_angle: float, y_angle: float) -> float:
        """
        一个点相对于原点移动两次，每次只知道移动方向相对于x轴的角度，求最终点相对于原点的角度
        """
        # 将角度转换为弧度
        x_rad = math.radians(x_angle)
        y_rad = math.radians(y_angle)

        # 计算总的角度变化
        x_total = math.cos(x_rad) + math.cos(y_rad)
        y_total = math.sin(x_rad) + math.sin(y_rad)

        # 计算最终点相对于原点的角度
        final_angle = math.degrees(math.atan2(y_total, x_total))

        # 确保角度在 [0, 360) 范围内
        final_angle = final_angle % 360

        return final_angle

    def _gen_leaf_node_latex(
        self,
        node: TreeFigureBuilder.Node,
        color: str,
        angle: float,
        level: int,
        list_angle: float,
    ) -> str:
        node_name = f"n{level}_{int(angle)}"
        tree_tex = f"child[concept, concept color={color}!{self.LEVEL_TRANPARENCY[level]}!black, grow={angle}] {{ node[concept] ({node_name}) {{{node.title}}} \n }}\n"
        # for list node
        angle_ = self.calculate_new_angle(list_angle, angle)
        quardrant = self._get_quardrant(angle_)
        position = ["north east", "north west", "south west", "south east"][quardrant]
        anchor = ["south west", "south east", "north east", "north west"][quardrant]
        shift = ["(1em, 0)", "(0, 0.5em)", "(-1em, 0)", "(0, -0.5em)"][quardrant]
        text_length = (
            max([len(text) for text in node.list_]) * self.TEXT_WIDTH_RATIO_FOR_LIST
        )
        list_tex = f"""\info[{text_length}]{{{node_name}.{position}}}{{above, anchor={anchor}, shift={{{shift}}}}}{{<items>}}\n"""
        items = ""
        for list_str in node.list_:
            items += f"\\item {list_str}\n"
        list_tex = list_tex.replace("<items>", items)
        self.list_tex += list_tex
        return tree_tex

    def _gen_node_latex(
        self,
        node: TreeFigureBuilder.Node,
        color: str,
        angle: float = 0,
        level=1,
        list_angle: float = 0,
    ) -> str:
        if not node.child:
            return self._gen_leaf_node_latex(node, color, angle, level + 1, list_angle)
        node_name = f"n{level}_{int(angle)}"
        tree_tex = f"child[concept, concept color={color}!{self.LEVEL_TRANPARENCY[level]}!black, grow={angle}] {{ node[concept] ({node_name}) {{{node.title}}} \n<child_tex> }}\n"
        child_tex = ""
        for child_angle, child in zip(
            np.linspace(
                angle - len(node.child) * self.NODE_DISTANCE / 2,
                angle + len(node.child) * self.NODE_DISTANCE / 2,
                len(node.child),
            ),
            node.child,
        ):
            child_tex += self._gen_node_latex(
                child, color, angle=child_angle, level=level + 1, list_angle=list_angle
            )
        tree_tex = tree_tex.replace("<child_tex>", child_tex)
        return tree_tex

    def gen_node_latex(self, node: TreeFigureBuilder.Node) -> str:
        self.list_tex = ""
        tree_tex = []
        for i, (angle, child) in enumerate(
            zip(np.arange(0, 360, 360 / len(node.child)), node.child)
        ):
            child_tex = self._gen_node_latex(
                child, color=f"c{i}", angle=angle, level=1, list_angle=angle
            )
            tree_tex.append(child_tex)

        tree_tex = f"node[root] {{{node.title}}} \n<child_tex>;\n<list_tex>".replace(
            "<child_tex>", "".join(tree_tex)
        )
        tree_tex = tree_tex.replace("<list_tex>", self.list_tex)
        return tree_tex

    def gen_latex_code(
        self, node: TreeFigureBuilder.Node, caption: str, file_name: str, label: str
    ):
        self.define_color()
        tree_code = self.gen_node_latex(node)

        tree_tex = load_file_as_string(self.init_tree_tex_path)
        tree_tex = tree_tex.replace("<define_color>", self.color_tex)
        tree_tex = tree_tex.replace("<tree_code>", tree_code)
        tree_tex = tree_tex.replace("<caption>", caption)
        tree_tex = tree_tex.replace("<label>", label)
        save_result(tree_tex, self.figure_dir / file_name)


class FlowFigureBuilder(BaseFigureBuilder):
    def __init__(self, task_id: str):
        super().__init__(task_id)


# python -m src.modules.latex_handler.latex_figure_builder
if __name__ == "__main__":
    task_id = "2025-02-13-1050_Trans"
    tree = TreeFigureBuilder(task_id=task_id)
    tree.run(mainbody_path=Path(f"{BASE_DIR}/outputs/{task_id}/tmp/mainbody.tex"))
    ttfb = TinyTreeFigureBuilder(task_id=task_id)
    ttfb.run()
