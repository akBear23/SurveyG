import os
import json
import re
import ast
from collections import defaultdict
from typing import List, Tuple
from itertools import combinations
from difflib import SequenceMatcher
from src.configs.logger import get_logger
from src.configs.config import ADVANCED_CHATAGENT_MODEL
from src.modules.utils import (
    load_meta_data,
    load_prompt,
)
from src.configs.config import BASE_DIR
from src.models.LLM import ChatAgent

logger = get_logger("src.modules.latex_handler.BaseTableBuilder")


class LatexBaseTableBuilder:
    def __init__(self, chat_agent: ChatAgent = None):
        """
        Base class initializer. This method is intentionally left empty to allow subclasses
        to implement their own initialization logic.
        """
        self.chat_agent = chat_agent if chat_agent is not None else ChatAgent()

    def clear_json_file(self, file_path):
        # 打开文件并清空内容
        with open(file_path, "w") as f:
            # 将空字典写入文件以清空内容
            json.dump([], f)

    def is_the_row_good(self, row: str, splitter: str = "&"):
        elements = row.strip().split(splitter)
        unexpected_element_count = 0
        for one in elements:
            if one.strip() in ["-", ""]:
                unexpected_element_count += 1
        if unexpected_element_count >= 2:
            return False
        return True

    def cite_name_match(self, data_list: List, cite_name: str) -> Tuple:
        """
        Retrieve relevant information from the attribute tree based on the cite name.

        Args:
            data_list: A list used to store the attribute tree.
            cite_name: The cite name of the paper.

        Returns:
            tuple: Output tuple contains information about the method description in the attribute tree.
        """
        for data in data_list:
            if (
                data["bib_name"] == cite_name
                and data["paper_type"] == "method"
                and data["attri"] is not None
                and len(data["attri"]["method"]["method abbreviation"].split()) < 2
            ):
                content = (
                    "method name:"
                    + data["attri"]["method"]["method name"]
                    + "\n"
                    + "method_step: \n "
                    + str(data["attri"]["method"]["method steps"])
                )
                complete_info = str(data["attri"])
                return (
                    complete_info,
                    content,
                    data["title"],
                    data["attri"]["method"]["method name"],
                    data["attri"]["method"]["method abbreviation"],
                    data["bib_name"],
                )
        return None, None, None, None, None, None

    def cite_name_match_count(self, data_list, cite_names):
        count = 0
        for cite_name in cite_names:
            if any(
                data["bib_name"] == cite_name and data["paper_type"] == "method"
                for data in data_list
            ):
                count += 1
        return count

    def cite_name_match_benchmark(self, data_list: List, cite_name):
        info = {}
        for data in data_list:
            if (
                data["bib_name"] == cite_name
                and data["paper_type"] == "benchmark"
                and data["attri"] is not None
                and len(data["attri"]["idea"]["benchmark abbreviation"].split()) < 2
                and self.convert_to_number(data["attri"]["dataset"]["size"]) is not None
                and self.convert_to_number(data["attri"]["dataset"]["size"]) < 10000000
            ):
                info["size"] = data["attri"]["dataset"]["size"]
                info["domain"] = data["attri"]["dataset"]["domain"]
                info["task format"] = data["attri"]["dataset"]["task format"]
                info["metric"] = data["attri"]["metrics"]["metric name"]
                info["bib_name"] = data["bib_name"]
                info["name"] = data["attri"]["idea"]["benchmark abbreviation"]
                # info = (
                # "Background: " + str(data['attri']['background']) + "\n" +
                # "Dataset information: " + str(data['attri']['dataset']) + "\n" +
                # "Metric information: " + str(data['attri']['metrics'])
                # )
                return info
        return None

    def extract_attributes(self, file_content, pri_attribute):
        """
        Extract the required information from the LLM response.
        Args:
            file_content: The response of LLM.

        Returns:
            tuple: A tuple containing the attribute name and its description.
        """
        # 修改为新的正则表达式
        primary_pattern = re.compile(
            r"\[Attribute:\s*(.*?)\]", re.DOTALL
        )  # 匹配 Attribute: Name
        description_pattern = re.compile(
            r"\[Description:\s*(.*?)\]", re.DOTALL
        )  # 匹配 Description: XXX

        # 提取匹配的内容
        primary_match = primary_pattern.search(file_content)
        description_match = description_pattern.search(file_content)

        # 初始化结果
        attribute_name = None
        description_text = None

        # 如果找到Primary Attribute，提取其内容
        if primary_match:
            attribute_name = primary_match.group(1)

        # 如果找到Description，提取其内容
        if description_match:
            description_text = description_match.group(1)

        if attribute_name is None or description_text is None:
            return None
        # 结果字典
        result = {
            "Primary Attribute Name": pri_attribute,
            "Secondary Attribute Name": attribute_name,
            "Description": description_text,
        }
        return result

    def save_attributes(self, attribute_name, description, file_name, type):
        """
        Save the attributes to the specified JSON file.
        """
        directory = os.path.dirname(file_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(file_name):
            data = []
        else:
            with open(file_name, "r") as file:
                data = json.load(file)
        if type == 0:
            new_entry = {
                "Primary Attribute Name": attribute_name,
                "Description": description,
            }
            # 检查是否已经存在相同的Primary Attribute Name
            exists = any(
                item.get("Primary Attribute Name") == attribute_name for item in data
            )
        elif type == 1:
            new_entry = {
                "Secondary Attribute Name": attribute_name,
                "Description": description,
            }
            # 检查是否已经存在相同的Primary Attribute Name
            exists = any(
                item.get("Secondary Attribute Name") == attribute_name for item in data
            )
        # 如果不存在，添加到列表并写入文件
        if not exists:
            data.append(new_entry)
            with open(file_name, "w") as file:
                json.dump(data, file, indent=4)

    # 处理文章的方法
    def process_article(self, result, secondary_attribute_path):
        """
        Load attribute data from the result.
        """
        # # 处理Primary Attribute Name
        # primary_attribute = result.get("Primary Attribute Name")
        # primary_description = result.get("Description1")
        # if primary_attribute and primary_description:
        #     save_attributes(primary_attribute, primary_description, primary_attribute_path, 0)

        # 处理Secondary Attribute Name
        secondary_attribute = result.get("Secondary Attribute Name")
        secondary_description = result.get("Description")
        if secondary_attribute and secondary_description:
            self.save_attributes(
                secondary_attribute, secondary_description, secondary_attribute_path, 1
            )

    def process_data(self, data_list):
        """
        Process a list of dictionaries to extract specific attributes and organize them
        into the desired format, while avoiding duplicate Secondary Attribute Names.
        """
        result = {}
        secondary_attributes = []
        seen_names = set()  # Set to track already processed 'Secondary Attribute Name'

        for item in data_list:
            if item is None:
                continue
            primary_attr = item.get("Primary Attribute Name")
            secondary_attr_name = item.get("Secondary Attribute Name")
            description = item.get("Description")

            # Skip if Secondary Attribute Name is duplicate
            if secondary_attr_name in seen_names:
                continue

            # Add unique Secondary Attribute Name to the set
            seen_names.add(secondary_attr_name)

            # Add the secondary attribute to the list
            secondary_attributes.append(
                {"Name": secondary_attr_name, "Description": description}
            )

            # Set the Primary Attribute for the result dictionary (assuming all Primary Attribute Names are the same)
            if "Primary Attribute" not in result:
                result["Primary Attribute"] = primary_attr

        # Assign the Secondary Attributes list to the result dictionary
        result["Secondary Attributes"] = secondary_attributes

        return result

    def replace_secondary_attributes(self, data_list, attribute_dict):
        """
        Replace 'Secondary Attribute Name' in data_list with corresponding keys from attribute_dict
        if the attribute name exists in attribute_dict values.

        Parameters:
            data_list (list): A list of dictionaries containing 'Secondary Attribute Name'.
            attribute_dict (dict): A dictionary mapping categories to attribute names.

        Returns:
            list: Updated list with 'Secondary Attribute Name' replaced where applicable.
        """
        # Create a reverse mapping of attributes to their categories
        reverse_mapping = {
            attr: key for key, attrs in attribute_dict.items() for attr in attrs
        }

        # Process each dictionary in the data list
        for item in data_list:
            secondary_name = item.get("Secondary Attribute Name")
            if secondary_name in reverse_mapping:
                # Replace the secondary attribute name with the category
                item["Secondary Attribute Name"] = reverse_mapping[secondary_name]

        return data_list

    def extract_and_convert(self, text):
        # 使用正则表达式提取 <Answer> 标签中的内容
        match = re.search(r"<Answer>\s*(\{.*?\})\s*</Answer>", text, re.DOTALL)
        if match:
            content = match.group(1)
            try:
                # 使用 ast.literal_eval 将字符串安全地解析为 Python 字典
                dictionary = ast.literal_eval(content)
                return dictionary
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing content: {e}")
                return None
        else:
            print("No valid content found in <Answer> tags.")
            return None

    def data_convert(self, triplets):
        """
        Convert the raw data into the desired format.
        """
        # 初始化字典
        data = defaultdict(lambda: defaultdict(list))

        # 遍历三元组数据
        for triplet in triplets:
            category = triplet["Category"]
            feature = triplet["Feature"]
            method = triplet["Method"]
            # 将方法添加到相应的特征下
            data[category][feature].append(method)

        # 将默认字典转换为目标数据格式
        final_data = {"Category": [], "Feature": [], "Method": []}

        # 构建最终的字典
        for category, features in data.items():
            final_data["Category"].append(category)
            feature_list = []
            method_list = []
            for feature, methods in features.items():
                feature_list.append(feature)
                method_list.append(methods)
            final_data["Feature"].append(feature_list)
            final_data["Method"].append(method_list)
        return final_data

    def extract_cite_name(self, text: str) -> List[str]:
        """
        Extracts the cite name from a paragraph.
        """
        # 使用正则表达式匹配 \cite{xxx} 中的内容
        result = []
        cite_names = re.findall(r"\\cite\{(.*?)\}", text)
        for name in cite_names:
            if name not in result:
                result.append(name)
        return result

    def load_table_data(self, dir_path):
        """
        Load the source files needed to generate the table.
        """
        data = []
        # 临时存储每个文件的读取结果
        temp_data = []

        # 遍历目录中的文件
        if not os.path.isdir(dir_path):
            print(f"The directory {dir_path} does not exist or is not accessible.")
            return None
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    result = json.load(file)  # 将 JSON 文件内容读取为 Python 字典
                    # 读取数据并按顺序添加
                    dict = {}
                    dict["Category"] = result["Primary Attribute Name"]
                    dict["Feature"] = result["Secondary Attribute Name"]
                    dict["Method"] = result["cite_name"]
                    dict["Order"] = result["order"]  # 获取顺序
                    temp_data.append(dict)

        # 按照 'Order' 排序数据
        temp_data.sort(key=lambda x: x["Order"])  # 按 'Order' 字段排序

        # 将排序后的数据添加到最终结果列表中
        for item in temp_data:
            data.append(
                {
                    "Category": item["Category"],
                    "Feature": item["Feature"],
                    "Method": item["Method"],
                }
            )

        return data

    def extract_section_content(self, tex_file_path: str, section_name: str) -> str:
        """
        Extracts the content of a specific section from a .tex file.

        Args:
            tex_file_path (str): The path to the .tex file to read.
            section_name (str): The name of the section to extract.

        Returns:
            str: The content of the specified section.
        """
        with open(tex_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # 正则表达式匹配指定的section及其内容
        pattern = re.compile(
            r"(\\section\{" + re.escape(section_name) + r"\}.*?)(?=\\section|$)",
            re.DOTALL,
        )
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        else:
            return None
            # return f"Section '{section_name}' not found."

    def extract_section_mainbody(self, tex_file_path: str, section_name: str) -> str:
        """
        Extracts the content of a specific section from a .tex file, excluding the section title and labels.

        Args:
            tex_file_path (str): The path to the .tex file to read.
            section_name (str): The name of the section to extract.

        Returns:
            str: The content of the specified section.
        """
        with open(tex_file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 正则表达式匹配指定的section正文，排除\section和\label部分
        pattern = re.compile(
            r"\\section\{"
            + re.escape(section_name)
            + r"\}(?:\s*\\label\{.*?\})?\s*(.*?)(?=\\subsection|\\section|$)",
            re.DOTALL,
        )

        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        else:
            return None

    def extract_subsection_content(
        self, tex_file_path: str, subsection_name: str
    ) -> str:
        """
        Extracts the content of a specific section from a .tex file.

        Args:
            tex_file_path (str): The path to the .tex file to read.
            section_name (str): The name of the section to extract.

        Returns:
            str: The content of the specified section.
        """
        with open(tex_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        # 正则表达式匹配指定的section及其内容
        pattern = re.compile(
            rf"(\\subsection\{{{re.escape(subsection_name)}\}}.*?)(?=(\\section|\\subsection|$))",
            re.DOTALL,
        )
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        else:
            return None
            # return f"subsection '{subsection_name}' not found."

    def extract_subsections(self, text):
        # 使用正则表达式匹配每个subsection标题及其内容
        # 匹配subsection及其内容
        subsection_pattern = r"(\\subsection\{.*?\}.*?)(?=\\subsection|$)"
        subsections = re.findall(subsection_pattern, text, re.DOTALL)

        title_pattern = r"\\subsection\{(.*?)\}"
        titles = [re.search(title_pattern, sub).group(1) for sub in subsections]
        # 返回subsection标题和内容
        return [sub.strip() for sub in subsections], [title for title in titles]

    def extract_section_title(self, text):
        # 使用正则表达式提取以 \section 开头的段落
        section_pattern = r"\\section\{.*?\}.*?\\label\{.*?\}"
        section_title = re.findall(section_pattern, text, re.DOTALL)
        return section_title[0]

    def extract_subsection_title(self, text):
        # 使用正则表达式提取以 \section 开头的段落
        section_pattern = r"\\subsection\{.*?\}.*?\\label\{.*?\}"
        section_title = re.findall(section_pattern, text, re.DOTALL)
        return section_title[0]

    def supplement_data(self, current_data, dir_path, target_size):
        """
        补充召回数据，确保数据总数达到 target_size。
        :param current_data: 当前召回的数据列表（字典类型）
        :param dir_path: 数据库文件夹路径
        :param target_size: 目标数据数量
        :return: 最终数据列表
        """
        # 从文件夹读取所有数据
        data_list = load_meta_data(dir_path)
        benchmark_list = []
        for data in data_list:
            if data["paper_type"] == "benchmark":
                benchmark_list.append(data)
        # 提取当前召回数据的 bib_name 字段集合
        current_bib_names = {item["bib_name"] for item in current_data}

        # 从数据库中过滤掉与当前已有数据重复的项，并提取指定字段
        remaining_data = []
        for item in benchmark_list:
            if (
                item["bib_name"] not in current_bib_names
                and item["attri"] is not None
                and len(item["attri"]["idea"]["benchmark abbreviation"].split()) < 2
                and self.convert_to_number(item["attri"]["dataset"]["size"]) is not None
                and self.convert_to_number(item["attri"]["dataset"]["size"]) < 10000000
            ):
                info = {
                    "name": item["attri"]["idea"]["benchmark abbreviation"],
                    "size": item["attri"]["dataset"]["size"],
                    "domain": item["attri"]["dataset"]["domain"],
                    "task format": item["attri"]["dataset"]["task format"],
                    "metric": item["attri"]["metrics"]["metric name"],
                    "bib_name": item["bib_name"],
                }
                remaining_data.append(info)

        # 补充数据
        supplemented_data = current_data[:]
        for item in remaining_data:
            if len(supplemented_data) < target_size:
                supplemented_data.append(item)
            else:
                break

        return supplemented_data

    def get_sections(self, survey_path: str) -> List[str]:
        """
        Get the section names of the survey.

        Args:
            survey_path (str): The path to the survey TeX file.

        Returns:
            List[str]: A list of section content strings.
        """
        tex = open(survey_path, "r").read()
        pattern = r"\\section{"
        match_l = list(re.finditer(pattern, tex))
        res = []
        for i in range(len(match_l) - 1):
            section_tex = tex[match_l[i].start() : match_l[i + 1].start()]
            res.append(section_tex)
        return res

    def save_table_file(self, latex_code, output_file):
        # Save the LaTeX code to a .tex file
        with open(output_file, "w") as file:
            file.write(latex_code)

    def generate_description(self, latex_code, content):
        prompt = load_prompt(
            f"{BASE_DIR}/resources/LLM/prompts/latex_table_builder/Table_description.txt",
            Latex=latex_code,
            Content=content,
        )
        result = self.chat_agent.remote_chat(
            text_content=prompt, model=ADVANCED_CHATAGENT_MODEL
        )
        result = self.extract_and_convert(result)
        if result is not None:
            caption = result.get("caption")  # 如果没有'caption'，返回 None
            introductory_sentence = result.get(
                "introductory sentence"
            )  # 如果没有'introductory sentence'，返回 None
            return caption, introductory_sentence
        return None, None

    def get_value_list(self, data):
        list1 = [list(d.values())[1] for d in data]
        list2 = [list(d.values())[2] for d in data]
        list3 = [list(d.values())[3] for d in data]
        return [list1, list2, list3]

    def validity_judge(self, data):
        count = 0
        for element in data:
            if "-" in element:
                count += 1

        # 判断是否超过列表长度的一半
        if count > len(data) / 2:
            return 0
        return 1

    def format_string(self, s):
        if not s:  # 如果是空字符串或None，直接返回
            return s
        words = s.split(" ")
        formatted_words = []
        for word in words:
            if len(word) == 2:  # 如果单词仅由两个字母组成，将其全部大写
                formatted_word = word.upper()
            elif "-" in word:  # 处理包含连字符的单词
                parts = word.split("-")
                formatted_word = "-".join([parts[0].capitalize()] + parts[1:])
            else:
                formatted_word = word.capitalize()
            formatted_words.append(formatted_word)
        return " ".join(formatted_words)

    def calculate_similarity(self, list_of_strings, threshold=0.7):
        """
        计算字符串列表的相似度指标，并跳过异常值。

        Args:
            list_of_strings (list): 字符串列表，每个元素是一个字符串。
            threshold (float): 相似度阈值，高于此值认为是高相似度。

        Returns:
            int: 高相似度字符串对的数量。
            float: 相似度指标（归一化）。
        """

        def preprocess(s):
            """预处理字符串：排序单词，转换为小写"""
            return " ".join(sorted(s.lower().split()))

        def similarity(s1, s2):
            """计算两个字符串的相似度（基于SequenceMatcher）"""
            return SequenceMatcher(None, s1, s2).ratio()

        def is_valid(s):
            """判断字符串是否为有效值（排除异常值）"""
            # 异常值定义：仅由标点符号组成或为空
            return bool(s.strip()) and not re.fullmatch(r"[-_.]+", s.strip())

        # 过滤无效字符串（跳过异常值）
        filtered_strings = [s for s in list_of_strings if is_valid(s)]

        # 预处理字符串
        processed_strings = [preprocess(s) for s in filtered_strings]

        # 计算两两组合的相似度
        high_similarity_pairs = 0
        total_pairs = 0
        for s1, s2 in combinations(processed_strings, 2):
            total_pairs += 1
            if similarity(s1, s2) >= threshold:
                high_similarity_pairs += 1

        # 相似度指标
        similarity_score = high_similarity_pairs / total_pairs if total_pairs > 0 else 0
        return similarity_score

    def convert_to_number(self, number_str):
        """
        Convert a number string, e.g., "10000000" or "1,000,000", to an integer.

        Args:
            number_str (str): The string representation of the number.

        Returns:
            int: The integer value of the number.
        """
        try:
            # Remove any commas in the string
            cleaned_str = number_str.replace(",", "")
            # Convert the cleaned string to an integer
            return int(cleaned_str)
        except ValueError:
            return None

    def parse_outline(self, data):
        # 初始化两个列表
        section_titles = []
        subsection_titles = []

        # 遍历sections提取标题
        for section in data["sections"]:
            # 提取section title
            if "section title" in section:
                section_titles.append(section["section title"])

            # 提取subsection title
            if "subsections" in section:
                for subsection in section["subsections"]:
                    if "subsection title" in subsection:
                        subsection_titles.append(subsection["subsection title"])
        return section_titles, subsection_titles

    def get_sections(self, survey_path: str) -> List[str]:
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

    def get_title(self, section: str) -> str:
        """
        Get the title of the section.
        """
        title = re.findall(r"\\section\{([^}]+)\}", section)[0]
        return title
