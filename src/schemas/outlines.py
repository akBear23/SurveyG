import json
from pathlib import Path
import re

from src.configs.logger import get_logger
from src.modules.utils import load_file_as_string, save_result
import os

logger = get_logger("Outlines")


class SingleOutline:
    def __init__(self, title: str, desc: str, sub: list = []) -> None:
        """Construct single outline

        Args:
            title (str): the title of this outline
            desc (str): the description, also what to write in this outline
            sub (list): point to the subsections
        """
        self.title: str = title
        self.desc: str = desc
        self.sub: list[SingleOutline] = sub

    @staticmethod
    def construct_secondary_outline_from_dict(dic: dict) -> None:
        """construct a secondary outline

        Args:
            dic (dict): a dict which contains the keys "subsection title" and "description"
        """
        return SingleOutline(dic["subsection title"], dic["description"])

    @staticmethod
    def construct_primary_outline_from_dict(dic: dict) -> None:
        """constrcut a primary outline, which contains several secondary outlines

        Args:
            dic (dict): a dict which contains the key "section title", "description" and "subsections"
        """
        dic.setdefault("subsections", [])
        sub = [
            SingleOutline.construct_secondary_outline_from_dict(x)
            for x in dic["subsections"]
        ]
        return SingleOutline(dic["section title"], dic["description"], sub)

    def __str__(self):
        return "\n".join([self.title, self.desc])


class Outlines:
    """The outline architecture of the survey."""

    def __init__(self, title: str, sections: list[SingleOutline]) -> None:
        """Construct the Outlines."""
        self.title: str = title
        self.sections: list[SingleOutline] = sections

    @staticmethod
    def from_saved(file_path: str) -> "Outlines":
        """load from saved json files, that always is a dict, containing "title" and "sections" keys."""
        dic = json.loads(load_file_as_string(file_path))
        title = dic["title"]
        sections = []
        for sec in dic["sections"]:
            sections.append(SingleOutline.construct_primary_outline_from_dict(sec))
        logger.debug("construct outlines from saved path: {}".format(file_path))
        return Outlines(title, sections)

    @staticmethod
    def from_dict(dic: dict):
        """Construct from a dict.

        Args:
            dic (dict): a dict contains "title" and "sections" keys.

        Returns:
            Outlines
        """
        title = dic["title"]
        sections = []
        for sec in dic["sections"]:
            sections.append(SingleOutline.construct_primary_outline_from_dict(sec))
        return Outlines(title, sections)

    def save_to_file(self, file_path: Path):
        """Save the Outlines instance to a JSON file."""
        dic = self.to_dict()  # Convert the Outlines instance to a dictionary
        save_result(json.dumps(dic, indent=4), file_path)
        logger.debug(f"Outlines saved to {file_path}")

    def to_dict(self) -> dict:
        """Return as dict."""

        dic = {"title": self.title, "sections": []}
        for section in self.sections:
            dic["sections"].append(
                {
                    "section title": section.title,
                    "description": section.desc,
                    "subsections": [
                        {
                            "subsection title": subsection.title,
                            "description": subsection.desc,
                        }
                        for subsection in section.sub
                    ],
                }
            )
        return dic

    def __str__(self) -> str:
        """print the Outlines

        Returns:
            str: print each section and subsection info
        """
        res = [self.title]
        for i, sec in enumerate(self.sections):
            res.append(f"{i + 1}. " + sec.__str__())
            for j, subsec in enumerate(sec.sub):
                res.append(f"{i + 1}.{j + 1} " + subsec.__str__())
        return "\n".join(res)

    def serial_no_to_single_outline(self, serial_no_raw: str) -> SingleOutline | None:
        """Given a serial no in a survey, map to the single outline.
        Tings like given "1.1", map to "1.1 xxx, xxxx"

        Args:
            serial_no (str): shaped like "1.1", "2.1" or "5"

        Returns:
            SingleOutline: corresponding single outline.
        """
        try:
            if "." in serial_no_raw:
                serial_no = re.search(r"\d+\.\d*", serial_no_raw).group(0)
                primary_section_index = int(serial_no.split(".")[0])
                secondary_section_index = serial_no.split(".")[1]
                if secondary_section_index != "":
                    secondary_section_index = int(secondary_section_index)
                    return self.sections[primary_section_index - 1].sub[
                        secondary_section_index - 1
                    ]
                else:
                    return self.sections[primary_section_index - 1]
            else:
                serial_no = re.search(r"\d+", serial_no_raw).group(0)
                primary_section_index = int(serial_no)
                return self.sections[primary_section_index - 1]
        except Exception as e:
            logger.error(
                f"Error occurs: {e}, the serial_no_raw is {serial_no_raw}, the serial_no is {serial_no}"
            )


def unitest():
    p = os.path.join("outputs", "2025-01-10-1935_recom", "outlines.json")
    outlines = Outlines.from_saved(p)
    # print(outlines)
    print(outlines.serial_no_to_single_outline("3"))


# python -m src.schemas.outlines
if __name__ == "__main__":
    unitest()
