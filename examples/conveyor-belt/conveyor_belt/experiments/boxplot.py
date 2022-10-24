# -*- coding: utf-8 -*-
#
# Copyright (C) 2022, Saarland University
# Copyright (C) 2022, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import dataclasses as d
import typing as t

import enum


class WhiskerType(enum.Enum):
    IQR_TIMES_1_5 = enum.auto()
    EXTREM_VALUES = enum.auto()
    HH_FANCY = enum.auto()


@d.dataclass(frozen=True)
class Box:
    label: str
    median: float
    upper_quartile: float
    lower_quartile: float
    upper_whisker: float
    lower_whisker: float
    outliers: t.Tuple[float, ...] = ()

    @classmethod
    def from_data(
        cls,
        label: str,
        values: t.Sequence[float],
        whisker_type: WhiskerType = WhiskerType.HH_FANCY,
    ) -> Box:
        assert len(values) % 4 == 0
        sorted_values = list(sorted(values))
        length = len(sorted_values)
        quartile_dividers = [quartile * (length // 4) for quartile in range(4 + 1)]
        median = sorted_values[quartile_dividers[2]]
        upper_quartile = sorted_values[quartile_dividers[3]]
        lower_quartile = sorted_values[quartile_dividers[1]]
        if whisker_type is WhiskerType.IQR_TIMES_1_5:
            interquartile_range = upper_quartile - lower_quartile
            upper_whisker = max(
                (
                    value
                    for value in sorted_values
                    if value < upper_quartile + 1.5 * interquartile_range
                ),
                default=sorted_values[-1],
            )
            lower_whisker = min(
                (
                    value
                    for value in sorted_values
                    if value > lower_quartile - 1.5 * interquartile_range
                ),
                default=sorted_values[0],
            )
            outliers = tuple(
                value
                for value in sorted_values
                if value > upper_whisker or value < lower_whisker
            )
        elif whisker_type is WhiskerType.HH_FANCY:
            upper_whisker = sorted_values[-1]
            lower_whisker = sorted_values[0]
            # use the outliers to show the 95% percentiles
            outliers = (
                sorted_values[int(len(sorted_values) * 0.95)],
                sorted_values[int(len(sorted_values) * 0.05)],
            )
        else:
            assert whisker_type is WhiskerType.EXTREM_VALUES
            upper_whisker = sorted_values[-1]
            lower_whisker = sorted_values[0]
            outliers = ()
        return cls(
            label,
            median,
            upper_quartile,
            lower_quartile,
            upper_whisker,
            lower_whisker,
            outliers,
        )

    @property
    def latex_source(self) -> str:
        outliers = "\n".join(map(str, self.outliers))
        return f"""
            \\addplot+[
                boxplot prepared={{
                    median={self.median},
                    upper quartile={self.upper_quartile},
                    lower quartile={self.lower_quartile},
                    upper whisker={self.upper_whisker},
                    lower whisker={self.lower_whisker},
                }},
            ] table [row sep=newline,y index=0] {{
                {outliers}
            }};
        """


@d.dataclass
class Plot:
    title: str = ""
    y_label: str = ""
    x_label: str = ""
    width: str = "5cm"
    height: str = "4cm"
    boxes: t.List[Box] = d.field(default_factory=list)

    @property
    def latex_source(self) -> str:
        return f"""
        \\begin{{tikzpicture}}[font=\\small]
            \\begin{{axis}}
                [
                    xtick={{{", ".join(map(str, range(1, len(self.boxes) + 1)))}}},
                    xticklabels={{{", ".join(box.label for box in self.boxes)}}},
                    boxplot/draw direction=y,
                    width={self.width},
                    height={self.height},
                    title={{{self.title}}},
                    ylabel={{{self.y_label}}},
                    xlabel={{{self.x_label}}},
                    every axis plot post/.append style={{
                        koehlma-blue,
                        solid,
                        mark=x,
                    }},
                ]
                {"".join(box.latex_source for box in self.boxes)}
            \\end{{axis}}
        \\end{{tikzpicture}}
        """
