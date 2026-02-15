from typing import List

import sympy as sp
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class MathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(..., description="Mathematical expression to evaluate")


@tool(
    "math",
    args_schema=MathArgs,
    return_direct=True,
    description="""
    A tool for evaluating arithmetic and mid-level math expressions (fractions, exponents, roots, trig).
    Input should be a valid mathematical expression.
    Examples: "2 + 2", "sin(pi/4) + cos(pi/4)", "sqrt(16) + 3^2", "(125 * 1.07) cm", "45% of 220"
    """,
)
def math_evaluator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    try:
        parsed_expr = parse_expr(
            expression,
            evaluate=True,
            transformations=standard_transformations
            + (
                implicit_multiplication_application,
                convert_xor,
            ),
        )
        simplified_expr = sp.simplify(parsed_expr)
        if simplified_expr.is_Rational and simplified_expr.is_integer is False:
            return f"{simplified_expr} (~{simplified_expr.evalf()})"
        return str(simplified_expr)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not evaluate expression: {exc}") from exc
