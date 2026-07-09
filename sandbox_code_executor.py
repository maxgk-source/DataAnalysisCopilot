"""
Sichere Code-Ausfuehrung fuer den Data Analysis Copilot.

Der Agent erzeugt Python-Code, aber dieser Code wird nicht lokal ausgefuehrt.
Stattdessen wird ein E2B-Code-Interpreter gestartet, bekommt nur den aktuellen
DataFrame als CSV und liefert Text, Tabellen und Diagramme zurueck.
"""

import ast
import base64
import io
import os
import re
import textwrap
from typing import Any, Optional

import pandas as pd


_SANDBOX_SETUP = """
import base64, io
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

df = pd.read_csv('/home/user/data.csv')
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
print(f"Loaded sandbox DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
"""


_CAPTURE = """
try:
    import matplotlib.pyplot as _plt, io as _io, base64 as _b64
    if _plt.get_fignums():
        _buf = _io.BytesIO()
        _plt.gcf().savefig(_buf, format='png', bbox_inches='tight', dpi=150)
        with open('/home/user/_fig.b64', 'w') as _f:
            _f.write(_b64.b64encode(_buf.getvalue()).decode('ascii'))
        _plt.close('all')
except Exception as _e:
    print(f"[figure capture error: {_e}]")

try:
    import pandas as _pd
    _result_obj = globals().get('result', None)
    if hasattr(_result_obj, 'to_csv'):
        if isinstance(_result_obj, _pd.Series):
            _result_obj = _result_obj.to_frame()
        _result_obj.to_csv('/home/user/_result.csv', index=False)
    elif _result_obj is not None:
        print(f"result = {_result_obj}")
except Exception as _e:
    print(f"[result capture error: {_e}]")
"""


class SandboxUnavailableError(RuntimeError):
    """Raised when E2B is not configured or not installed."""


class UnsafeGeneratedCodeError(ValueError):
    """Raised when generated code contains blocked operations."""


def _load_dotenv_if_available() -> None:
    """Laedt .env wie im E2B-Beispiel, falls python-dotenv installiert ist."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


_ALLOWED_IMPORT_ROOTS = {
    "math",
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "statistics",
}

_BLOCKED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "vars",
}

_BLOCKED_ATTRIBUTE_ROOTS = {
    "builtins",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}

_BLOCKED_METHOD_NAMES = {
    "genfromtxt",
    "imread",
    "load",
    "load_dataset",
    "loadtxt",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_hdf",
    "read_json",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "save",
    "savefig",
    "savetxt",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
}


def _clean_code_lines(lines: list[str]) -> str:
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.rstrip()
        stripped = cleaned_line.strip()
        if stripped.startswith("```") or stripped == "`":
            continue

        # Entfernt typische nummerierte Listen aus LLM-Antworten:
        # "1. code", "1) code" oder "1: code".
        numbered = re.match(r"^(\s*)\d+[\.\):]\s+(.*)$", cleaned_line)
        if numbered:
            cleaned_line = f"{numbered.group(1)}{numbered.group(2)}"

        cleaned_lines.append(cleaned_line)

    return textwrap.dedent("\n".join(cleaned_lines)).strip().strip("`").strip()


def strip_code_fences(text: str) -> str:
    """Extrahiert Python-Code aus einer LLM-Antwort."""
    fenced = re.search(r"```\s*(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return _clean_code_lines(fenced.group(1).splitlines())

    cleaned_text = _clean_code_lines(text.strip().splitlines())
    try:
        ast.parse(cleaned_text)
        return cleaned_text
    except SyntaxError:
        pass

    lines = text.strip().splitlines()
    code_lines = []
    code_started = False
    code_start_patterns = (
        "import ",
        "from ",
        "df",
        "result",
        "print(",
        "plt.",
        "sns.",
        "fig",
        "ax",
        "if ",
        "for ",
        "while ",
        "try:",
        "with ",
    )

    for line in lines:
        raw_line = line.rstrip()
        stripped = raw_line.strip()
        if not stripped:
            if code_started:
                code_lines.append("")
            continue

        if stripped.startswith("```") or stripped == "`":
            continue

        numbered = re.match(r"^\d+[\.\):]\s+(.*)$", stripped)
        if numbered:
            stripped = numbered.group(1)

        assignment_like = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped)
        if not code_started and not (stripped.startswith(code_start_patterns) or assignment_like):
            continue

        code_started = True
        code_lines.append(raw_line)

    return _clean_code_lines(code_lines) if code_lines else _clean_code_lines(text.strip().splitlines())


def _root_name(node: ast.AST) -> Optional[str]:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _stderr_is_warning_only(stderr: str) -> bool:
    """Erkennt stderr-Ausgaben, die nur Python-Warnungen enthalten."""
    text = stderr.strip()
    if not text:
        return False
    warning_markers = [
        "Warning:",
        "FutureWarning:",
        "DeprecationWarning:",
        "UserWarning:",
        "RuntimeWarning:",
    ]
    error_markers = ["Traceback (most recent call last):", "Error:", "Exception:"]
    return any(marker in text for marker in warning_markers) and not any(
        marker in text for marker in error_markers
    )


def validate_generated_code(code: str) -> None:
    """Blockiert lokale/remote Nebenwirkungen, bevor Code an E2B geht."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise UnsafeGeneratedCodeError(f"Der generierte Code ist kein gueltiges Python: {error}") from error

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _ALLOWED_IMPORT_ROOTS:
                    raise UnsafeGeneratedCodeError(f"Import '{alias.name}' ist in der Sandbox nicht erlaubt.")

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise UnsafeGeneratedCodeError(f"Import aus '{node.module}' ist in der Sandbox nicht erlaubt.")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALL_NAMES:
                raise UnsafeGeneratedCodeError(f"Aufruf von '{node.func.id}' ist in der Sandbox nicht erlaubt.")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_METHOD_NAMES:
                raise UnsafeGeneratedCodeError(
                    f"Aufruf von '{node.func.attr}' ist in der Sandbox nicht erlaubt."
                )
            root = _root_name(node.func)
            if root in _BLOCKED_ATTRIBUTE_ROOTS:
                raise UnsafeGeneratedCodeError(f"Zugriff auf '{root}' ist in der Sandbox nicht erlaubt.")

        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise UnsafeGeneratedCodeError("Dunder-Attribute sind in der Sandbox nicht erlaubt.")
            root = _root_name(node)
            if root in _BLOCKED_ATTRIBUTE_ROOTS:
                raise UnsafeGeneratedCodeError(f"Zugriff auf '{root}' ist in der Sandbox nicht erlaubt.")

        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise UnsafeGeneratedCodeError("Dunder-Namen sind in der Sandbox nicht erlaubt.")


class E2BSandbox:
    """Kleine E2B-Huelle fuer pandas-Analysen."""

    def __init__(self, dataframe: pd.DataFrame, api_key: Optional[str] = None, timeout: int = 600):
        _load_dotenv_if_available()

        if api_key:
            os.environ["E2B_API_KEY"] = api_key

        if not os.getenv("E2B_API_KEY"):
            raise SandboxUnavailableError(
                "E2B_API_KEY fehlt. Lege ihn als Umgebungsvariable oder in .streamlit/secrets.toml ab."
            )

        try:
            from e2b_code_interpreter import Sandbox as _Sandbox
        except ImportError as error:
            raise SandboxUnavailableError(
                "Das Paket 'e2b-code-interpreter' fehlt. Installiere es mit: pip install e2b-code-interpreter"
            ) from error

        if hasattr(_Sandbox, "create"):
            self._sbx = _Sandbox.create()
        else:
            self._sbx = _Sandbox(timeout=timeout)

        self._sbx.files.write("/home/user/data.csv", dataframe.to_csv(index=False))
        self._sbx.run_code(_SANDBOX_SETUP)

    def run(self, code: str) -> dict[str, Any]:
        validate_generated_code(code)
        execution = self._sbx.run_code(code + "\n" + _CAPTURE)

        result: dict[str, Any] = {
            "output": "\n".join(getattr(execution.logs, "stdout", []) or []),
            "figure_bytes": None,
            "table": None,
            "error": None,
            "warning": None,
            "code": code,
        }

        stderr = "\n".join(getattr(execution.logs, "stderr", []) or [])
        if stderr.strip():
            if _stderr_is_warning_only(stderr):
                result["warning"] = stderr
            else:
                result["error"] = stderr

        try:
            b64 = self._sbx.files.read("/home/user/_fig.b64")
            if b64:
                result["figure_bytes"] = base64.b64decode(b64.strip())
                self._sbx.run_code(
                    "import os; os.path.exists('/home/user/_fig.b64') and os.remove('/home/user/_fig.b64')"
                )
        except Exception:
            for item in getattr(execution, "results", []) or []:
                if hasattr(item, "png") and item.png:
                    result["figure_bytes"] = base64.b64decode(item.png)
                    break

        try:
            csv = self._sbx.files.read("/home/user/_result.csv")
            if csv:
                result["table"] = pd.read_csv(io.StringIO(csv))
                self._sbx.run_code(
                    "import os; os.path.exists('/home/user/_result.csv') and os.remove('/home/user/_result.csv')"
                )
        except Exception:
            pass

        return result

    def close(self) -> None:
        try:
            self._sbx.kill()
        except Exception:
            pass


def execute_code_in_sandbox(code: str, dataframe: pd.DataFrame, api_key: Optional[str] = None) -> dict[str, Any]:
    """Fuehrt Code in einer kurzlebigen E2B-Sandbox aus."""
    sandbox = E2BSandbox(dataframe=dataframe, api_key=api_key)
    try:
        return sandbox.run(code)
    finally:
        sandbox.close()
