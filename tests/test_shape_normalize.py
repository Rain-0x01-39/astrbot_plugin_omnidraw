"""形状参数规范化纯函数单测（不依赖 astrbot 环境）。

运行: python tests/test_shape_normalize.py
通过 AST 从 providers/base.py 提取目标函数源码执行，绕开 aiohttp/astrbot 依赖。
每个用例覆盖一个独立行为。
"""
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict

BASE_PY = Path(__file__).resolve().parent.parent / "providers" / "base.py"

TARGET_FUNCTIONS = {
    "align_size_to_16",
    "_nearest_gpt1_size",
    "_ratio_to_pixels",
    "normalize_image_shape_params",
}


def load_functions() -> Dict[str, Any]:
    tree = ast.parse(BASE_PY.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in TARGET_FUNCTIONS
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, str(BASE_PY), "exec")
    namespace: Dict[str, Any] = {"re": re, "Dict": Dict, "Any": Any}
    exec(code, namespace)
    return namespace


def main() -> int:
    ns = load_functions()
    norm = ns["normalize_image_shape_params"]
    align = ns["align_size_to_16"]
    failures: list = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            failures.append(f"  FAIL {name}: got {got!r}, want {want!r}")
        else:
            print(f"  ok {name}")

    print("== 16 对齐 ==")
    check("1500x900 -> 16 对齐", align("1500x900"), "1504x896")
    check("非 WxH 透传", align("2K"), "2K")

    print("== size 像素优先 ==")
    check("像素 + 比例 -> 只留像素", norm("gpt-image-2", {"size": "1536x864", "aspect_ratio": "1:1"}),
          {"size": "1536x864"})
    check("像素 + 比例 + resolution -> resolution 被丢",
          norm("gpt-image-2", {"size": "1500x900", "aspect_ratio": "16:9", "resolution": "2K"}),
          {"size": "1504x896"})
    check("gpt-2 任意像素直传", norm("gpt-image-2", {"size": "1500x900"}), {"size": "1504x896"})
    check("gpt-1 非档位像素 -> 档位映射", norm("gpt-image-1.5", {"size": "1536x864"}), {"size": "1536x1024"})

    print("== aspect_ratio -> 像素 ==")
    check("gpt-2 16:9 -> 1824x1024", norm("gpt-image-2", {"aspect_ratio": "16:9"}), {"size": "1824x1024"})
    check("gpt-1 16:9 -> 1536x1024", norm("gpt-image-1.5", {"aspect_ratio": "16:9"}), {"size": "1536x1024"})
    check("gpt-2 21:9 -> 2384x1024", norm("gpt-image-2", {"aspect_ratio": "21:9"}), {"size": "2384x1024"})
    check("中文冒号", norm("gpt-image-2", {"aspect_ratio": "16：9"}), {"size": "1824x1024"})
    check("比例归一 10:1 -> 3:1", norm("gpt-image-2", {"aspect_ratio": "10:1"}), {"size": "3072x1024"})
    check("保留其它参数", norm("gpt-image-2", {"aspect_ratio": "16:9", "quality": "high"}),
          {"size": "1824x1024", "quality": "high"})

    print("== tier 档位 ==")
    check("2K + 16:9 组合", norm("gpt-image-2", {"size": "2K", "aspect_ratio": "16:9"}), {"size": "3648x2048"})
    check("4K clamp 到 3840", norm("gpt-image-2", {"size": "4K"}), {"size": "3840x3840"})

    print("== 非法/空输入 ==")
    check("空 kwargs", norm("gpt-image-2", {}), {})
    check("乱码", norm("gpt-image-2", {"size": "abc", "aspect_ratio": "xyz"}), {})
    check("只有 resolution 被清", norm("gpt-image-2", {"resolution": "2K"}), {})

    if failures:
        print(f"\nFAILED {len(failures)}:")
        for failure in failures:
            print(failure)
        return 1
    print("\nall passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
