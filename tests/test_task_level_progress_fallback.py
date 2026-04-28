import importlib
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest import mock

from data_generation.task_level.generation.raw.config import RuntimeConfig


class ProgressFallbackTests(unittest.TestCase):
    def test_progress_module_imports_without_rich_and_uses_tqdm_fallback(self):
        module_name = "data_generation.task_level.generation.raw.progress"
        removed_modules = {
            name: module
            for name, module in list(sys.modules.items())
            if name == module_name or name == "rich" or name.startswith("rich.")
        }
        for name in removed_modules:
            sys.modules.pop(name, None)

        real_import = __import__

        def _import_without_rich(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("rich disabled for test")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=_import_without_rich):
                progress_module = importlib.import_module(module_name)

            runtime_config = RuntimeConfig(
                composite_task="PrepareCoffee",
                num_runs=1,
                model="gemini-3-flash-preview",
                sdk="google-genai",
                project="demo-project",
                location="global",
                temperature=0.5,
                max_workers=1,
                max_retries=2,
            )

            stderr_buffer = StringIO()
            with redirect_stderr(stderr_buffer):
                progress_handles = progress_module._create_progress_handles(
                    runtime_config,
                    disable_progress=False,
                )
                progress_module._close_progress_handles(progress_handles)

            self.assertIsNone(progress_module.RichProgress)
            self.assertIsNone(progress_handles.display)
            with self.assertRaises(RuntimeError):
                progress_module.StaticQueuedTimeElapsedColumn()
        finally:
            sys.modules.pop(module_name, None)
            for name in list(sys.modules):
                if name == "rich" or name.startswith("rich."):
                    sys.modules.pop(name, None)
            sys.modules.update(removed_modules)
