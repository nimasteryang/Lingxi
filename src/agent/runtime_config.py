"""
config

location to store all configuration information
"""

import asyncio
import logging
import os
import time
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv
from git import Repo
from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter

# Consolidated imports to avoid having imports in methods
from swerex.runtime.abstract import BashAction, WriteFileRequest

from src.agent.constant import RUNTIME_DIR
from src.agent.github_utils import get_issue_close_commit, get_issue_description, parse_github_issue_url
from src.agent.logging_config import configure_logging, get_logger
from src.agent.swerex_utils import extract_git_diff_swerex_container, load_swe_instance_for_swerex

# Setup logging
logger = get_logger(__name__)


class RuntimeType(Enum):
    """Enum to represent different runtime types with improved implementation"""

    LOCAL = auto()
    SWEREX = auto()

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return self.name


class RuntimeConfig:
    """
    Singleton class to hold the runtime configuration.

    Each configuration loading entry point starts with `load_from`.
    The class manages different runtime environments (local, SWEREx).
    """

    _instance = None

    def __new__(cls):
        """Implement the singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance or create it if it doesn't exist"""
        return cls()

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)"""
        cls._instance = None

    def _initialize(self):
        """Initialize all instance attributes with default values"""
        self.initialized = False
        self.preset = None
        self.runtime_dir = RUNTIME_DIR
        self.proj_name = None
        self.proj_path = None
        self.issue_desc = None
        self.commit_head = None
        self.runtime_type = None
        self.swe_instance = None
        self.swe_rex_deployment = None

    def load(self, owner: str, project: str, commit_id: str) -> None:
        """Load configuration for a GitHub repository"""
        self.proj_name = f"{owner}/{project}"
        self.proj_path = os.path.join(self.runtime_dir, self.proj_name)
        self.commit_head = commit_id

        self.initialized = True
        self.runtime_type = RuntimeType.LOCAL
        self.runtime_setup()

    def load_from_local(self, path: str) -> None:
        """Load configuration from a local path"""
        self.proj_path = path
        self.proj_name = "/".join(os.path.split("/")[-2:])

        self.initialized = True
        self.runtime_type = RuntimeType.LOCAL
        self.runtime_setup()



    def load_from_swe_rex_docker_instance(self, instance_id: str, checkout_commit: str | None = None) -> None:
        """Load configuration from a SWEREx Docker instance"""
        from datasets import load_dataset

        swe_instances = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", cache_dir=RUNTIME_DIR)
        # swe_instances = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test", cache_dir=RUNTIME_DIR)

        found = False
        for entry in swe_instances:
            if entry["instance_id"] == instance_id:
                found = True
                self.swe_instance = entry.copy()
                self.issue_desc = entry["problem_statement"]
                break

        if not found:
            raise ValueError(f"Invalid SWE instance id: {instance_id}")

        self.swe_rex_deployment, self.proj_path = asyncio.run(load_swe_instance_for_swerex(instance_id, checkout_commit))

        self.initialized = True
        self.runtime_type = RuntimeType.SWEREX

    

    def load_from_github_issue_url(self, issue_url: str) -> None:
        """Load configuration from a GitHub issue URL"""
        owner, project, issue = parse_github_issue_url(issue_url)

        if not owner:
            raise ValueError(f"Invalid GitHub issue URL: {issue_url}")

        self.proj_name = f"{owner}/{project}"
        self.proj_path = os.path.join(self.runtime_dir, self.proj_name)
        self.issue_desc = get_issue_description(owner, project, issue)
        self.commit_head = get_issue_close_commit(owner, project, issue)

        checkout_parent = False
        if self.commit_head:
            logger.info(f"Located closing commit @ {self.commit_head} for\n\t{issue_url}")
            checkout_parent = True

        self.initialized = True
        self.runtime_type = RuntimeType.LOCAL
        self.runtime_setup()

        if checkout_parent:
            self.checkout_parent_commit()

    def runtime_setup(self) -> None:
        """Set up the runtime environment"""
        if not self.initialized:
            raise RuntimeError("Configuration is not initialized")

        # Setup runtime if doesn't exist
        os.makedirs(self.runtime_dir, exist_ok=True)

        if not os.path.exists(self.proj_path):
            git_url = f"https://github.com/{self.proj_name}"
            logger.info(f"Cloning {self.proj_name} to\n\t{self.proj_path}")
            repo = Repo.clone_from(git_url, self.proj_path)
        else:
            repo = Repo(self.proj_path)

        if self.commit_head:
            try:
                repo.git.checkout(self.commit_head)
            except Exception as e:
                logger.error(f"Unable to checkout commit for {self.proj_name}: {e}\n\tUsing default commit")

        # Reset repo
        repo.git.reset("--hard")
        repo.git.clean("-xdf")

        self.commit_head = repo.commit().hexsha

    def checkout_parent_commit(self) -> None:
        """Check out the parent commit"""
        if not os.path.isdir(self.proj_path):
            raise ValueError(f"Project path does not exist: {self.proj_path}")

        try:
            repo = Repo(self.proj_path)
        except Exception as e:
            raise RuntimeError(f"Unable to initialize repository at {self.proj_path}: {e}")

        try:
            parent = repo.commit().parents[0]
            repo.git.checkout(parent.hexsha)
        except Exception as e:
            raise RuntimeError(f"Unable to checkout parent commit: {e}")

        self.commit_head = repo.commit().hexsha

    def dump_config(self) -> Dict[str, Any]:
        # TODO: remove dump_config with patch, use replay to replay the actions
        """Dump the configuration to a dictionary"""
        self._ensure_initialized()

        # Directly call the appropriate extract_git_diff method based on runtime type
        if self.runtime_type == RuntimeType.LOCAL:
            patch = self.extract_git_diff_local()
        elif self.runtime_type == RuntimeType.SWEREX:
            # patch = self.extract_git_diff_swerex_wrapper()
            patch = ""
            pass
        else:
            raise ValueError(f"Unsupported runtime type: {self.runtime_type}")

        return {
            "runtime_type": int(self.runtime_type),
            "preset": self.preset,
            "path": self.proj_path,
            "swe_instance_id": self.swe_instance["instance_id"] if self.swe_instance else "",
            "patch": patch,
        }


    def pretty_print_runtime(self) -> None:
        """Print the current runtime configuration"""
        self._ensure_initialized()

        if self.runtime_type == RuntimeType.LOCAL:
            logger.info("Current configuration type is LOCAL")
            logger.info(f"Runtime Dir: {self.runtime_dir}")
            logger.info(f"Project Name: {self.proj_name}")
            logger.info(f"Project Path: {self.proj_path}")
            logger.info(f"Current Commit: {self.commit_head}")
        elif self.runtime_type == RuntimeType.SWEREX:
            logger.info("Current configuration type is SWEREX")
            logger.info(f"SWE Bench instance: {self.swe_instance['instance_id']}")
            logger.info(f"SWEREX deployment: {self.swe_rex_deployment}")
            logger.info(f"SWEREX project path: {self.proj_path}")

    def apply_git_diff(self, patch: str) -> Tuple[int, str]:
        """Apply a git diff to the current repository"""
        self._ensure_initialized()

        # Directly call the appropriate apply_git_diff method based on runtime type
        if self.runtime_type == RuntimeType.LOCAL:
            return self.apply_git_diff_local(patch)
        elif self.runtime_type == RuntimeType.SWEREX:
            return self.apply_git_diff_swerex(patch)
        else:
            raise ValueError(f"Unsupported runtime type: {self.runtime_type}")

    def apply_git_diff_local(self, patch: str) -> Tuple[int, str]:
        """Apply a git diff in a local environment"""
        self._ensure_runtime_type(RuntimeType.LOCAL)

        import subprocess

        GIT_APPLY_CMDS = [
            "git apply --verbose",
            "git apply --verbose --reject",
            "patch --batch --fuzz=5 -p1",
        ]

        misc_ident = str(uuid.uuid1())
        tmp_patch_name = f"tmp_patch_{misc_ident[:4]}.patch"
        tmp_f_name = Path(f"{self.runtime_dir}/tmp/{tmp_patch_name}")

        os.makedirs(tmp_f_name.parent, exist_ok=True)

        with open(tmp_f_name, "w", encoding="utf-8") as f:
            f.write(patch)

        for git_apply_cmd in GIT_APPLY_CMDS:
            process = subprocess.Popen(
                "/bin/bash",
                cwd=self.proj_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                shell=True,
            )
            err_code, _ = process.communicate(f"cat {tmp_f_name} | {git_apply_cmd}\necho $?")
            logger.debug(err_code)
            if err_code.splitlines()[-1].strip() == "0":
                return 0, "Patch successfully applied"
            else:
                logger.warning(f"Failed to apply patch with: {git_apply_cmd}")

        return -2, "Patch failed to apply"

    def apply_git_diff_swerex(self, patch: str) -> Tuple[int, str]:
        """Apply a git diff in a SWEREx environment"""
        self._ensure_runtime_type(RuntimeType.SWEREX)

        swe_rex_runtime = self.swe_rex_deployment.runtime
        misc_ident = str(uuid.uuid1())
        tmp_patch_name = f"tmp_patch_{misc_ident[:4]}.patch"
        dest_f_name = f"/tmp/{tmp_patch_name}"

        # Write the patch file directly to the SWEREX container
        asyncio.run(swe_rex_runtime.write_file(WriteFileRequest(content=patch, path=dest_f_name)))
        # input("Press Enter to continue...")
        GIT_APPLY_CMDS = [
            f"git apply --verbose {dest_f_name}",
            f"git apply --verbose --reject {dest_f_name}",
            f"patch --batch --fuzz=5 -p1 -i {dest_f_name}",
        ]

        for git_apply_cmd in GIT_APPLY_CMDS:
            logger.info(f"Trying applying patch with {git_apply_cmd!r}")
            try:
                cmd_output = asyncio.run(
                    swe_rex_runtime.run_in_session(BashAction(command=git_apply_cmd, check="ignore"))
                )
                logger.debug(f"cmd_output: {cmd_output}")
                if cmd_output.exit_code is None:
                    return 0, "Patch successfully applied"
                else:
                    logger.warning(
                        f"Failed to apply patch: {git_apply_cmd}, exit code: {cmd_output.exit_code}"
                    )
            except Exception as e:
                logger.error(f"Error when applying patch with {git_apply_cmd}: {e}")

        return -2, "Patch failed to apply"

    def extract_git_diff_local(self) -> str:
        """Extract git diff from a local repository"""
        self._ensure_runtime_type(RuntimeType.LOCAL)

        logger.info("Extracting git diff from local repository")

        import subprocess

        process = subprocess.Popen(
            "/bin/bash",
            cwd=self.proj_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
        )
        out, _ = process.communicate("git -c core.fileMode=false diff --exit-code --no-color")
        return out

    def extract_git_diff_swerex_wrapper(self) -> str:
        """Wrapper for extract_git_diff_swerex_container that ensures the correct instance is used"""
        self._ensure_initialized()
        if int(self.runtime_type) != int(RuntimeType.SWEREX):
            raise ValueError(
                f"Expected RuntimeType.SWEREX (value {int(RuntimeType.SWEREX)}), got {self.runtime_type} (value {int(self.runtime_type)})"
            )
        return extract_git_diff_swerex_container(self)

    # Helper methods
    def _ensure_initialized(self) -> None:
        """Ensure that the configuration is initialized"""
        if not self.initialized:
            raise RuntimeError("Configuration is not initialized")

    def _ensure_runtime_type(self, expected_type: RuntimeType) -> None:
        """Ensure that the runtime type matches the expected type"""
        self._ensure_initialized()
        if self.runtime_type != expected_type:
            raise ValueError(f"Expected runtime type {expected_type}, got {self.runtime_type}")


# Simple function to load environment config
def load_env_config():
    """Load environment configuration from .env file"""
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_file)
