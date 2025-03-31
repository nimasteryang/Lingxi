"""
config

location to store all configuration information
"""

import os
from enum import Enum

import yaml
from dotenv import load_dotenv
from git import Repo
from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyWordCompleter

from agent.constant import PRESET_DIR, PRESET_OPTIONS, RUNTIME_DIR


class RuntimeType(Enum):
    LOCAL = 1, "LOCAL"
    
    def __int__(self):
        return self.value[0]
    
    def __str__(self):
        return self.value[1]
    
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, int):
            for member in cls:
                if member.value[0] == value:
                    return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


PRESET = "gitpython-developers+GitPython@1413.yaml"


assert PRESET in PRESET_OPTIONS, f"{PRESET} not found in {PRESET_DIR}"


def load_env_config():
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_file)


class RuntimeConfig:
    """
    Singleton class to hold the runtime configuration

    Each configuration loading entry point starts with `load_from`
    """

    _instance = None
    initialized = False

    preset = None

    runtime_dir = RUNTIME_DIR
    proj_name = None
    proj_path = None
    issue_desc = None
    commit_head = None

    runtime_type: RuntimeType = None

    def __new__(cls, force_new_instance=False):
        if cls._instance is None or force_new_instance:
            instance = super().__new__(cls)
            instance.__init__()  # Initialize a new instance
            if not force_new_instance:
                cls._instance = instance
            return instance
        return cls._instance

    def load(self, owner, project, commit_id):
        self.proj_name = owner + "/" + project
        self.proj_path = os.path.join(self.runtime_dir, self.proj_name)
        self.commit_head = commit_id

        self.initialized = True
        self.runtime_type = RuntimeType.LOCAL
        self.runtime_setup()

    def load_from_local(self, path):
        self.proj_path = path
        self.proj_name = "/".join(os.path.split("/")[-2:])
        self.initialized = True
        self.runtime_type = RuntimeType.LOCAL
        self.runtime_setup()

    def load_from_preset(self, preset):
        assert preset in PRESET_OPTIONS

        print(f"Loading preset: `{preset}`")

        with open(os.path.join(PRESET_DIR, preset)) as f:
            preset_data = yaml.unsafe_load(f)

        self.preset = preset

        self.proj_name = preset_data.project_name
        self.proj_path = os.path.join(self.runtime_dir, self.proj_name)

        self.issue_desc = preset_data.issue_description
        self.commit_head = preset_data.checkout_commit

        self.initialized = True
        self.runtime_type = RuntimeType.LOCAL
        self.runtime_setup()

    def load_from_dynamic_select_preset(self):
        preset_options = FuzzyWordCompleter(
            PRESET_OPTIONS,
        )
        preset_selected = prompt("Select a present:", completer=preset_options)
        assert preset_selected in PRESET_OPTIONS
        self.load_from_preset(preset_selected)

    def load_from_github_issue_url(self, issue_url):
        from agent.github_utils import (
            get_issue_close_commit,
            get_issue_description,
            parse_github_issue_url,
        )

        owner, project, issue = parse_github_issue_url(issue_url)
        # TODO cache fetch results from github

        if not owner:
            raise ValueError(f"Invalid GitHub issue URL passed in: {issue_url}")

        self.proj_name = owner + "/" + project
        self.proj_path = os.path.join(self.runtime_dir, self.proj_name)
        self.issue_desc = get_issue_description(owner, project, issue)
        self.commit_head = get_issue_close_commit(owner, project, issue)

        checkout_parent = False
        if self.commit_head:
            print(f"Located closing commit @ {self.commit_head} for\n\t{issue_url}")
            checkout_parent = True

        self.initialized = True

        self.runtime_type = RuntimeType.LOCAL

        self.runtime_setup()
        if checkout_parent:
            self.checkout_parent_commit()

    def runtime_setup(self):
        assert self.initialized

        # setup runtime if doesn't exist
        if not os.path.exists(self.runtime_dir):
            print(f"{self.runtime_dir} doesn't exist, creating...")
            os.makedirs(self.runtime_dir)

        if not os.path.exists(self.proj_path):
            git_url = f"https://github.com/{self.proj_name}"
            print(f"Cloning {self.proj_name} to\n\t{self.proj_path}")
            repo = Repo.clone_from(git_url, self.proj_path)
        else:
            repo = Repo(self.proj_path)

        if self.commit_head:
            try:
                repo.git.checkout(self.commit_head)
            except Exception:
                print(f"[E] Unable to checkout commit for {self.proj_name}\n\tUsing default commit")

        # reset repo
        repo.git.reset("--hard")
        repo.git.clean("-xdf")

        self.commit_head = repo.commit().hexsha

    def checkout_parent_commit(self):
        assert os.path.isdir(self.proj_path)

        try:
            repo = Repo(self.proj_path)
        except Exception:
            print(f"[E] unable to initialize {self.proj_name} at {self.proj_path}")

        try:
            parent = repo.commit().parents[0]
            repo.git.checkout(parent.hexsha)
        except Exception:
            print(f"[E] unable to checkout parent for {self.proj_name}")

        self.commit_head = repo.commit().hexsha

    def dump_config(self):
        from agent.sepl_tools import extract_git_diff_container, extract_git_diff_local

        if self.runtime_type == RuntimeType.LOCAL:
            extract_git_diff = extract_git_diff_local
        else:
            raise NotImplementedError

        return {
            "runtime_type": int(self.runtime_type),
            "preset": self.preset,
            "path": self.proj_path,
            "patch": extract_git_diff(),
        }

    def load_from_dump_config(self, config):
        from agent.sepl_tools import apply_git_diff

        if config["runtime_type"] == int(RuntimeType.LOCAL):
            if config["preset"]:
                print("=== Loading config with preset")
                self.load_from_preset(config["preset"])
            else:
                print("=== Loading config with local path")
                self.load_from_local(config["path"])

        print("Applying diff from dump config")
        print(config["patch"])
        print("*" * 50)
        apply_git_diff(config["patch"])

    def pretty_print_runtime(self):
        if self.runtime_type == RuntimeType.LOCAL:
            print("Current configuration type is LOCAL")
            print(f"Runtime Dir: {self.runtime_dir}")
            print(f"Project Name: {self.proj_name}")
            print(f"Project Path: {self.proj_path}")
            print(f"Current Commit: {self.commit_head}")


if __name__ == "__main__":
    rc = RuntimeConfig()
    # config.load_from_dynamic_select_preset()
    rc.load_from_github_issue_url("https://github.com/tpope/vim-vinegar/issues/136")

    rc.pretty_print_runtime()
