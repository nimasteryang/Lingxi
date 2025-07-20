import asyncio
from typing import Tuple, Any
from datasets.arrow_dataset import shutil
from swerex.runtime.abstract import CreateBashSessionRequest, BashAction, Command, WriteFileRequest
from swerex.deployment.docker import DockerDeployment
import os
import uuid
from swerex.deployment.config import DockerDeploymentConfig
from src.agent.constant import DOCKER_MAP_DIR, REPO_MAP_DIR


def extract_git_diff_swerex_container(runtime_config_obj=None):
    # Import locally to avoid circular import
    from src.agent.runtime_config import RuntimeConfig, RuntimeType
    
    # Use provided runtime_config if available, otherwise create a new one
    rc = runtime_config_obj if runtime_config_obj is not None else RuntimeConfig()
    
    print("Extracting git diff from SWEREX container")
    
    if not rc.initialized:
        print("ERROR: RuntimeConfig is not initialized")
        return ""
        
    # Compare by int value instead of direct enum comparison for stability
    if int(rc.runtime_type) != int(RuntimeType.SWEREX):
        print(f"ERROR: Expected RuntimeType.SWEREX (value {int(RuntimeType.SWEREX)}), got {rc.runtime_type} (value {int(rc.runtime_type)})")
        return ""
        
    if not rc.swe_rex_deployment:
        print("ERROR: No SWE-REX deployment available")
        return ""
    
    try:
        swe_rex_runtime = rc.swe_rex_deployment.runtime
        
        # First ensure we're in the right directory
        print("Running 'cd /testbed'")
        cd_result = asyncio.run(swe_rex_runtime.run_in_session(
            BashAction(command="cd /testbed", check="ignore")
        ))
        print(f"cd result: {cd_result.exit_code}")
        
        # Make sure all files are added to git tracking
        print("Running 'git add -A'")
        add_result = asyncio.run(swe_rex_runtime.run_in_session(
            BashAction(command="git add -A", check="ignore")
        ))
        print(f"git add result: {add_result.exit_code}")
        
        # Get the diff
        print("Running git diff")
        git_diff_result = asyncio.run(swe_rex_runtime.run_in_session(
            BashAction(command="git -c core.fileMode=false diff --exit-code --cached --no-color", check="ignore")
        ))
        print(f"Git diff result: '{git_diff_result.output}'")

        patch = git_diff_result.output
        normalized_patch = "\n".join(patch.splitlines())
        # add a new line to the patch
        normalized_patch = normalized_patch + "\n"
        return normalized_patch
        
    except Exception as e:
        print(f"ERROR in extract_git_diff_swerex_container: {e}")
        return ""

async def load_swe_instance_for_swerex(instance_id: str,checkout_commit: str | None = None) -> Tuple[DockerDeployment, str]:
    repo, name = instance_id.split('__')
    docker_image_name = f'swebench/sweb.eval.x86_64.{repo}_1776_{name}:latest'

    # Create a unique local directory to map into the container
    tmp_folder_name = str(uuid.uuid4())[:8]
    docker_map_path = os.path.join(DOCKER_MAP_DIR, tmp_folder_name)
    os.makedirs(docker_map_path, exist_ok=True)
    print(f"docker_map_path: {docker_map_path}")
    # Prepare docker_args for volume mapping
    docker_args = [
        "-v", f"{docker_map_path}:/docker_map"
    ]
    # You can add more docker_args as needed, e.g. user, etc.

    config = DockerDeploymentConfig(
        image=docker_image_name,
        docker_args=docker_args,
        python_standalone_dir=None
        # ...add other config fields as needed
    )
    deployment = config.get_deployment()
    await deployment.start()
    
    swe_rex_runtime = deployment.runtime

    await swe_rex_runtime.create_session(CreateBashSessionRequest())

    if checkout_commit:
        print(await swe_rex_runtime.run_in_session(BashAction(command=f"git checkout {checkout_commit}", check="ignore")))

    print(await swe_rex_runtime.run_in_session(BashAction(command="git config user.name 'Temp User' && git config user.email 'temp@example.com' && git commit -am 'swe-bench-extra'", check="ignore")))
    print(await swe_rex_runtime.run_in_session(BashAction(command="mv /testbed/ /docker_map/")))
    print(await swe_rex_runtime.run_in_session(BashAction(command="chmod -R 777 /docker_map/testbed")))
    print(await swe_rex_runtime.run_in_session(BashAction(command="ln -s /docker_map/testbed /testbed")))
    print(await swe_rex_runtime.run_in_session(BashAction(command="cd /testbed")))
    
    project_path = os.path.join(docker_map_path, "testbed")
    
    # Ensure a pristine copy in repo_map (for wiki)
    repo_map_path = os.path.join(REPO_MAP_DIR, f"{instance_id}")
    docker_map_repo_path = os.path.join(docker_map_path, "testbed")
    if not os.path.exists(repo_map_path):
        print(f"Copying pristine copy of {instance_id} to {repo_map_path} for wiki")
        shutil.copytree(docker_map_repo_path, repo_map_path)
    
    print(f"Project path: {project_path}")
    return deployment, project_path


    