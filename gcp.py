import argparse
import os
import subprocess
from dataclasses import asdict, dataclass, fields
from typing import Literal, get_args


@dataclass
class Config:
    cluster_name: str = "gke-dist"
    num_nodes: int = 2
    machine_type: str = "n1-standard-4"
    disk_size: int = 100
    disk_type: str = "pd-ssd"
    accelerator_type: str = "nvidia-tesla-t4"
    accelerator_count: int = 2
    container_registry: str = "gcr.io"
    gcp_project_id: str = "wandb-client-cicd"
    image_name: str = "sh22-dist-gpu"
    python_version: str = "3.8"
    git_branch: str = "main"
    pod_config: str = "pod.yaml"


Action = Literal[
    "update-components",
    "create-cluster",
    "install-gpu-drivers",
    "get-credentials",
    "delete-cluster",
    "build-image",
    "start-pods",
    "delete-pods",
    "noop",
    "e2e",  # run all steps in a meta way
]


def main(action: Action, config: Config):
    """

    :param action:
    :param config:
    :return:
    """
    """
    # setup gcloud:
    # generate new key if you don't have one:
    https://console.cloud.google.com/iam-admin/serviceaccounts/details/100090846244674923262/keys
    gcloud auth activate-service-account --key-file=${HOME}/gcloud-service-key.json
    # gcloud --quiet config set project $GOOGLE_PROJECT_ID
    # gcloud --quiet config set compute/zone $GOOGLE_COMPUTE_ZONE
    gcloud auth configure-docker --quiet << pipeline.parameters.container_registry >>
    """

    image_name = f"{config.container_registry}/{config.gcp_project_id}/{config.image_name}:latest"

    if action == "update-components":
        # update gcloud components
        subprocess.run(["gcloud", "--quiet", "components", "update"])
    elif action == "create-cluster":
        # create cluster gke-yea
        subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "create",
                config.cluster_name,
                f"--num-nodes={config.num_nodes}",
                f"--machine-type={config.machine_type}",
                f"--disk-size={config.disk_size}",
                f"--disk-type={config.disk_type}",
                f"--accelerator=type={config.accelerator_type},count={config.accelerator_count}",
            ]
        )
        # install GPU drivers
        subprocess.run(
            [
                "kubectl",
                "apply",
                "-f",
                "https://raw.githubusercontent.com/GoogleCloudPlatform"
                "/container-engine-accelerators/master/nvidia-driver-installer"
                "/cos/daemonset-preloaded-latest.yaml",
            ]
        )
    elif action == "install-gpu-drivers":
        # install GPU drivers
        subprocess.run(
            [
                "kubectl",
                "apply",
                "-f",
                "https://raw.githubusercontent.com/GoogleCloudPlatform"
                "/container-engine-accelerators/master/nvidia-driver-installer"
                "/cos/daemonset-preloaded-latest.yaml",
            ]
        )
    elif action == "get-credentials":
        # get credentials
        subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "get-credentials",
                config.cluster_name,
            ]
        )
    elif action == "delete-cluster":
        # delete cluster
        subprocess.run(
            [
                "gcloud",
                "container",
                "clusters",
                "delete",
                config.cluster_name,
            ]
        )
    elif action == "build-image":
        # use buildx to build image
        # subprocess.run(["docker", "buildx", "create", "--use"])
        # build docker image
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                # "linux/amd64",
                "linux/amd64,linux/arm64",
                "--push",
                "-t",
                image_name,
                "--build-arg",
                f"PYTHON_VERSION={config.python_version}",
                "--build-arg",
                f"GIT_BRANCH={config.git_branch}",
                ".",
            ]
        )
    elif action == "start-pods":
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key is not None:
            subprocess.run(
                [
                    "cp",
                    config.pod_config,
                    f"{config.pod_config}.tmp",
                ]
            )
            subprocess.run(
                [
                    "sed",
                    "-i",
                    "-e",
                    f"""s/WANDB_API_KEY_PLACEHOLDER/{api_key}/g""",
                    config.pod_config,
                ]
            )
        # spin up GPU pod
        subprocess.run(["kubectl", "apply", "-f", config.pod_config])
        if api_key is not None:
            try:
                subprocess.run(
                    [
                        "mv",
                        f"{config.pod_config}.tmp",
                        config.pod_config,
                    ]
                )
            except Exception as e:
                print(e)
    elif action == "delete-pods":
        subprocess.run(["kubectl", "delete", "-f", config.pod_config])
    elif action == "noop":
        pass


if __name__ == "__main__":
    actions = get_args(Action)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=actions, help="command to run")
    for field in fields(Config):
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=field.default,
            help=f"type: {field.type.__name__}; default: {field.default}",
        )

    args = vars(parser.parse_args())
    command = args.pop("command")
    main(action=command, config=Config(**args))
