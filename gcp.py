import argparse
import os
import subprocess
from dataclasses import dataclass
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
    "get-credentials",
    "delete-cluster",
    "build-image",
    "push-image",
    "start-pod",
    "delete-pod",
]


def main(action: Action, config: Config):
    """

    :param action:
    :param config:
    :return:
    """
    """
    # setup gcloud:
    echo $GCLOUD_SERVICE_KEY > ${HOME}/gcloud-service-key.json
    gcloud auth activate-service-account --key-file=${HOME}/gcloud-service-key.json
    gcloud --quiet config set project $GOOGLE_PROJECT_ID
    gcloud --quiet config set compute/zone $GOOGLE_COMPUTE_ZONE
    gcloud auth configure-docker --quiet << pipeline.parameters.container_registry >>
    """

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
        # build docker image
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                f"{config.container_registry}/{config.gcp_project_id}/{config.image_name}:latest",
                "--build-arg",
                f"PYTHON_VERSION={config.python_version}",
                "--build-arg",
                f"GIT_BRANCH={config.git_branch}",
            ]
        )
    elif action == "push-image":
        # push image
        subprocess.run(["docker", "push"])
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
    elif action == "start-pod":
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
    elif action == "delete-pod":
        subprocess.run(["kubectl", "delete", "-f", config.pod_config])


if __name__ == "__main__":
    actions = get_args(Action)
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command", title="action", description="Command to run"
    )

    parsers = dict()
    for a in actions:
        parsers[a] = subparsers.add_parser(a)

    # todo: add parameters to individual commands, add them to config
    args = parser.parse_args()
    conf = Config()
    main(action=args.command, config=conf)
