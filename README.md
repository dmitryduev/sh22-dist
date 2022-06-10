# Distributed training example using [Weights & Biases](https://wandb.ai)

This tutorial shows how to use Weights & Biases to train a model on a distributed cluster.
We will spin up a GKE cluster and deploy two pods with two GPUs for distributed training.

## The `gcp.py` utility

Use the `gcp.py` utility to manage the GKE cluster, pods etc.

```shell
usage: gcp.py [-h] [--cluster_name CLUSTER_NAME] [--num_nodes NUM_NODES]
              [--machine_type MACHINE_TYPE] [--disk_size DISK_SIZE]
              [--disk_type DISK_TYPE] [--accelerator_type ACCELERATOR_TYPE]
              [--accelerator_count ACCELERATOR_COUNT]
              [--container_registry CONTAINER_REGISTRY]
              [--gcp_project_id GCP_PROJECT_ID] [--image_name IMAGE_NAME]
              [--python_version PYTHON_VERSION] [--git_branch GIT_BRANCH]
              [--pod_config POD_CONFIG]
              {update-components,create-cluster,install-gpu-drivers,get-credentials,delete-cluster,build-image,start-pods,delete-pod,noop}

positional arguments:
  {update-components,create-cluster,install-gpu-drivers,get-credentials,delete-cluster,build-image,start-pods,delete-pod,noop}
                        command to run

optional arguments:
  -h, --help            show this help message and exit
  --cluster_name CLUSTER_NAME
                        type: str; default: gke-dist
  --num_nodes NUM_NODES
                        type: int; default: 2
  --machine_type MACHINE_TYPE
                        type: str; default: n1-standard-4
  --disk_size DISK_SIZE
                        type: int; default: 100
  --disk_type DISK_TYPE
                        type: str; default: pd-ssd
  --accelerator_type ACCELERATOR_TYPE
                        type: str; default: nvidia-tesla-t4
  --accelerator_count ACCELERATOR_COUNT
                        type: int; default: 2
  --container_registry CONTAINER_REGISTRY
                        type: str; default: gcr.io
  --gcp_project_id GCP_PROJECT_ID
                        type: str; default: wandb-client-cicd
  --image_name IMAGE_NAME
                        type: str; default: sh22-dist-gpu
  --python_version PYTHON_VERSION
                        type: str; default: 3.8
  --git_branch GIT_BRANCH
                        type: str; default: main
  --pod_config POD_CONFIG
                        type: str; default: pod.yaml
```


## Setup instructions

- Install and setup gcloud: https://cloud.google.com/sdk/docs/install

- Update gcloud to the latest version:
```shell
python gcp.py update-components
```

- Generate a new service account key if you don't have one:
https://console.cloud.google.com/iam-admin/serviceaccounts/details/100090846244674923262/keys

Save it as, e.g. `gcloud-service-key.json` and tell gcloud to use it:
```shell
gcloud auth activate-service-account --key-file=gcloud-service-key.json
```

- Configure `gcloud` to use Google's container registry:
```shell
gcloud auth configure-docker --quiet gcr.io
```

- Build the Docker image that we will use to create `kubernetes` pods:
```shell
python gcp.py build-image --image_name=sh22-dist-gpu --python_version=3.8 --git_branch=main
```
This command uses `buildx` to build the image, which is targeted at both `x86` and `arm` platforms.
Once complete, the image will be available in the `gcr.io` container registry.

- Create a cluster with the following command:
```shell
python gcp.py create-cluster
```
This command will also install the GPU drivers.

- Point `gcloud` to the cluster:
```shell
python gcp.py get-credentials
```

- Deploy the pods with the following command:
```shell
export WANDB_API_KEY=<your-api-key>
python gcp.py start-pods
```
It takes about a minute to start the pods.

## Running training

- Once the pods are running, you can `ssh` into them.
Get the pod names with the following command:
```shell
kubectl get pods
```

Then you can `ssh` into the pod with the following command:
```shell
kubectl exec -it <POD_NAME> -- /bin/bash
```

Once you are in the pods, you are ready to train your very own model in a
multi-node, multi-GPU fashion :sunglasses:.

- Download TinyImageNet:
```shell
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

- On the first node, run:
```shell
python -m torch.distributed.launch \
  --nnodes=2 --node_rank=0 --nproc_per_node=2 \
  --master_addr=<IP_ON_THE_LOCAL_NETWORK> --master_port=<FREE_PORT> \
  --wandb_run_group=go-sdk-1 \
  main.py /wandb/sh22-dist/tiny-imagenet-200/
```

- On the second node, run:
```shell
python -m torch.distributed.launch \
  --nnodes=2 --node_rank=1 --nproc_per_node=2 \
  --master_addr=<IP_OF_THE_FIRST_NODE_ON_THE_LOCAL_NETWORK> --master_port=<FREE_PORT> \
  --wandb_run_group=go-sdk-1 \
  main.py /wandb/sh22-dist/tiny-imagenet-200/
```

You can monitor how your training is going with the following command:
```shell
nvtop
```

## Cleanup

Once you are done, you can use the `gcp.py` utility to delete the pods and cluster:
```shell
python gcp.py delete-pods
python gcp.py delete-cluster
```
