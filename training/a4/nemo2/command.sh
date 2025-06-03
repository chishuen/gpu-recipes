# helm uninstall justinpan-nemo-qwen

# : ${PROJECT_ID:=supercomputer-testing}
# : ${IMAGE:=us-central1-docker.pkg.dev/${PROJECT_ID}/${USER}-nemo/nemo2:0.2}

# docker build -t $IMAGE -f docker/Dockerfile .
# docker push $IMAGE

# helm install justinpan-nemo-qwen --set workload.image=$IMAGE helm_context/


export PROJECT=supercomputer-testing
export REGION=us-central1
export CLUSTER_NAME=gke-a4-eng
# export IMAGE=us-central1-docker.pkg.dev/supercomputer-testing/justinpan-nemo/nemo2:0.2
export IMAGE=gcr.io/supercomputer-testing/chishuen-nemo2:v0.0.2
export WORKLOAD_NAME=chishuen-8gpus-gbs128-tp4-pp2-cp1-vp2

gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

helm install $WORKLOAD_NAME --set workload.image=$IMAGE helm_context/