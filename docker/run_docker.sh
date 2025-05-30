#!/usr/bin/env bash
if [ -z ${TAG} ]; then
  echo "No tag given. Defaulting to latest"
  TAG="latest"
fi

IMAGE=dev

DOCKER_ARGS=(
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -e DOCKER_MACHINE_NAME="${IMAGE}:${TAG}"
  --network=host
  --ulimit core=99999999999:99999999999
  --ulimit nofile=1024
  --privileged
  --rm
  -e DISPLAY=$DISPLAY
  -e QT_X11_NO_MITSHM=1
  --ipc=host
)

# Parse --mac flag
MAC_MODE=0
for arg in "$@"; do
  if [ "$arg" == "--mac" ]; then
    MAC_MODE=1
    # Remove --mac from arguments
    set -- "${@/--mac/}"
    break
  fi
done

# Add GPU support if available
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected, enabling --gpus all"
  DOCKER_ARGS+=(--gpus all)
fi

if [ $MAC_MODE -eq 1 ]; then
  DOCKER_ARGS+=(
    -v /Users/$USER/workspace/:/home/$USER/workspace
  )
else
  DOCKER_ARGS+=(
    -v /home/$USER:/home/$USER
  )
fi

docker run "${DOCKER_ARGS[@]}" -it ${IMAGE}:${TAG}