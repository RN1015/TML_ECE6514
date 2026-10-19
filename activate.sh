#!/bin/bash
cd "$HOME/reasoning-sft-gh200" || return 1
source .venv/bin/activate
export HF_HOME="$PWD/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$PWD/cache/xdg"
