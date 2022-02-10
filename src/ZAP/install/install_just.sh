#!/bin/bash
# https://github.com/casey/just

DESTINATION_DIRECTORY=$HOME/.just

curl -LSfs https://japaric.github.io/trust/install.sh | \
  sh -s -- --git casey/just --target x86_64-unknown-linux-musl --to $DESTINATION_DIRECTORY
