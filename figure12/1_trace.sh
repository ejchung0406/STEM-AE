#!/usr/bin/bash

uv run gdown 1ZBKIuVOSNBmArERGVXr1DxntkoZEcLOL

mkdir -p trace_nvbit

tar -xvf trace.tar.gz -C trace_nvbit

rm trace.tar.gz