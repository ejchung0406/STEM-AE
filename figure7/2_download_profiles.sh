#!/usr/bin/bash

# gdown TODO
uv run gdown 1w7Y74kHPOXiirZnpG79MzcWIOnUfaKm6
uv run gdown 1H6Rp19sDl9cSrPXQdqErZL1ANbF_0gqw

mkdir -p results_ncu-example
mkdir -p results_photon-example

tar -xzvf results_ncu.tar.gz -C results_ncu-example
tar -xzvf results_photon.tar.gz -C results_photon-example

rm results_ncu.tar.gz
rm results_photon.tar.gz