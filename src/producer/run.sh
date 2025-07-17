#!/bin/bash

export PYTHONPATH=$(pwd)/src/producer

python3 generate_schema.py
python3 produce_avro_schema.py 