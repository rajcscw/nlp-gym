#!/bin/bash

# list all tests here, so that we run one test file at a time
# since pytest's collection takes a lot of memory
pytest -v tests/envs/multi_label/test_env.py
pytest -v tests/envs/question_answering/test_env.py
#pytest -v tests/envs/sequence_tagging/test_env.py