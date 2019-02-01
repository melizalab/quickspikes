#!/bin/bash
# This script is used to build manylinux wheels. It should be run in a docker container:
# docker run --rm -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /io/dev/build-wheels.sh
set -e -x

# Compile wheels
for PYBIN in /opt/python/{cp27*,cp3*}/bin; do
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/{cp27*,cp3*}/bin; do
    "${PYBIN}/pip" install quickspikes --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" -w /io/test/)
done
