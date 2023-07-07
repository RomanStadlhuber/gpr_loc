#!/bin/bash

echo "installing the Open3D development package"
echo "see: http://www.open3d.org/docs/latest/getting_started.html#development-version-pip"
echo "please make sure you have pip >=23.1.1 installed"
echo "if this is not the case, run pip install --upgrade pip"
pip install --trusted-host www.open3d.org -f http://www.open3d.org/docs/latest/getting_started.html open3d
