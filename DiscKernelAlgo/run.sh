## Copyright 2020 Alexander Liniger

## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at

##     http://www.apache.org/licenses/LICENSE-2.0

## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
###########################################################################
###########################################################################
#!/bin/bash
for i in 0.1 0.05 0.04 0.03 0.02 0.01 0.005 0.004 0.003 0.002 0.0015 0.00125 0.001
do
  echo "Running Disc Kernel Algo for kappa_max = $i"
  ./jq-linux64 '.kappa_max = $newVal' --argjson newVal $i Model/car.json > tmp.$$.json && mv tmp.$$.json Model/car.json
  ./DiscKernelAlgo
done
