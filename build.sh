#!/usr/bin/env /usr/bin/bash
mkdir -p build
pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config release
cp Release/text_detection_postprocess.* ..
popd
