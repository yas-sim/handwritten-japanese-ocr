mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config release
: cmake .. -DCMAKE_BUILD_TYPE=Debug
: cmake --build . --config debug
copy Release\text_detection_postprocess.* ..
cd ..
