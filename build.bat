mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config release
copy Release\text_detection_postprocess.pyd ..
cd ..
