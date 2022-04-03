cmake_minimum_required(VERSION 3.10)

project(text_detection_postprocess)

find_package(InferenceEngine REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui)
find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs "${PYTHON_VERSION_STRING}" EXACT REQUIRED)
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND
)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

set(CMAKE_CPP_FLAGS "-ggdb -D_DEBUG")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(target_name text_detection_postprocess)
add_library(${target_name} MODULE text_detection_postprocess.cpp postprocess.cpp)

target_include_directories(${target_name} PRIVATE src/ ${InferenceEngine_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(${target_name} ${PYTHON_LIBRARIES} ${InferenceEngine_LIBRARIES} opencv_core opencv_imgproc opencv_highgui)
set_target_properties(${target_name} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
endif()
