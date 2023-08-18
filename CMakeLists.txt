cmake_minimum_required(VERSION 3.10)
project(sam)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# ���ñ�������
set(CMAKE_BUILD_TYPE Debug)

# Find OpenCV
set(OpenCV_INCLUDE_DIRS "./opencv/include")
set(OpenCV_LIBS "../opencv/lib/opencv_world455d")

# Find ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIRS "./onnxruntime-win-x64-1.15.1/include")
set(ONNXRUNTIME_LIBRARIES "../onnxruntime-win-x64-1.15.1/lib/onnxruntime")

# ���ӿ�ִ���ļ�������Ĳ���·��
add_library(sam SHARED common.h SimpleTokenizer.h SimpleTokenizer.cpp CLIP.h CLIP.cpp  SAM.h SAM.cpp)
target_include_directories(sam PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(sam ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARIES})
target_compile_definitions(sam PRIVATE SAM_EXPORTS)

# ��������cppDemo��ִ���ļ�
add_executable(cppDemo common.h cppDemo.cpp)
target_include_directories(cppDemo PRIVATE ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(cppDemo
                      ${OpenCV_LIBS}
                      sam)
