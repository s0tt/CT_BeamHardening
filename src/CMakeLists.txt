cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ExtFilterPytorchModel)

find_package(Torch REQUIRED)

add_executable(ExtFilterPytorchModel ExtFilterPytorchModel.cpp PytorchModel.cpp)
target_link_libraries(ExtFilterPytorchModel "${TORCH_LIBRARIES}")
set_property(TARGET ExtFilterPytorchModel PROPERTY CXX_STANDARD 14)