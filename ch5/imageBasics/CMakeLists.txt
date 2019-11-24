project(imageBasics)

add_executable(imageBasics imageBasics.cpp)
# 鏈接OpenCV庫
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(undistortImage undistortImage.cpp)
# 鏈接OpenCV庫
target_link_libraries(undistortImage ${OpenCV_LIBS})
