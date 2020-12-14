add_executable(lut_exe lut.cc)
target_link_libraries(lut_exe pegasus)

add_executable(repacking_exe repacking.cc)
target_link_libraries(repacking_exe pegasus)
