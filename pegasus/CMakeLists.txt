target_sources(pegasus PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/lwe.cc
  ${CMAKE_CURRENT_LIST_DIR}/rlwe.cc
  ${CMAKE_CURRENT_LIST_DIR}/repack.cc
  ${CMAKE_CURRENT_LIST_DIR}/runtime.cc
  ${CMAKE_CURRENT_LIST_DIR}/gateboot.cc
  ${CMAKE_CURRENT_LIST_DIR}/pegasus_runtime.cc
  ${CMAKE_CURRENT_LIST_DIR}/rotgroup_helper.cc 
  ${CMAKE_CURRENT_LIST_DIR}/linear_transform.cc
  ${CMAKE_CURRENT_LIST_DIR}/chevb_approximator.cc
)
