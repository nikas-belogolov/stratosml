set(STRATOSML_SRCS "core.hpp")
add_library(stratosml SHARED ${STRATOSML_SRCS})

# Collect all header files in the library.
file(GLOB_RECURSE INCLUDE_H_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
file(GLOB_RECURSE INCLUDE_HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)
set(INCLUDE_FILES ${INCLUDE_H_FILES} ${INCLUDE_HPP_FILES})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_custom_command(TARGET stratosml POST_BUILD
  COMMENT "Moving header files to include/stratosml/"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/stratosml/)

  add_custom_target(move_headers ALL
    COMMENT "Moving header files to include/stratosml"
)

foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(
    TARGET move_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
      ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file}
      ${CMAKE_BINARY_DIR}/include/stratosml/${incl_file}
      DEPENDS ${INCLUDE_FILES})
endforeach()

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/stratosml DESTINATION include)

install(TARGETS stratosml
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib)
