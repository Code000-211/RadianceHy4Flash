# The kernel-library override (rad_add_kernel_override, cmake/RadAddPlugin.cmake) on the host: a
# library of three rows over two units, and overrides of it made of one rebuilt unit, for
# tests/kernel_override_test.cpp. Each file lands in a home of its own under
# $RADIANCE_HOME/testoverride, never in a kernels/ the engine scans, and none is installed.
#
#   <base>/kernels/radtest_ov<base>.so   the library                 (every answer 1xx)
#   <base>-over/kernels/...              the same name, unit A rebuilt (unit A's answers 2xx)
#   <base>-none/kernels/...              a unit that defines no row's hook: refused at open
#
# Twice, because a library reaches its units' functions in one of two ways and the override has to
# serve both: `plain` leaves its references preemptible, as a library linked with nothing special
# does (libr4d), so the dynamic linker binds them to the override that brought the library in;
# `sym` is linked -Bsymbolic, which binds them inside the library, so only the override's own
# replacement of the rows' hooks moves them.
set(RAD_PLUGIN_INSTALL OFF)
set(OVT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OVT_HOME ${RAD_PLUGIN_OUT}/testoverride)

foreach(kind plain sym)
  set(base radtest_ov${kind})
  rad_add_plugin(${base} testoverride/${kind}/kernels
                 SOURCES ${OVT_DIR}/ovt_rows.cpp ${OVT_DIR}/ovt_unit_a.cpp
                         ${OVT_DIR}/ovt_unit_b.cpp)
  target_compile_definitions(${base} PRIVATE OVT_NAME="${base}")
  if(kind STREQUAL "sym")
    target_link_options(${base} PRIVATE "LINKER:-Bsymbolic")
  endif()

  rad_add_kernel_override(${base}_over BASE ${base} SOURCES ${OVT_DIR}/ovt_unit_a.cpp
                          HOME ${OVT_HOME}/${kind}-over)
  target_compile_definitions(${base}_over PRIVATE OVT_VARIANT=2)

  rad_add_kernel_override(${base}_none BASE ${base} SOURCES ${OVT_DIR}/ovt_unit_none.cpp
                          HOME ${OVT_HOME}/${kind}-none)
endforeach()

add_dependencies(kernel_override_test radtest_ovplain radtest_ovplain_over radtest_ovplain_none
                                      radtest_ovsym radtest_ovsym_over radtest_ovsym_none)
