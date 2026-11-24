# The kernel-library override (rad_add_kernel_override, cmake/RadAddPlugin.cmake) on the host: a
# library of three rows over two units, and overrides of it made of one rebuilt unit, for
# tests/kernel_override_test.cpp. Each file lands in a home of its own under
# $RADIANCE_HOME/testoverride, never in a kernels/ the engine scans, and none is installed.
#
#   <base>/kernels/radtest_ov<base>.so   the library                 (every answer 1xx)
#   <base>-over/kernels/...              the same name, unit A rebuilt (unit A's answers 2xx)
#   <base>-none/kernels/...              a unit that defines no row's hook: refused at open
#   <base>-over3/kernels/...             unit A rebuilt again (answers 3xx): the next build a reload
#                                        takes (kernel_reload_test.cpp)
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

  rad_add_kernel_override(${base}_over3 BASE ${base} SOURCES ${OVT_DIR}/ovt_unit_a.cpp
                          HOME ${OVT_HOME}/${kind}-over3)
  target_compile_definitions(${base}_over3 PRIVATE OVT_VARIANT=3)
endforeach()

add_dependencies(kernel_override_test radtest_ovplain radtest_ovplain_over radtest_ovplain_none
                                      radtest_ovsym radtest_ovsym_over radtest_ovsym_none)
if(TARGET kernel_reload_test)
  add_dependencies(kernel_reload_test radtest_ovplain radtest_ovplain_over radtest_ovplain_none
                                      radtest_ovplain_over3 radtest_ovsym radtest_ovsym_over
                                      radtest_ovsym_none radtest_ovsym_over3)
endif()

# kernel_reload_e2e_test's library, for a running engine: one host row for `sample_argmax` that
# gives every row a fixed token (rl_rows.cpp, rl_unit.cpp), and overrides of it made of rl_unit.cpp
# built four ways, each in a home of its own under $RADIANCE_HOME/testoverride:
#
#   rl/kernels/radtest_rl.so   the library (token 3)
#   rl-a/, rl-b/               the unit rebuilt to give token 5, and token 9
#   rl-scratch/                token 9 with another scratch size: a reload is refused
#   rl-init/                   token 9 with the row's init defined too: a reload is refused
rad_add_plugin(radtest_rl testoverride/rl/kernels
               SOURCES ${OVT_DIR}/rl_rows.cpp ${OVT_DIR}/rl_unit.cpp)
set(RL_VARIANTS "a|RL_TOKEN=5" "b|RL_TOKEN=9" "scratch|RL_TOKEN=9,RL_SCRATCH=128"
                "init|RL_TOKEN=9,RL_INIT=1")
set(RL_TARGETS radtest_rl)
foreach(v ${RL_VARIANTS})
  string(REGEX REPLACE "[|,]" ";" v "${v}")
  list(POP_FRONT v tag)
  rad_add_kernel_override(radtest_rl_${tag} BASE radtest_rl SOURCES ${OVT_DIR}/rl_unit.cpp
                          HOME ${OVT_HOME}/rl-${tag})
  target_compile_definitions(radtest_rl_${tag} PRIVATE ${v})
  list(APPEND RL_TARGETS radtest_rl_${tag})
endforeach()
if(TARGET kernel_reload_e2e_test)
  add_dependencies(kernel_reload_e2e_test ${RL_TARGETS})
endif()
