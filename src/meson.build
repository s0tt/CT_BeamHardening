name = 'ExtFilterPytorchModel'

if meson.get_compiler('cpp').get_argument_syntax() != 'msvc'
  add_project_arguments('-Wextra', language : 'cpp')
  add_project_arguments('-Wno-non-virtual-dtor', language : 'cpp')

  #add_project_arguments('-Wsuggest-override', language : 'cpp')
  add_project_arguments('-Wno-unknown-warning-option', language : 'cpp')
endif

qt5 = import('qt5')
project_incdir = include_directories('.')

# General stuff
#thread_dep = dependency('threads')
rt_dep = meson.get_compiler('cpp').find_library('rt', required : false)

qt_modules = [
  'Core',
  'DBus',
  'Gui',
  'Widgets',
  'OpenGL', # TODO: Remove
]
# Note: moc does not like "include_type:'system'" because meson only considers -I and -D for moc, not -isystem: <https://github.com/mesonbuild/meson/blob/398df5629863e913fa603cbf02c525a9f501f8a8/mesonbuild/modules/qt.py#L193>. See also <https://github.com/mesonbuild/meson/pull/8139>. Note that currently voxie should compile even without this workaround, see update-dbus-proxies.sh.
qt5_dep = dependency('qt5', modules : qt_modules, include_type : 'system')
qt5_dep_moc = dependency('qt5', modules : qt_modules)

moc_files = qt5.preprocess(
  moc_sources : [
  ],
  moc_headers : [
    'CNNAiCt.hpp',

  ],
  include_directories : [
    project_incdir,
    voxie_incdir, # Workaround for https://github.com/mesonbuild/meson/pull/8139
  ],
  dependencies : [ qt5_dep_moc],
)

main = executable(
  name,
  [
    moc_files,
    'ExtFilterPytorchModel.cpp',
    'CNNAiCt.cpp',
  ],
  include_directories : [
    project_incdir,
  ],
  link_args : [
    '-lGL',
    '-lGLU',
  ],
  implicit_include_directories : false,
  install : true,
  dependencies : [
    torch_dep,
    voxiebackend_dep,
    voxieclient_dep,
    qt5_dep,
  ],
  #cpp_args : [ '-D_GLIBCXX_USE_CXX11_ABI=0' ],
  #build_rpath : join_paths(meson.current_source_dir(), libtorch_lib_path),
)
configure_file(
  input : name + '.json',
  output : name + '.json',
  copy : true,
)
install_data(name + '.json')
