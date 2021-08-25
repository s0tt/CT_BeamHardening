project('dl-beamhardening', ['cpp'], default_options : [ 'buildtype=release', 'cpp_std=c++14' ])


command = 'cd "$MESON_SOURCE_ROOT" && subprojects/git-submodules-update-careful'
res = run_command('sh', '-c', 'exec >&2 && ' + command)
if res.returncode() != 0
  error('Command "' + command + '" failed:\n' + res.stderr())
elif res.stderr() != ''
  message('Command "' + command + '" output:\n' + res.stderr())
endif

# use the apt libtorch-dev package
libtorch_include_path = '/usr/include'
libtorch_lib_path = '/usr/lib'

# use manual libtorch
#libtorch_path = '/your/absolut/path/libtorch'
#libtorch_include_path = libtorch_path + '/include'
#ibtorch_lib_path = libtorch_path + '/lib'

torch_dep = declare_dependency(
      include_directories : include_directories([
        #libtorch_include_path,
        libtorch_include_path +'/torch/csrc/api/include'
      ]),
      link_args : [
        #'-L' + join_paths(meson.current_source_dir(), libtorch_lib_path),
        #'-Wl, -rpath,' + join_paths(meson.current_source_dir(), libtorch_lib_path),
        '-ltorch',
        #'-lkineto',
        '-ltorch_cpu',
        '-lc10',
      ],
    )

voxie_proj = subproject('voxie-intern', default_options : [ 'libvoxie=disabled', 'main=disabled', 'plugins=disabled', 'ext=disabled', 'extra=disabled', 'test=disabled' ])
voxieclient_dep = voxie_proj.get_variable('voxieclient_dep')
voxiebackend_dep = voxie_proj.get_variable('voxiebackend_dep')
voxie_incdir = voxie_proj.get_variable('project_incdir') # Workaround for https://github.com/mesonbuild/meson/pull/8139
add_project_arguments('-DQT_NO_KEYWORDS', language : 'cpp')

subdir('src')
