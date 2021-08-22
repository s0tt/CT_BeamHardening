project('dl-beamhardening', ['cpp'], default_options : [ 'buildtype=release', 'cpp_std=c++14' ])

command = 'cd "$MESON_SOURCE_ROOT" && subprojects/git-submodules-update-careful'
res = run_command('sh', '-c', 'exec >&2 && ' + command)
if res.returncode() != 0
  error('Command "' + command + '" failed:\n' + res.stderr())
elif res.stderr() != ''
  message('Command "' + command + '" output:\n' + res.stderr())
endif

#cpp = meson.get_compiler('cpp')
#myLib_dep = cpp.find_library('libtorch', dirs: '/usr/include/libtorch/', required: true)
torch_dep = dependency('torch')

voxie_proj = subproject('voxie-intern', default_options : [ 'libvoxie=disabled', 'main=disabled', 'plugins=disabled', 'ext=disabled', 'extra=disabled', 'test=disabled' ])
voxieclient_dep = voxie_proj.get_variable('voxieclient_dep')
voxiebackend_dep = voxie_proj.get_variable('voxiebackend_dep')
voxie_incdir = voxie_proj.get_variable('project_incdir') # Workaround for https://github.com/mesonbuild/meson/pull/8139

subdir('src')
