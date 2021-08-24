#include "PytorchModel.hpp"

PytorchModel::PytorchModel() {}

void PytorchModel::infere(
    vx::Array3<float const>& inputVolume, vx::Array3<float>& outputVolume,
    vx::ClaimedOperation<de::uni_stuttgart::Voxie::ExternalOperationRunFilter>&
        prog) {
  torch::Tensor tensor = torch::rand({2, 3});

  // Create a vector of inputs.

  std::vector<torch::jit::IValue> inputs;

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // TODO: make this generic, not with my absolut path
    module = torch::jit::load(
        "/home/so/Git/dl_beamhardening/models/checkpoints/cnn-ai-ct_trace.pt");
  } catch (const c10::Error& e) {
    qWarning() << "PytorchModel::infere error loading the model trace\n";
    return;
  }

  int nx = inputVolume.size<0>();
  int ny = inputVolume.size<1>();
  int nz = inputVolume.size<2>();

  auto data = inputVolume.data();

  // TODO remove const_cast and clone data instead
  torch::Tensor inputTensor =
      torch::from_blob(const_cast<float*>(data), {nx, ny, nz});

  int xmin, xmax, ymin, ymax, zmin, zmax, radius, xL, yL, zL;
  float dist, xDist, yDist, zDist;

  // iterate over volume in z direction
  for (int z = 2; z < nz - 3; z++) {
    auto sample =
        inputTensor
            .index({torch::indexing::Slice(), torch::indexing::Slice(),
                    torch::indexing::Slice(z - 2, z + 3)})
            .contiguous();

    // auto sample = inputTensor.slice(2, z, z + 4)

    inputs.push_back(sample);
  }

  // Execute the model and turn its output into a tensor.
  at::Tensor outputTensor = module.forward(inputs).toTensor();
  std::cout << outputTensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  nx = outputVolume.size<0>();
  ny = outputVolume.size<1>();
  nz = outputVolume.size<2>();

  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        outputVolume(x, y, z) = outputTensor[x][y][z].item<float>();
      }
    }
    HANDLEDBUSPENDINGREPLY(
        prog.opGen().SetProgress(((float)x) / nx, vx::emptyOptions()));
  }

  HANDLEDBUSPENDINGREPLY(prog.opGen().SetProgress(1.00, vx::emptyOptions()));
}
