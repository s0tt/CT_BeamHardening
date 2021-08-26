#include "PytorchModel.hpp"

PytorchModel::PytorchModel() {}

void PytorchModel::infere(
    vx::Array3<float const>& inputVolume, vx::Array3<float>& outputVolume,
    int batchSize,
    vx::ClaimedOperation<de::uni_stuttgart::Voxie::ExternalOperationRunFilter>&
        prog) {
  torch::Tensor tensor = torch::rand({2, 3});

  // Create a vector of inputs.

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    // TODO: make this generic, not with my absolut path
    module = torch::jit::load(
        "/home/so/Git/dl_beamhardening/models/checkpoints/cnn-ai-ct_trace.pt");
    //qDebug() << "WEIGHTS:" << module.startLayer.named_parameters()["weight"];
  } catch (const c10::Error& e) {
    qWarning() << "PytorchModel::infere error loading the model trace\n";
    return;
  }

  int nx = inputVolume.size<0>();
  int ny = inputVolume.size<1>();
  int nz = inputVolume.size<2>();

  auto data = inputVolume.data();

  // TODO remove const_cast and clone data instead
  // torch::Tensor inputTensor = torch::ones({nx, ny, nz});
  torch::Tensor inputTensor =
      torch::from_blob(const_cast<float*>(data), {nz, ny, nx});

  // transpose z and x to get back actual dimensions
  inputTensor = inputTensor.transpose(0, 2);

  std::vector<torch::Tensor> batchList;
  std::vector<int> indices;
  torch::TensorList tensors;

  // iterate over volume in x direction
  // TODO: handle borders where we can't get 5 neighbour slices
  for (int y = 2; y < ny - 3; y++) {
    auto sample = inputTensor
                      .index({torch::indexing::Slice(),
                              torch::indexing::Slice(y - 2, y + 3),
                              torch::indexing::Slice()})
                      .transpose(0, 1)
                      .unsqueeze(0);
    batchList.push_back(sample);
    indices.push_back(y);

    // check if enough samples for specified batch size
    if (batchList.size() == batchSize) {

      // cat samples to 4-dim tensor with (sample_dim, slice_dim, y_dim, z_dim)
      auto batch = torch::cat({batchList});

      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(batch);
      at::Tensor outputTensor = module.forward(inputs).toTensor();

      // write output tensor to voxie volume
      int outputIdx = 0;
      for (int inputIdx : indices) {
        for (int x = 0; x < nx; x++) {
          for (int z = 0; z < nz; z++) {
            outputVolume(x, inputIdx, z) =
                outputTensor[outputIdx][0][x][z].item<float>();
          }
        }
        outputIdx++;
      }
      indices.clear();
      inputs.clear();
      batchList.clear();
    }
    HANDLEDBUSPENDINGREPLY(
        prog.opGen().SetProgress(((float)y) / ny, vx::emptyOptions()));
  }

  HANDLEDBUSPENDINGREPLY(prog.opGen().SetProgress(1.00, vx::emptyOptions()));
}
