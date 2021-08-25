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
      torch::from_blob(const_cast<float*>(data), {nx, ny, nz});

  qDebug() << "InputTensor Size:" << inputTensor.dim();
  qDebug() << "inputTensor Size:" << inputTensor.size(0) << ","
           << inputTensor.size(1) << "," << inputTensor.size(2);

  int xmin, xmax, ymin, ymax, zmin, zmax, radius, xL, yL, zL;
  float dist, xDist, yDist, zDist;

  std::vector<torch::Tensor> batchList;
  std::vector<int> indices;
  torch::TensorList tensors;

  // iterate over volume in x direction
  // TODO: handle borders where we can't get 5 neighbour slices
  for (int x = 2; x < nx - 3; x++) {
    auto sample =
        inputTensor
            .index({torch::indexing::Slice(x - 2, x + 3),
                    torch::indexing::Slice(), torch::indexing::Slice()})
            .unsqueeze(0);
    batchList.push_back(sample);
    indices.push_back(x);

    qDebug() << "Sample Size:" << sample.size(0) << "," << sample.size(1) << ","
             << sample.size(2) << "," << sample.size(3);

    // check if enough samples for specified batch size
    if (batchList.size() == batchSize) {
      // cat samples to 4-dim tensor with (sample_dim, slice_dim, y_dim, z_dim)
      auto batch = torch::cat({batchList});
      qDebug() << "Batch Size:" << batch.size(0) << "," << batch.size(1) << ","
               << batch.size(2) << "," << batch.size(3);

      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(batch);
      at::Tensor outputTensor = module.forward(inputs).toTensor();

      qDebug() << "OutputTensor Size:" << outputTensor.size(0) << ","
               << outputTensor.size(1) << "," << outputTensor.size(2) << ","
               << outputTensor.size(3);

      // write output tensor to voxie volume
      int outputIdx = 0;
      for (int inputIdx : indices) {
        for (int y = 0; y < ny; y++) {
          for (int z = 0; z < nz; z++) {
            outputVolume(inputIdx, y, z) =
                outputTensor[outputIdx][0][y][z].item<float>();
          }
        }
        outputIdx++;
      }
      indices.clear();
      inputs.clear();
      batchList.clear();
    }
    HANDLEDBUSPENDINGREPLY(
        prog.opGen().SetProgress(((float)x) / nx, vx::emptyOptions()));
  }

  HANDLEDBUSPENDINGREPLY(prog.opGen().SetProgress(1.00, vx::emptyOptions()));
}
