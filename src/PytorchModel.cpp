#include "PytorchModel.hpp"

PytorchModel::PytorchModel() {}

void PytorchModel::computeNaive(
    vx::Array3<const float>& inputVolume, vx::Array3<float>& outputVolume,
    vx::ClaimedOperation<de::uni_stuttgart::Voxie::ExternalOperationRunFilter>&
        prog) {
  int nx = inputVolume.size<0>();
  int ny = inputVolume.size<1>();
  int nz = inputVolume.size<2>();
  int xmin, xmax, ymin, ymax, zmin, zmax, radius;
  float dist, xDist, yDist, zDist;
  for (int x = 0; x < nx; x++) {
    HANDLEDBUSPENDINGREPLY(
        prog.opGen().SetProgress(((float)x) / nx, vx::emptyOptions()));
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        dist = inputVolume(x, y, z);
        radius = ceil(dist) - 1;
        if (dist > 0) {
          xmin = std::max((x - radius), 0);
          xmax = std::min((x + radius), nx - 1);
          ymin = std::max((y - radius), 0);
          ymax = std::min((y + radius), ny - 1);
          zmin = std::max((z - radius), 0);
          zmax = std::min((z + radius), nz - 1);
          for (int x2 = xmin; x2 <= xmax; x2++) {
            xDist = (x - x2) * (x - x2);
            for (int y2 = ymin; y2 <= ymax; y2++) {
              yDist = (y - y2) * (y - y2);
              for (int z2 = zmin; z2 <= zmax; z2++) {
                zDist = (z - z2) * (z - z2);
                if (sqrt(xDist + yDist + zDist) <= dist) {
                  outputVolume(x2, y2, z2) =
                      std::max(outputVolume(x2, y2, z2), dist);
                }
              }
            }
          }
        }
      }
    }
  }
  HANDLEDBUSPENDINGREPLY(prog.opGen().SetProgress(1.00, vx::emptyOptions()));
}

void PytorchModel::compute(
    vx::Array3<const float>& inputVolume, vx::Array3<float>& outputVolume,
    vx::ClaimedOperation<de::uni_stuttgart::Voxie::ExternalOperationRunFilter>&
        prog) {
  int nx = inputVolume.size<0>();
  int ny = inputVolume.size<1>();
  int nz = inputVolume.size<2>();
  bool* isProcessed = new bool[nx * ny * nz];
  for (int i = 0; i < nx * ny * nz; i++) {
    isProcessed[i] = false;
  }
  int xmin, xmax, ymin, ymax, zmin, zmax, radius, xL, yL, zL;
  float dist, xDist, yDist, zDist;
  for (int x = 0; x < nx; x++) {
    HANDLEDBUSPENDINGREPLY(
        prog.opGen().SetProgress(((float)x) / nx, vx::emptyOptions()));
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
        if (inputVolume(x, y, z) > 0) {
          while (!(isProcessed[x + y * nx + z * nx * ny])) {
            xL = x;
            yL = y;
            zL = z;

            // find index of next unprocessed local max
            // while local max not found
            while (true) {
              xmin = std::max((xL - 1), 0);
              xmax = std::min((xL + 1), nx - 1);
              ymin = std::max((yL - 1), 0);
              ymax = std::min((yL + 1), ny - 1);
              zmin = std::max((zL - 1), 0);
              zmax = std::min((zL + 1), nz - 1);
              if ((inputVolume(xL, yL, zL) < inputVolume(xmin, yL, zL)) &&
                  !(isProcessed[xmin + yL * nx + zL * nx * ny])) {
                xL = xmin;
              } else if ((inputVolume(xL, yL, zL) <
                          inputVolume(xmax, yL, zL)) &&
                         !(isProcessed[xmax + yL * nx + zL * nx * ny])) {
                xL = xmax;
              } else if ((inputVolume(xL, yL, zL) <
                          inputVolume(xL, ymin, zL)) &&
                         !(isProcessed[xL + ymin * nx + zL * nx * ny])) {
                yL = ymin;
              } else if ((inputVolume(xL, yL, zL) <
                          inputVolume(xL, ymax, zL)) &&
                         !(isProcessed[xL + ymax * nx + zL * nx * ny])) {
                yL = ymax;
              } else if ((inputVolume(xL, yL, zL) <
                          inputVolume(xL, yL, zmin)) &&
                         !(isProcessed[xL + yL * nx + zmin * nx * ny])) {
                zL = zmin;
              } else if ((inputVolume(xL, yL, zL) <
                          inputVolume(xL, yL, zmax)) &&
                         !(isProcessed[xL + yL * nx + zmax * nx * ny])) {
                zL = zmax;
              } else {
                // local max found
                break;
              }
            }

            // calculate local thickness in sphere
            dist = inputVolume(xL, yL, zL);
            radius = ceil(dist) - 1;
            xmin = std::max((xL - radius), 0);
            xmax = std::min((xL + radius), nx - 1);
            ymin = std::max((yL - radius), 0);
            ymax = std::min((yL + radius), ny - 1);
            zmin = std::max((zL - radius), 0);
            zmax = std::min((zL + radius), nz - 1);
            for (int x2 = xmin; x2 <= xmax; x2++) {
              xDist = (xL - x2) * (xL - x2);
              for (int y2 = ymin; y2 <= ymax; y2++) {
                yDist = (yL - y2) * (yL - y2);
                for (int z2 = zmin; z2 <= zmax; z2++) {
                  zDist = (zL - z2) * (zL - z2);
                  if (sqrt(xDist + yDist + zDist) <= dist) {  // in sphere?
                    outputVolume(x2, y2, z2) =
                        std::max(outputVolume(x2, y2, z2), dist);
                    if (ceil(dist - sqrt(xDist + yDist + zDist)) >=
                        ceil(inputVolume(x2, y2, z2))) {
                      // is the smaller sphere totally contained here?
                      isProcessed[x2 + y2 * nx + z2 * nx * ny] = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  delete[] isProcessed;
  HANDLEDBUSPENDINGREPLY(prog.opGen().SetProgress(1.00, vx::emptyOptions()));
}
