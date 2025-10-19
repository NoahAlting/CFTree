#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/IO/PLY.h>  // write_PLY

#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

using K     = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = K::Point_3;
using Mesh  = CGAL::Surface_mesh<K::Point_3>;

static bool read_xyz(const std::string& path, std::vector<Point>& pts) {
  std::ifstream fin(path);
  if (!fin) return false;
  double x, y, z;
  // Read triples (skips comments/blank lines automatically if formatted)
  while (fin >> x >> y >> z) {
    pts.emplace_back(x, y, z);
  }
  return !pts.empty();
}

int main(int argc, char** argv){
  if(argc < 2){
    std::cerr << "Usage: awrap_points <input.xyz> [ralpha=20] [roffset=50] [out.ply|-]\n";
    return 2;
  }
  const std::string in  = argv[1];
  const double ralpha   = (argc > 2) ? std::stod(argv[2]) : 15.0;
  const double roffset  = (argc > 3) ? std::stod(argv[3]) : 50.0;
  const std::string out = (argc > 4) ? argv[4] : "-";

  std::vector<Point> pts;
  if(!read_xyz(in, pts)){
    std::cerr << "Failed to read XYZ points from " << in << "\n";
    return 3;
  }

  auto bb = CGAL::bbox_3(pts.begin(), pts.end());
  const double dx = bb.xmax() - bb.xmin();
  const double dy = bb.ymax() - bb.ymin();
  const double dz = bb.zmax() - bb.zmin();
  const double diag   = std::sqrt(dx*dx + dy*dy + dz*dz);
  const double alpha  = diag / ralpha;
  const double offset = alpha / roffset;

  Mesh wrap;
  CGAL::alpha_wrap_3(pts, alpha, offset, wrap);

  if(out == "-"){
    CGAL::IO::set_binary_mode(std::cout);
    CGAL::IO::write_PLY(std::cout, wrap);
  } else {
    std::ofstream fout(out, std::ios::binary);
    if(!fout){
      std::cerr << "Cannot open output file: " << out << "\n";
      return 5;
    }
    CGAL::IO::write_PLY(fout, wrap);
    fout.close();
  }
  return 0;
}
