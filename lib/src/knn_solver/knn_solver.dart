import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class KnnSolver implements Serializable {
  /// Finds k nearest neighbours for either record in train feature matrix
  Iterable<Iterable<Neighbour<Vector>>> findKNeighbours(Matrix features);
}
