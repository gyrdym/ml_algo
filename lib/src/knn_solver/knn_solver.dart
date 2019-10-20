import 'package:ml_algo/src/knn_solver/neigbour.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

abstract class KnnSolver {
  Matrix get trainFeatures;

  Matrix get trainOutcomes;

  int get k;

  Distance get distanceType;

  bool get standardize;

  /// Finds [k] nearest neighbours for either record in [features]
  /// basing on [trainFeatures] and [trainOutcomes]
  Iterable<Iterable<Neighbour<Vector>>> findKNeighbours(Matrix features);
}
