import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

abstract class RandomBinaryProjectionSearcher {
  factory RandomBinaryProjectionSearcher(DataFrame data, int digitCapacity,
      {int seed, DType dtype}) = RandomBinaryProjectionSearcherImpl;

  Iterable<Neighbour> query(Vector point, int k, int searchRadius);
}
