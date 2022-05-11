import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/create_random_binary_projection_searcher.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

abstract class RandomBinaryProjectionSearcher {
  factory RandomBinaryProjectionSearcher(DataFrame data, int digitCapacity,
          {int? seed, DType dtype = DType.float32}) =>
      createRandomBinaryProjectionSearcher(data, digitCapacity,
          seed: seed, dtype: dtype);

  Iterable<Neighbour> query(Vector point, int k, int searchRadius);
}
