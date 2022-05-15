import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';

RandomBinaryProjectionSearcher createRandomBinaryProjectionSearcher(
    DataFrame dataset, int digitCapacity,
    {int? seed, DType dtype = DType.float32}) {
  final header = dataset.header;
  final matrix = dataset.toMatrix(dtype);

  return RandomBinaryProjectionSearcherImpl(header, matrix, digitCapacity,
      seed: seed);
}
