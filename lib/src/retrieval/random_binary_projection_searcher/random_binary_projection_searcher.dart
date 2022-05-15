import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/create_random_binary_projection_searcher.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// Random Binary Projection is a an algorithm that randomly partitions all
/// reference data points into different bins, which makes it possible to
/// perform efficient K Nearest Neighbours search, since there is no need to
/// search for the neighbours through the entire data: it's needed to
/// visit just a few bins to look for the neighbours.
///
/// Internally, the bins are represented by a Hash Map, where the key is an integer
/// index, and the value is a list of corresponding points. A record of the map
/// is a bin. Usually, points in the same bin are quite similar.
abstract class RandomBinaryProjectionSearcher with SerializableMixin {
  /// Takes [data], trains the model on it and returns [RandomBinaryProjectionSearcher] instance.
  ///
  /// Training means to distribute all points from the reference [data] by bins.
  /// A bin is
  ///
  /// Parameters:
  ///
  /// [digitCapacity] A number of bits of a bin index. Examples:
  ///
  /// - [digitCapacity] equals 2, possible binary bin indices: 01, 00, 11, 10
  ///
  /// - [digitCapacity] equals 3, possible binary bin indices: 000, 001, 010, ...
  ///
  /// - [digitCapacity] equals 4, possible binary bin indices: 0000, 0001, 0010, ...
  ///
  /// [seed] A seed value for the random generator which will be used to get bin indices
  ///
  /// [dtype] A data type for matrix representation of the [data]
  ///
  /// Example:
  ///
  /// ```dart
  /// import 'package:ml_algo/ml_algo.dart';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// void main() {
  ///   final data = DataFrame([
  ///     ['feature_1', 'feature_2', 'feature_3']
  ///     [10, 20, 30],
  ///     [11, 19, 31],
  ///     [19, 43, 20],
  ///     [89, 97, 11],
  ///     [12, 32, 10],
  ///   ]);
  ///   final digitCapacity = 3;
  ///
  ///   final searcher = RandomBinaryProjectionSearcher(data, digitCapacity, seed: 4);
  /// }
  /// ```
  factory RandomBinaryProjectionSearcher(DataFrame data, int digitCapacity,
          {int? seed, DType dtype = DType.float32}) =>
      createRandomBinaryProjectionSearcher(data, digitCapacity,
          seed: seed, dtype: dtype);

  /// Creates a [RandomBinaryProjectionSearcher] instance from a JSON-serializable
  /// object. The constructor works in conjunction with [saveAsJson] method. The
  /// latter serializes the instance and save the output to a JSON-file.
  ///
  /// Example:
  ///
  /// ```dart
  /// import 'dart:io';
  /// import 'package:ml_algo/ml_algo.dart';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// void main() async {
  ///   final data = DataFrame([
  ///     ['feature_1', 'feature_2', 'feature_3']
  ///     [10, 20, 30],
  ///     [11, 19, 31],
  ///     [19, 43, 20],
  ///     [89, 97, 11],
  ///     [12, 32, 10],
  ///   ]);
  ///   final digitCapacity = 3;
  ///
  ///   final searcher = RandomBinaryProjectionSearcher(data, digitCapacity, seed: 4);
  ///
  ///   await searcher.saveAsJson('path/to/json/file');
  ///
  ///   // ...
  ///
  ///   final file = File('path/to/json/file');
  ///   final jsonSource = await file.readAsString();
  ///
  ///   final restoredSearcher = RandomBinaryProjectionSearcher.fromJson(jsonSource);
  ///
  ///   print(searcher.columns);
  ///   // (feature_1, feature_2, feature_3)
  ///
  ///   print(searcher.points);
  ///   // Matrix 5 x 3:
  ///   // (10.0, 20.0, 30.0)
  ///   // (11.0, 19.0, 31.0)
  ///   // (19.0, 43.0, 20.0)
  ///   // (89.0, 97.0, 11.0)
  ///   // (12.0, 32.0, 10.0)
  ///
  ///   print(searcher.seed);
  ///   // 4
  ///
  ///   print(searcher.digitCapacity);
  ///   // 3
  /// }
  /// ```
  factory RandomBinaryProjectionSearcher.fromJson(String jsonSource) =
      RandomBinaryProjectionSearcherImpl.fromJson;

  /// A seed value for the random generator which was used to calculate bin indices
  int? get seed;

  /// A number of bits of a bin index. Examples:
  ///
  /// - [digitCapacity] equals 2, possible binary bin indices: 01, 00, 11, 10
  ///
  /// - [digitCapacity] equals 3, possible binary bin indices: 000, 001, 010, ...
  ///
  /// - [digitCapacity] equals 4, possible binary bin indices: 0000, 0001, 0010, ...
  int get digitCapacity;

  /// Column names of a dataset that was used to train the model
  ///
  /// Example:
  ///
  /// ```dart
  /// import 'package:ml_algo/ml_algo.dart';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// void main() {
  ///   final data = DataFrame([
  ///     ['feature_1', 'feature_2', 'feature_3']
  ///     [10, 20, 30],
  ///     [11, 19, 31],
  ///     [19, 43, 20],
  ///     [89, 97, 11],
  ///     [12, 32, 10],
  ///   ]);
  ///   final digitCapacity = 3;
  ///
  ///   final searcher = RandomBinaryProjectionSearcher(data, digitCapacity, seed: 4);
  ///
  ///   print(searcher.columns);
  ///   // (feature_1, feature_2, feature_3)
  /// }
  /// ```
  Iterable<String> get columns;

  /// Matrix representation of a dataset that was used to train the model
  ///
  /// Example:
  ///
  /// ```dart
  /// import 'package:ml_algo/ml_algo.dart';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// void main() {
  ///   final data = DataFrame([
  ///     ['feature_1', 'feature_2', 'feature_3']
  ///     [10, 20, 30],
  ///     [11, 19, 31],
  ///     [19, 43, 20],
  ///     [89, 97, 11],
  ///     [12, 32, 10],
  ///   ]);
  ///   final digitCapacity = 3;
  ///
  ///   final searcher = RandomBinaryProjectionSearcher(data, digitCapacity, seed: 4);
  ///
  ///   print(searcher.points);
  ///   // Matrix 5 x 3:
  ///   // (10.0, 20.0, 30.0)
  ///   // (11.0, 19.0, 31.0)
  ///   // (19.0, 43.0, 20.0)
  ///   // (89.0, 97.0, 11.0)
  ///   // (12.0, 32.0, 10.0)
  /// }
  /// ```
  Matrix get points;

  /// Accepts a [point] and finds it's [k] nearest neighbours. The search is
  /// performed along bins in [searchRadius] from the [point]'s bin. The greater
  /// [searchRadius] is, the more bins will be examined by the algorithm.
  ///
  /// A neighbour is represented by an index in the [points] matrix and the
  /// distance between the neighbour and the query [point]
  ///
  ///
  /// Example:
  ///
  /// ```dart
  /// import 'package:ml_algo/ml_algo.dart';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  /// import 'package:ml_linalg/vector.dart';
  ///
  /// void main() {
  ///   final data = DataFrame([
  ///     [10, 20, 30],
  ///     [11, 19, 31],
  ///     [19, 43, 20],
  ///     [89, 97, 11],
  ///     [12, 32, 10],
  ///   ], headerExists: false);
  ///   final digitCapacity = 3;
  ///
  ///   final searcher = RandomBinaryProjectionSearcher(data, digitCapacity, seed: 4);
  ///   final point = Vector.fromList([11, 19, 31]);
  ///
  ///   final k = 3; // we will search for 3 nearest neighbour
  ///   final searchRadius = 3;
  ///
  ///   final neighbours = searcher.query(point, k, searchRadius);
  ///
  ///   print(neighbours);
  ///   // ((Index: 1, Distance: 0.0), (Index: 0, Distance: 1.7320508075688772), (Index: 4, Distance: 24.71841418861655))
  ///   // To access a neighbour, refer to `searcher.points` by the neighbour index
  /// }
  /// ```
  Iterable<Neighbour> query(Vector point, int k, int searchRadius);
}
