import 'package:ml_algo/src/model_selection/exception/empty_ratio_collection_exception.dart';
import 'package:ml_algo/src/model_selection/exception/invalid_ratio_sum_exception.dart';
import 'package:ml_algo/src/model_selection/exception/outranged_ratio_exception.dart';
import 'package:ml_algo/src/model_selection/exception/too_small_ratio_exception.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

/// Splits the given [data] into parts depending on [ratios]. The number of the
/// produced parts can be described as `ratios.length + 1`
///
/// Each ratio value must be a double number within the range 0..1 (both
/// exclusive), it can be interpreted as a percentage of the total amount of the
/// [data]'s rows.
///
/// A ratio value should not be too small - it should not produce a split of
/// the size less than 1 row.
///
/// Sum of all passed ratios must be less than 1.
///
/// In case of violation of any of the rules written above appropriate
/// exceptions will be thrown.
///
/// An example:
///
/// ````Dart
/// final source = [
///     ['feature_1', 'feature_3', 'feature_3'],
///     [     100.00,        null,      200.33],
///     [      -2221,        1002,       70009],
///     [       9008,       10006,        null],
///     [       7888,       10002,      300918],
///     [     500981,       29918,     5008.55],
///   ];
///   final data = DataFrame(source);
///   final splits = splitData(data, [0.2, 0.3]);
///
///   print(splits[0].header); // ('feature_1', 'feature_3', 'feature_3')
///   print(splits[0].rows); // [100.00, null, 200.33],
///
///   print(splits[1].header); // ('feature_1', 'feature_3', 'feature_3')
///   print(splits[1].rows); // [
///                          //   [-2221,  1002, 70009]
///                          //   [ 9008, 10006,  null],
///                          // ]
///
///   print(splits[2].header); // ('feature_1', 'feature_3', 'feature_3')
///   print(splits[2].rows); // [
///                          //   [ 7888, 10002,  300918]
///                          //   [500981, 29918, 5008.55],
///                          // ]
/// ````
List<DataFrame> splitData(DataFrame data, Iterable<double> ratios) {
  if (ratios.isEmpty) {
    throw EmptyRatioCollectionException();
  }

  final inputRows = data.rows.toList();
  var start = 0;
  var ratioSum = 0.0;

  return ratios.map((ratio) {
    if (ratio <= 0 || ratio >= 1) {
      throw OutRangedRatioException(ratio);
    }

    ratioSum += ratio;

    if (ratioSum >= 1) {
      throw InvalidRatioSumException();
    }

    final rawSplitSize = inputRows.length * ratio;

    if (rawSplitSize < 1) {
      throw TooSmallRatioException(ratio, inputRows.length);
    }

    final end = start +
        (rawSplitSize.ceil() == inputRows.length
            ? rawSplitSize.floor()
            : rawSplitSize.ceil());
    final rows = inputRows.sublist(start, end);

    start = end;

    return DataFrame(
      rows,
      headerExists: false,
      header: data.header,
    );
  }).toList()
    ..add(DataFrame(inputRows.sublist(start),
        headerExists: false, header: data.header));
}
