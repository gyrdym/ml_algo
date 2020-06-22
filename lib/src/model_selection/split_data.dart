import 'package:ml_algo/src/model_selection/exception/empty_ratio_collection_exception.dart';
import 'package:ml_algo/src/model_selection/exception/invalid_ratio_sum_exception.dart';
import 'package:ml_algo/src/model_selection/exception/outranged_ratio_exception.dart';
import 'package:ml_algo/src/model_selection/exception/too_small_ratio_exception.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

List<DataFrame> splitData(DataFrame data, Iterable<double> ratios) {
  if (ratios.isEmpty) {
    throw EmptyRatioCollectionException();
  }

  final inputRows = data
      .rows
      .toList();
  var start = 0;
  var ratioSum = 0.0;

  return ratios
      .map((ratio) {
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

        final end = start + (rawSplitSize.ceil() == inputRows.length
            ? rawSplitSize.floor()
            : rawSplitSize.ceil());
        final rows = inputRows.sublist(start, end);

        start = end;

        return DataFrame(
            rows,
            headerExists: false,
            header: data.header,
        );
      })
      .toList()..add(DataFrame(
        inputRows.sublist(start),
        headerExists: false,
        header: data.header
      ));
}
