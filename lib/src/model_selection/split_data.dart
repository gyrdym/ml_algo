import 'package:ml_dataframe/ml_dataframe.dart';

List<DataFrame> splitData(DataFrame data, Iterable<num> ratios) {
  if (ratios.isEmpty) {
    throw Exception('Ratio collection must contain at least one element');
  }

  final inputRows = data
      .rows
      .toList();
  var start = 0;
  var ratioSum = 0.0;

  return ratios
      .map((ratio) {
        if (ratio <= 0 || ratio >= 1) {
          throw Exception('Ratio value must be within the range 0..1 (both '
              'exclusive), $ratio given');
        }

        ratioSum += ratio;

        if (ratioSum >= 1) {
          throw Exception('Ratios sum is more than or equal to 1');
        }

        final rawSplitSize = inputRows.length * ratio;

        if (rawSplitSize < 1) {
          throw Exception('Ratio is too small comparing to the input data size: '
              'ratio $ratio, min ratio value '
              '${(1 / inputRows.length).toStringAsFixed(2)}');
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
